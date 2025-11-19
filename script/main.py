import device
import torch
from torch_geometric.data import Data
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from sklearn.metrics import precision_score, recall_score, f1_score
from model import GCN
import os
import sys
import clip
import ast
import numpy as np

dataset_name = 'pheme'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# base directory of this script
_script_dir = os.path.abspath(os.path.dirname(__file__))

# candidate locations to look for the pheme_analysis_results.csv
candidates = [
    os.path.join(_script_dir, 'dataset', dataset_name, 'pheme_analysis_results.csv'),      # script/dataset/...
    os.path.join(_script_dir, '..', 'dataset', dataset_name, 'pheme_analysis_results.csv'), # repo_root/dataset/...
    os.path.join(_script_dir, 'dataset', f'{dataset_name}_analysis_results.csv'),
    os.path.join(_script_dir, '..', 'dataset', f'{dataset_name}_analysis_results.csv'),
]

csv_path = None
for p in candidates:
    if os.path.exists(p):
        csv_path = p
        break

# 如果没找到，给出提示并创建占位文件（可选）
if csv_path is None:
    # 你可以把下面行为改为抛错而不是创建占位文件
    target_dir = os.path.join(_script_dir, 'dataset', dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    csv_path = os.path.join(target_dir, 'pheme_analysis_results.csv')
    # 创建一个小的占位 CSV（避免后续读取出错），提醒用户替换为真实数据
    pd.DataFrame({
        'id': [1],
        'analysis': [0],
        'prob': [0.0]
    }).to_csv(csv_path, index=False, encoding='utf-8')
    print(f"[Warning] 未在候选路径找到 pseusdo CSV，已在 {csv_path} 创建占位文件，请替换为真实数据。")

# 读取时尝试常见编码
encodings = ['utf-8', 'utf-8-sig', 'gbk', 'cp1252', 'latin1']
psesudo_data = None
last_exc = None
for enc in encodings:
    try:
        psesudo_data = pd.read_csv(csv_path, encoding=enc)
        break
    except UnicodeDecodeError as e:
        last_exc = e
if psesudo_data is None:
    raise UnicodeDecodeError(f"无法用常见编码读取 {csv_path}（最后一次错误: {last_exc}）。请确认文件存在且为 UTF-8/GBK 等常见编码。")

def parse_analysis_value(v):
    # 1) 缺失值
    if pd.isna(v):
        return 0
    # 2) 如果已经是数值类型
    if isinstance(v, (int, np.integer, float, np.floating)):
        return int(v)
    # 3) 如果是 bytes（极少见）
    if isinstance(v, (bytes, bytearray)):
        try:
            s = v.decode('utf-8', errors='ignore').strip()
            return int(float(s))
        except:
            pass
    # 4) 如果是字符串，尝试解析
    if isinstance(v, str):
        s = v.strip()
        # 尝试直接转数值
        try:
            return int(float(s))
        except:
            pass
        # 尝试 literal_eval（将类似 "[0]" 或 "['0']" 变为列表）
        try:
            val = ast.literal_eval(s)
        except Exception:
            val = None
        if val is not None:
            # 列表或元组
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float)
                if arr.size == 0:
                    return 0
                # 如果是一维多元素（可能是一种 one-hot 或概率向量），取 argmax
                if arr.size > 1:
                    return int(arr.argmax())
                else:
                    return int(arr.flat[0])
            # dict 类型，尝试常见字段
            if isinstance(val, dict):
                for key in ('label', 'pred', 'pred_label', 'analysis'):
                    if key in val:
                        try:
                            return int(val[key])
                        except:
                            try:
                                return int(float(val[key]))
                            except:
                                pass
                # 如果 dict 里只有一个明显的数值字段也可尝试其它逻辑
        # 如果字符串是文本标签，做映射
        text_map = {'fake':0, 'real':1, 'true':1, 'false':0, 'rumor':0, 'non-rumor':1, 'nonrumor':1}
        low = s.lower()
        if low in text_map:
            return text_map[low]
        # 如果还没解析成功，尝试取首个数字字符
        digits = ''.join(ch for ch in s if (ch.isdigit() or ch == '.' or ch == '-'))
        if digits:
            try:
                return int(float(digits))
            except:
                pass
        raise ValueError(f"无法解析 analysis 字段的值: {repr(v)}")
    # 5) 如果本来是 list/tuple/ndarray
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.array(v, dtype=float)
        if arr.size == 0:
            return 0
        if arr.size > 1:
            return int(arr.argmax())
        return int(arr.flat[0])
    # 6) 其它类型
    raise ValueError(f"未知类型的 analysis 字段: {type(v)} -> {repr(v)}")

# 尝试批量解析，并把解析错误打印出来以便诊断
parsed_labels = []
errors = []
for idx, val in enumerate(psesudo_data['analysis'].tolist()):
    try:
        parsed_labels.append(parse_analysis_value(val))
    except Exception as e:
        errors.append((idx, val, str(e)))
        parsed_labels.append(0)  # 出错时使用默认标签 0 或根据需要调整

if errors:
    print("解析 pseusdo analysis 列时遇到以下问题（index, 原始值, 错误）：")
    for e in errors[:20]:  # 只显示前 20 个问题样例
        print(e)
    # 如果很多错误，建议打开 CSV 人工检查这些行

# 指定不同的下载源或手动下载
try:
    # 尝试不同的下载根目录
    model_path = clip._download(clip._MODELS["ViT-B/32"], download_root="./clip_models")
    clip_model, preprocess = clip.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
except:
    # 如果仍然失败，尝试使用默认方式
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu", jit=False)

# 添加当前目录到路径，确保可以导入预处理模块
sys.path.append(os.path.dirname(__file__))

try:
    from preprocess import preprocess_dataset, check_dataset_structure
except ImportError:
    print("警告: 无法导入预处理模块")


def ensure_datasets_ready():
    """确保所有数据集都已准备就绪"""
    datasets = ['pheme', 'twitter', 'weibo']

    for dataset in datasets:
        gcn_train_path = os.path.join(_script_dir, 'dataset', dataset, 'dataforGCN_train.csv')

        if not os.path.exists(gcn_train_path):
            print(f"检测到缺少 {gcn_train_path}，执行预处理...")
            try:
                # 检查数据集结构
                check_dataset_structure(dataset)
                # 执行预处理
                success = preprocess_dataset(dataset)
                if not success:
                    print(f"错误: {dataset} 数据集预处理失败")
                    return False
            except Exception as e:
                print(f"预处理过程中出错: {e}")
                return False
        else:
            print(f"{dataset} 数据集已就绪")

    return True


# 在主程序开始前调用
if __name__ == "__main__":
    if not ensure_datasets_ready():
        print("数据集准备失败，程序退出")
        sys.exit(1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = pd.read_csv('dataset/' + dataset_name + '/dataforGCN_train.csv')
test_data = pd.read_csv('dataset/' + dataset_name + '/dataforGCN_test.csv')

tweet_embeds = torch.load('dataset/' +dataset_name+ '/TweetEmbeds.pt', map_location=device)
tweet_graph = torch.load('dataset/' + dataset_name + '/TweetGraph.pt', map_location=device)

# 使用前面解析后的parsed_labels创建psesudo_labels
psesudo_labels = torch.tensor(np.array(parsed_labels, dtype=np.int64), dtype=torch.long).to(device)

# prob 列也做稳健解析（若列为字符串形式），把无法解析的填为 0
psesudo_probs = pd.to_numeric(psesudo_data.get('prob', pd.Series([0]*len(psesudo_data))), errors='coerce').fillna(0).values
psesudo_probs = torch.tensor(psesudo_probs.astype(float), dtype=torch.float).to(device)

label_list_train = train_data["label"].tolist()
label_list_test = test_data["label"].tolist()

labels = []
for label_list in [label_list_train, label_list_test]:
    labels_i = torch.tensor(label_list, dtype=torch.long)
    labels.append(labels_i)

labels = torch.cat(labels, 0)

data = Data(
    x=tweet_embeds.float(),
    edge_index=tweet_graph.coalesce().indices(),
    edge_attr=None,
    train_mask=torch.tensor([True]*len(label_list_train) + [False]*(len(labels)-len(label_list_train))).bool(),
    test_mask=torch.tensor([False]*len(label_list_train) + [True]*(len(labels)-len(label_list_train))).bool(),
    y=labels
).to(device)
num_features = tweet_embeds.shape[1]
num_classes = 2

data.x = torch.cat([data.x, torch.zeros((data.num_nodes, num_classes), device=device)], dim=1)

class UniMP(torch.nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels, num_layers,
                heads, dropout=0.3):
        super().__init__()

        self.num_classes = num_classes

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            if i < num_layers:
                out_channels = hidden_channels // heads
                concat = True
            else:
                out_channels = num_classes
                concat = False
            conv = TransformerConv(in_channels, out_channels, heads,
                                concat=concat, beta=True, dropout=dropout)
            self.convs.append(conv)
            in_channels = hidden_channels

            if i < num_layers:
                self.norms.append(torch.nn.LayerNorm(hidden_channels))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index)).relu()
        x = self.convs[-1](x, edge_index)
        return x

data.y = data.y.view(-1)
model = GCN(num_features + num_classes, num_classes, hidden_channels=64,
            num_layers=3, heads=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

train_mask = data.train_mask
test_mask = data.test_mask
test_mask_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

def train(label_rate=0.95):
    model.train()

    data.x[:, -num_classes:] = 0

    train_mask_idx = train_mask.nonzero(as_tuple=False).view(-1)
    mask = torch.rand(train_mask_idx.shape[0]) < label_rate
    train_labels_idx = train_mask_idx[mask]
    train_unlabeled_idx = train_mask_idx[~mask]

    # Select top 5% of test samples based on pre-computed probabilities
    num_pseudo = int(len(test_mask_idx) * 0.05)
    topk_indices = torch.topk(psesudo_probs, num_pseudo).indices

    test_psesudo_idx = test_mask_idx[topk_indices]
    selected_psesudo_labels = psesudo_labels[topk_indices]
    data.x[
        torch.cat([train_labels_idx, test_psesudo_idx]),
        -num_classes:
    ] = F.one_hot(
        torch.cat([data.y[train_labels_idx], selected_psesudo_labels]),
        num_classes
    ).float()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_unlabeled_idx], data.y[train_unlabeled_idx])
    loss.backward()
    optimizer.step()

    use_labels = True
    n_label_iters = 1

    if use_labels and n_label_iters > 0:
        unlabel_idx = torch.cat([train_unlabeled_idx, data.test_mask.nonzero(as_tuple=False).view(-1)])
        with torch.no_grad():
            for _ in range(n_label_iters):
                torch.cuda.empty_cache()
                out = out.detach()
                data.x[unlabel_idx, -num_classes:] = F.softmax(out[unlabel_idx], dim=-1)
                out = model(data.x, data.edge_index)

    return loss.item()

max_test_acc = 0
max_precision = 0
max_recall = 0
max_f1 = 0

best_epoch = 0

@torch.no_grad()
def test():
    model.eval()

    data.x[:, -num_classes:] = 0

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

    data.x[train_idx, -num_classes:] = F.one_hot(data.y[train_idx], num_classes).float()

    unlabel_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    n_label_iters = 1
    for _ in range(n_label_iters):
        out = model(data.x, data.edge_index)
        data.x[unlabel_idx, -num_classes:] = F.softmax(out[unlabel_idx], dim=-1)

    out = model(data.x, data.edge_index)
    pred = out[test_mask].argmax(dim=-1)

    y_true = data.y[test_mask].cpu().numpy()
    y_pred = pred.cpu().numpy()

    test_acc = (pred == data.y[test_mask]).sum().item() / pred.size(0)

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    val_acc = 0

    return val_acc, test_acc, precision, recall, f1

best_precision = 0
best_recall = 0
best_f1 = 0

for epoch in range(1, 3001):
    loss = train()
    val_acc, test_acc, precision, recall, f1 = test()
    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, '
    #     f'Test Acc: {test_acc:.4f}, Precision: {precision:.4f}, '
    #     f'Recall: {recall:.4f}, F1: {f1:.4f}')

    if test_acc > max_test_acc:
        max_test_acc = test_acc
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        best_epoch = epoch

print(f'Best Epoch: {best_epoch}, Max Test Acc: {max_test_acc:.4f}, '
    f'Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}')
