# name=generate_clip_embeds.py
# 用法: 在项目根激活 venv 后运行 python generate_clip_embeds.py
import os
import torch
import pandas as pd
import clip  # 已安装本地 CLIP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

dataset = 'pheme'
base = os.path.join('dataset', dataset)
train_path = os.path.join(base, 'dataforGCN_train.csv')
test_path = os.path.join(base, 'dataforGCN_test.csv')

if not (os.path.exists(train_path) and os.path.exists(test_path)):
    raise FileNotFoundError("请先准备 dataforGCN_train.csv 和 dataforGCN_test.csv（或运行 preprocess.py）")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
texts = train['text'].astype(str).tolist() + test['text'].astype(str).tolist()
num_nodes = len(texts)
print(f"Generating CLIP text embeddings for {num_nodes} texts on {device}")

batch_size = 64
embeds_list = []
model.eval()
with torch.no_grad():
    for i in range(0, num_nodes, batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = clip.tokenize(batch_texts).to(device)  # tokenizes to tensor
        text_features = model.encode_text(tokens)
        # optional: normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        embeds_list.append(text_features.cpu())
embeds = torch.cat(embeds_list, dim=0)
embeds_path = os.path.join(base, 'TweetEmbeds.pt')
torch.save(embeds, embeds_path)
print(f"Wrote real CLIP embeddings to {embeds_path}, shape={embeds.shape}")

# 同样需要 graph（如果没有生成图逻辑，这里生成简单自环图或 kNN 图）
# 下面示例生成自环图（每个节点一个自环）
idx = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
vals = torch.ones(num_nodes, dtype=torch.float)
sparse_adj = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes)).coalesce()
graph_path = os.path.join(base, 'TweetGraph.pt')
torch.save(sparse_adj, graph_path)
print(f"Wrote TweetGraph.pt to {graph_path}")