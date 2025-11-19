# name=generate_placeholders.py
# 用法: 在项目根激活 venv 后运行 python generate_placeholders.py
import os
import torch
import pandas as pd

dataset = 'pheme'
base = os.path.join('dataset', dataset)
os.makedirs(base, exist_ok=True)

# 读取 train/test 以确定节点数（preprocess 已应当生成 dataforGCN_*）
# 如果没有则用一个默认的数量（例如 10）
train_path = os.path.join(base, 'dataforGCN_train.csv')
test_path = os.path.join(base, 'dataforGCN_test.csv')

if os.path.exists(train_path) and os.path.exists(test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    num_nodes = len(train) + len(test)
    print(f"Found train/test: num_nodes = {num_nodes}")
else:
    # 若不存在，使用小规模占位并也生成 minimal dataforGCN 文件
    print("dataforGCN_train/test 未找到，生成最小占位 train/test 并使用默认节点数 6")
    num_nodes = 6
    os.makedirs(base, exist_ok=True)
    pd.DataFrame({'id':[1,2,3],'text':['示例1','示例2','示例3'],'label':[0,1,0]}).to_csv(train_path, index=False, encoding='utf-8')
    pd.DataFrame({'id':[4,5,6],'text':['示例4','示例5','示例6'],'label':[1,0,1]}).to_csv(test_path, index=False, encoding='utf-8')

# Embedding 维度（CLIP ViT-B/32 常用 512，可按需要改）
embed_dim = 512
embeds = torch.randn(num_nodes, embed_dim)
embeds_path = os.path.join(base, 'TweetEmbeds.pt')
torch.save(embeds, embeds_path)
print(f"Wrote embeddings to {embeds_path}, shape = {embeds.shape}")

# 构造简单图：自环（每个节点一个自环）
idx = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)  # shape [2, num_nodes]
vals = torch.ones(num_nodes, dtype=torch.float)
sparse_adj = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes))
sparse_adj = sparse_adj.coalesce()
graph_path = os.path.join(base, 'TweetGraph.pt')
torch.save(sparse_adj, graph_path)
print(f"Wrote sparse graph to {graph_path}, indices shape = {sparse_adj.coalesce().indices().shape}")