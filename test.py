"""
test.py
------------------------------------------
输出指标：
- Rank-1 / Rank-5
- mAP
------------------------------------------
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from model_mobilevit_ea_1 import mobile_vit_small as create_model
from data.data_loader import load_data

# ==============================
# 配置参数
# ==============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = "cifar-10"
root = "data/data-cifar10"  # 替换成实际数据集路径
num_query = 1000
num_train = 5000
batch_size = 64
num_workers = 4
code_length = 48
model_path = "checkpoints/model_weights_48_0.98275.pth"   # 加载模型权重

# ==============================
# 加载数据
# ==============================
print("加载数据中 ...")
query_dataloader, _, retrieval_dataloader = load_data(
    dataset, root, num_query, num_train, batch_size, num_workers
)

# ==============================
# 加载模型
# ==============================
print(f"加载模型权重: {model_path}")
model = create_model(num_classes=code_length).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.eval()

# ==============================
# 哈希码生成函数
# ==============================
def generate_code(model, dataloader, code_length, device):
    model.eval()
    code = torch.zeros([len(dataloader.dataset), code_length])
    with torch.no_grad():
        for data, _, index in tqdm(dataloader, desc="生成哈希码"):
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()
    return code


# ==============================
# 检索指标计算函数
# ==============================
def evaluate_metrics(query_code, query_targets, retrieval_code, retrieval_targets, Ks=[1,5]):
    query_code = query_code.to(device)
    retrieval_code = retrieval_code.to(device)
    query_targets = query_targets.to(device)
    retrieval_targets = retrieval_targets.to(device)

    num_query = query_code.size(0)
    sim_matrix = query_code @ retrieval_code.t()  # 相似度矩阵

    rank_correct = {k: 0 for k in Ks}
    ap_list = []
    ndcg_list = []
    precision_at_k = {k: [] for k in Ks}
    recall_at_k = {k: [] for k in Ks}

    for i in tqdm(range(num_query), desc="计算指标"):
        gt = (query_targets[i] @ retrieval_targets.t() > 0).float()
        if gt.sum() == 0:
            continue

        sim = sim_matrix[i]
        sorted_index = torch.argsort(sim, descending=True)
        sorted_gt = gt[sorted_index]

        relevant = sorted_gt.nonzero(as_tuple=False).squeeze()
        if relevant.numel() == 0:
            continue

        # Rank@K
        for k in Ks:
            if sorted_gt[:k].sum() > 0:
                rank_correct[k] += 1

        # mAP
        precision = torch.arange(1, len(relevant) + 1, device=device).float() / (relevant + 1).float()
        ap = precision.mean().item()
        ap_list.append(ap)

        # Precision@K & Recall@K
        total_relevant = gt.sum().item()
        for k in Ks:
            retrieved_relevant = sorted_gt[:k].sum().item()
            precision_at_k[k].append(retrieved_relevant / k)
            recall_at_k[k].append(retrieved_relevant / total_relevant)

        # NDCG
        gains = sorted_gt
        discounts = torch.log2(torch.arange(2, len(gains) + 2, device=device).float())
        dcg = (gains / discounts).sum().item()
        ideal_gains = torch.sort(gt, descending=True)[0]
        ideal_dcg = (ideal_gains / torch.log2(torch.arange(2, len(ideal_gains) + 2, device=device).float())).sum().item()
        ndcg_list.append(dcg / ideal_dcg if ideal_dcg > 0 else 0)

    results = {
        "Rank@K": {k: rank_correct[k] / num_query for k in Ks},
        "mAP": np.mean(ap_list),
        "Precision@K": {k: np.mean(precision_at_k[k]) for k in Ks},
        "Recall@K": {k: np.mean(recall_at_k[k]) for k in Ks},
        "NDCG": np.mean(ndcg_list),
    }
    return results


# ==============================
# 主流程
# ==============================
print("生成 query / retrieval 哈希码 ...")
query_code = generate_code(model, query_dataloader, code_length, device)
retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
query_targets = query_dataloader.dataset.get_onehot_targets()
retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()

print("计算综合指标 ...")
metrics = evaluate_metrics(query_code, query_targets, retrieval_code, retrieval_targets)

print("\n=================== 测试结果 ===================")
print(f"mAP:       {metrics['mAP']*100:.2f}%")
# print(f"NDCG:      {metrics['NDCG']*100:.2f}%")
for k, v in metrics["Rank@K"].items():
    print(f"Rank@{k}:  {v*100:.2f}%")
# for k, v in metrics["Precision@K"].items():
#     print(f"Precision@{k}: {v*100:.2f}%")
# for k, v in metrics["Recall@K"].items():
#     print(f"Recall@{k}:    {v*100:.2f}%")
print("================================================")
