import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate
from loguru import logger
from loss.HALH_Loss import HALH_Loss
from data.data_loader import sample_dataloader
from model_mobilevit_ea_1 import mobile_vit_small as create_model

def get_config():
    config = {
        "topK": 59000,   
    }
    return config
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='mobilevit_s.pt', help='initial weights path')
parser.add_argument('--freeze-layers', type=bool, default=False, help='是否冻结除分类头外的参数')
parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')   
args = parser.parse_args()

def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    """
        qB: 查询集哈希码 [num_query, bit]
        rB: 检索库哈希码 [num_gallery, bit]
        queryL, retrievalL: one-hot 标签
    """
    queryL = queryL.to(torch.int32).cpu().numpy()
    retrievalL = retrievalL.to(torch.int32).cpu().numpy()
    qB = qB.cpu().numpy()
    rB = rB.cpu().numpy()

    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))

    for iter in tqdm(range(num_query)):
        gnd = (queryL[iter, :] @ retrievalL.transpose()) > 0   
        hamm = 0.5 * (rB.shape[1] - qB[iter, :] @ rB.T)   
        ind = np.argsort(hamm)  
        gnd = gnd[ind] 
        tgnd = gnd[0:topk] 
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:    
            continue

        count = np.linspace(1, tsum, tsum)  
        all_sim_num = np.sum(gnd)  
        prec_sum = np.cumsum(gnd)    
        return_images = np.arange(1, num_gallery + 1)    

        prec[iter, :] = prec_sum / return_images   
        recall[iter, :] = prec_sum / all_sim_num    

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0   
        topkmap_ = np.mean(count / (tindex))    
        topkmap = topkmap + topkmap_

    topkmap = topkmap / num_query   

    
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()   
    prec = prec[index]   
    recall = recall[index]
    cum_prec = np.mean(prec, 0)   
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall
    
def train(
        query_dataloader,
        retrieval_dataloader,
        code_length,
        device,
        lr,  
        max_iter,  
        max_epoch,  
        num_samples,  
        batch_size,  
        root,
        dataset,
        gamma,  
        lambda0,
        topk,
):
   
    model = create_model(num_classes=code_length).to(device)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = HALH_Loss(code_length, gamma,lambda0)

    num_retrieval = len(retrieval_dataloader.dataset)  
    U = torch.zeros(num_samples, code_length).to(device)  
    B = torch.randn(num_retrieval, code_length).to(    
        device)  
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)  
    best_mAP = 0.0
    start = time.time()

    for it in range(max_iter):
        iter_start = time.time()

        
        train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, num_samples, batch_size, root, dataset)
        sample_index = sample_index.to(device)
        
        train_targets = train_dataloader.dataset.get_onehot_targets().to(device)
        
        S = (train_targets @ retrieval_targets.t() > 0).float()
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))   

        
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r

        
        for epoch in range(max_epoch):
            for batch, (data, targets, index) in enumerate(train_dataloader):
                data, targets, index = data.to(device), targets.to(device), index.to(device)
                optimizer.zero_grad()

                F = model(data)  
                U[index, :] = F.data   
                cnn_loss = criterion(F, B, S[index, :], sample_index[index])   
                cnn_loss.backward()
                optimizer.step()

        expand_U = torch.zeros(B.shape).to(device)  
        a = expand_U[sample_index, :] = U
        expand_U[sample_index, :] = U   
        B = solve_dcc(B, U, expand_U, S, code_length, gamma)   

        query_code = generate_code(model, query_dataloader, code_length, device)
        config = get_config()
        mAP, cum_prec, cum_recall = CalcTopMapWithPR(query_code.to(device),
                                                     query_dataloader.dataset.get_onehot_targets().to(device),
                                                     B, retrieval_targets,
                                                     config["topK"])

        index_range = 59000 // 100  
        index = [i * 100 - 1 for i in range(1, index_range + 1)]
        max_index = max(index)  
        overflow = - index_range * 100  
        index = index + [max_index + i for i in range(1, overflow + 1)]
        c_prec = cum_prec[index]  
        c_recall = cum_recall[index]   

        pr_data = {
            "index": index,
            "P": c_prec.tolist(),
            "R": c_recall.tolist()
        }

        if best_mAP < mAP:
            best_mAP = mAP

            config["pr_curve_path"] = f"log/MVit/mobilevit_noEA/cifar-10_{code_length}_{best_mAP:.5f}.json"
            os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
            with open(config["pr_curve_path"], 'w') as f:
                f.write(json.dumps(pr_data))
            print("PR曲线保存至", config["pr_curve_path"])

        iter_loss = calc_loss(U, B, S, code_length, sample_index, gamma)
        logger.debug('[iter:{}/{}][loss:{:.2f}][map:{:.4f}][iter_time:{:.2f}]'.format(it + 1, max_iter, iter_loss, mAP, time.time() - iter_start))

    logger.info('[总训练时间:{:.2f}]'.format(time.time() - start))
    
    os.makedirs('checkpoints', exist_ok=True)

    torch.save(query_code.cpu(), os.path.join(
        'checkpoints', f'query_code_{code_length}_{best_mAP:.5f}.pth'))
    torch.save(B.cpu(), os.path.join(
        'checkpoints', f'database_code_{code_length}_{best_mAP:.5f}.pth'))
    torch.save(query_dataloader.dataset.get_onehot_targets().cpu(), os.path.join(
        'checkpoints', f'query_targets_{code_length}_{best_mAP:.5f}.pth'))
    torch.save(retrieval_targets.cpu(), os.path.join(
        'checkpoints', f'database_targets_{code_length}_{best_mAP:.5f}.pth'))
    torch.save(model.state_dict(), os.path.join(
        'checkpoints', f'model_weights_{code_length}_{best_mAP:.5f}.pth'))

    print(f"模型权重及哈希码已保存至 checkpoints/，比特长度: {code_length}, best_mAP: {best_mAP:.5f}")

    return best_mAP

def solve_dcc(B, U, expand_U, S, code_length, gamma):
   
    Q = (code_length * S).t() @ U + gamma * expand_U  

    
    for bit in range(code_length):
        
        q = Q[:, bit]  
        u = U[:, bit]  
        
        B_prime = torch.cat((B[:, :bit], B[:, bit + 1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit + 1:]), dim=1)
        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()   

    return B

def calc_loss(U, B, S, code_length, omega, gamma):
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    quantization_loss1 = 0.1 * U.mean(dim=1).pow(2).mean()

    loss1 = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])
    loss = loss1+quantization_loss1

    return loss.item()

def generate_code(model, dataloader, code_length, device):

    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code= model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code
