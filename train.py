import os
import torch
import argparse
import model     
from loguru import logger   
from data.data_loader import load_data

# ==================================
# 主函数入口
# ==============================
def run():

    args = load_config()   # 读取命令行参数（学习率、批大小、哈希长度等）


    logger.add('logs/{time}.log', rotation='500 MB', level='INFO')
    logger.info(args)

    torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动优化，提高 GPU 训练速度

    query_dataloader, _, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.num_query,
        args.num_samples,
        args.batch_size,
        args.num_workers,
    )

    print("\n===== GPU 设备验证 =====")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"当前 CUDA 设备: {torch.cuda.current_device()}")  
    print(f"设备数量: {torch.cuda.device_count()}")
    print(f"当前设备名称: {torch.cuda.get_device_name()}\n")

    # ==========================================================
    #   针对不同的哈希码长度，依次训练模型并计算 mAP（检索性能指标）
    # =================================================
    for code_length in args.code_length:
        mAP = model.train(   
            query_dataloader,      # 查询集
            retrieval_dataloader,  # 检索库
            code_length,     # 哈希码长度（12, 24, 32, 48）
            args.device,   # 设备（GPU/CPU）
            args.lr,    # 学习率
            args.max_iter,   # 最大迭代次数
            args.max_epoch,   # 最大训练轮数
            args.num_samples,   # 样本数量
            args.batch_size,   # 批大小
            args.root,     # 数据路径
            args.dataset,  # 数据集名称
            args.gamma,    # 超参数 γ
            args.lambda0,
            args.topk,    # 计算 top-k 检索精度
        )

        # 记录训练结果（mAP）
        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, mAP))

        print("\n===== GPU 设备验证 =====")
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        print(f"当前 CUDA 设备: {torch.cuda.current_device()}")  # 应该输出 1
        print(f"设备数量: {torch.cuda.device_count()}")
        print(f"当前设备名称: {torch.cuda.get_device_name()}\n")


# ==============================================================
# 参数解析函数：定义所有可调超参数与默认值
# ========================================================
def load_config():
    """
        加载配置参数（命令行可修改），控制训练过程的各项超参数
    """
    parser = argparse.ArgumentParser(description='ADSH_PyTorch')

    # 数据集与路径设置
    parser.add_argument('--dataset', default='cifar-10', type=str, help='Dataset name.')
    parser.add_argument('--root', default='./data/data-cifar10', type=str, help='Path of dataset')

    # 训练超参数
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--code-length', default='48', type=str,
                        help='Binary hash code length.(default: 48,32,24,12)')   # 16,32,，48,64
    parser.add_argument('--max-iter', default=250, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('--max-epoch', default=1, type=int,
                        help='Number of epochs.(default: 3)')

    # 数据子集配置
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-samples', default=5000, type=int,
                        help='Number of sampling data points.(default: 2000)')

    # 运行线程与计算配置
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    # parser.add_argument('--gpu', default=0, type=int,
    #                     help='Using gpu.(default: False)')             
    parser.add_argument('--gamma', default=0.1, type=float,    # 参数设置
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--lambda0', default=0.1, type=float)    # 超参数设置

    # GPU 选择
    parser.add_argument('--gpu', default=2, type=int,
                        help='Using gpu.(default: False)')


    args = parser.parse_args()

    # 设备选择逻辑
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(args.gpu)

    # ==========================================================
    # 将 code_length 从字符串转为整数列表
    # =================================================
    args.code_length = list(map(int, args.code_length.split(',')))

    return args

# ==============================================================
# 程序入口
# ==========================================================
if __name__ == '__main__':
    run()
