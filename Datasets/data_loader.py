from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import os

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile

import data.cifar10 as cifar10
import data.nus_wide as nuswide
import data.flickr25k as flickr25k
import data.imagenet as imagenet

from data.transform import train_transform, encode_onehot

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(dataset, root, num_query, num_train, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    if dataset == 'cifar-10':
        query_dataloader, train_dataloader, retrieval_dataloader = cifar10.load_data(root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'nus-wide-tc10':
        query_dataloader, train_dataloader, retrieval_dataloader = nuswide.load_data(10,
                                                                                     root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'nus-wide-tc21':
        query_dataloader, train_dataloader, retrieval_dataloader = nuswide.load_data(21,
                                                                                     root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers
                                                                                     )
    elif dataset == 'flickr25k':
        query_dataloader, train_dataloader, retrieval_dataloader = flickr25k.load_data(root,
                                                                                       num_query,
                                                                                       num_train,
                                                                                       batch_size,
                                                                                       num_workers,
                                                                                       )
    elif dataset == 'imagenet':
        query_dataloader, train_dataloader, retrieval_dataloader = imagenet.load_data(root,
                                                                                      batch_size,
                                                                                      num_workers,
                                                                                      )
    else:
        raise ValueError("Invalid dataset name!")

    return query_dataloader, train_dataloader, retrieval_dataloader


def sample_dataloader(dataloader, num_samples, batch_size, root, dataset):
    """
    Sample data from dataloder.

    Args
        dataloader (torch.utils.data.DataLoader): Dataloader.
        num_samples (int): Number of samples.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        sample_index (int): Sample index.
        dataset(str): Dataset name.

    Returns
        sample_dataloader (torch.utils.data.DataLoader): Sample dataloader.
    """
    # 数据集59000张图片的张量列表
    data = dataloader.dataset.data
    # 数据集59000张图片的标签
    targets = dataloader.dataset.targets
    # 生成一个大小为59000的随机排列向量，其中每个元素都对应数据集中一个样本的索引(范围为0-58999)，再选取前2000个,即随机选取2000个图片的索引
    sample_index = torch.randperm(data.shape[0])[:num_samples]
    # 提取出上一步随机的2000张图片的张量数据
    data = data[sample_index]
    # 提取对应的2000张图片标签
    targets = targets[sample_index]
    # 将数据包装到数据加载器中。
    sample = wrap_data(data, targets, batch_size, root, dataset)

    return sample, sample_index

#将数据包装到数据加载器中。
#将在数据集59000张图片中随机选取2000张图片，利用其张量数据以及对应的标签进行数据增强，最后包装到数据加载器中
def wrap_data(data, targets, batch_size, root, dataset):
    """
    Wrap data into dataloader.

    Args
        data (np.ndarray): Data.
        targets (np.ndarray): Targets.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        dataset(str): Dataset name.

    Returns
        dataloader (torch.utils.data.dataloader): Data loader.
    """
    class MyDataset(Dataset):
        def __init__(self, data, targets, root, dataset):
            self.data = data
            self.targets = targets
            self.root = root
            self.transform = train_transform()
            self.dataset = dataset
            if dataset == 'cifar-10':
                self.onehot_targets = encode_onehot(self.targets, 10)
            else:
                self.onehot_targets = self.targets

        def __getitem__(self, index):
            if self.dataset == 'cifar-10':
                #Image.fromarray() 函数提供了一种便捷的方式将 numpy 数组中存储的图像数据转换为 PIL.Image 对象，并进一步进行裁剪、缩放、旋转、变换等操作。
                img = Image.fromarray(self.data[index])
                if self.transform is not None:
                    img = self.transform(img)
            else:
                img_path = os.path.join(self.root, self.data[index])

                # ------------------ ① 检查文件是否存在 ------------------
                if not os.path.exists(img_path):
                    print(f"❌ [DataLoader] 图片路径不存在: {img_path}")
                    # 返回一张全黑图片占位，防止中断
                    img = Image.new('RGB', (224, 224), (0, 0, 0))
                else:
                    # ------------------ ② 尝试打开图片 ------------------
                    try:
                        img = Image.open(img_path).convert('RGB')
                    except Exception as e:
                        print(f"⚠️ [DataLoader] 无法打开图片: {img_path}, 错误: {e}")
                        # 若图片损坏则用黑图代替
                        img = Image.new('RGB', (224, 224), (0, 0, 0))

                # ------------------ ③ 图像预处理 ------------------
                if self.transform is not None:
                    img = self.transform(img)

            return img, self.targets[index], index
            #     img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
            #     img = self.transform(img)
            # return img, self.targets[index], index

        def __len__(self):
            return self.data.shape[0]

        def get_onehot_targets(self):
            """
            Return one-hot encoding targets.
            """
            return torch.from_numpy(self.onehot_targets).float()

    #实例化数据加载器
    dataset = MyDataset(data, targets, root, dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader
