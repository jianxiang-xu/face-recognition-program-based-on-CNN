import os.path as osp
from PIL import Image
from torch.utils.data import Dataset  # 导入Dataset父类
from torchvision import transforms
import os  # 用于处理文件路径和列表
import torch

class mtcnn_dataset(Dataset):  # 继承自Dataset
    def __init__(self, img_dir):
        self.img_dir = img_dir
        # 获取文件夹下所有图片文件的路径列表
        self.data_list = [osp.join(img_dir, f) for f in os.listdir(img_dir) 
                         if f.lower().endswith('.jpg')]
        # 从文件夹名称末尾判断标签
        dir_name = osp.basename(img_dir)
        if dir_name.endswith('1'):
            self.label = float(1)
        elif dir_name.endswith('0'):
            self.label = float(0)

    def __getitem__(self, index):
        img_path = self.data_list[index]  # 获取对应索引的图片路径
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
        return (img_tensor, self.label)

    def __len__(self):
        return len(self.data_list)


class ldmk_dataset(Dataset):
    def __init__(self, data_list, data_dir):
        """
        :param train_data_list: [train_data_num,[img_path,labels,[offsets],[landmark]]
        :return:
        """
        self.data_list = data_list
        self.data_dir = data_dir

    def __getitem__(self, index):
        item = self.data_list[index]
        img_path = osp.join(self.data_dir, item[0])
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
        landmark = torch.FloatTensor(item[1])

        return (img_tensor, landmark)

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    # 示例：假设存在名为"images_1"的文件夹（标签为1）
    dataset = mtcnn_dataset(img_dir="images_1")
    if len(dataset) > 0:
        print(dataset[0])  # 输出第一张图片的张量和标签
    else:
        print("数据集为空")