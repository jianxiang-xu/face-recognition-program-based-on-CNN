from model import O_Net
import torch
from torch import nn
import os.path as osp
from torch.utils.data import DataLoader
from dataset import ldmk_dataset

def load_txt(data_path):
    samples = []
    with open(data_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) < 11:  # 确保至少有图片路径+10个关键点坐标
                continue
            img_path = parts[0]
            landmark = [float(s) for s in parts[1:11]]  # 取10个关键点
            samples.append([img_path, landmark])
    return samples

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0001

# 修正：添加数据目录参数
dataset = ldmk_dataset(
    data_list=load_txt(r"D:\systemDir\Desktop\experiment\CNN\dataset\ldmk_faces\O_Net_dataset.txt"),
    data_dir=r"D:\systemDir\Desktop\experiment\CNN\dataset\ldmk_faces"  # 根据实际数据目录修改
)
dataloader = DataLoader(dataset, batch_size=24, shuffle=True, pin_memory=False)

# 修正：重新定义损失函数为独立函数
def landmark_loss(pred_landmark, gt_landmark):
    pred_landmark = torch.squeeze(pred_landmark)
    gt_landmark = torch.squeeze(gt_landmark)
    valid_gt_landmark = gt_landmark
    valid_pred_landmark = pred_landmark
        
    return nn.MSELoss()(valid_pred_landmark, valid_gt_landmark)

DEBUG = False


def train_model(model_saved_pth, max_iteration=5000):
    model = O_Net()
    # 加载模型时增加验证和提示
    if osp.exists(model_saved_pth):
        try:
            model.load_state_dict(torch.load(model_saved_pth))
            print(f"成功加载模型参数：{model_saved_pth}")
        except Exception as e:
            print(f"加载模型失败，将从随机初始化开始：{e}")
    else:
        print(f"未找到模型文件，将从随机初始化开始")
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iteration = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        imgs, landmarks = batch  # 修正：变量名统一
        imgs = imgs.to(DEVICE)
        landmarks = landmarks.to(DEVICE)
        
        output = model(imgs)
        loss = landmark_loss(output, landmarks)  # 修正：损失函数调用方式
        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 20 == 0:
            print(f'迭代{iteration}次, loss={loss.item()}')
            torch.save(model.state_dict(), model_saved_pth)
        if iteration > max_iteration:
            break

if __name__ == '__main__':
    train_model(model_saved_pth=r"D:\systemDir\Desktop\experiment\CNN\myMTCNN\model_O_1.pth")
