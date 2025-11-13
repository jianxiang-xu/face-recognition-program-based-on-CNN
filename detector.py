from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
from model import R_Net, O_Net  # 导入O_Net模型
import torch
import numpy as np

def detect(img_path, min_size=120):
    """多尺度检测人脸候选框（P-Net思路）"""
    # 加载R-Net模型
    r_model = R_Net()
    r_model.load_state_dict(torch.load(r"D:\systemDir\Desktop\experiment\CNN\myMTCNN\model_R.pth"))
    r_model.eval()  # 推理模式

    # 读取图像
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f'加载图像失败: {e}')
        return []

    w0, h0 = img.width, img.height
    bboxes = []
    scale = 0.5  # 尺度因子

    while True:
        # 计算当前尺度图像尺寸
        w = int(w0 * scale)
        h = int(h0 * scale)
        if w < min_size or h < min_size:
            break  # 小于最小尺寸则停止

        # 缩放图像
        scaled_img = img.resize((w, h))
        tensor = ToTensor()(scaled_img).unsqueeze(0)  # 增加批次维度
        window_size = 24
        stride = 2

        # 滑动窗口检测
        for i in range(0, h - window_size, stride):
            for j in range(0, w - window_size, stride):
                # 截取窗口区域
                window = tensor[:, :, i:i+window_size, j:j+window_size]
                # 模型预测
                with torch.no_grad():
                    cls = r_model(window).item()
                
                if cls > 0.98:  # 置信度阈值
                    # 转换回原图坐标
                    x1 = int(j / scale)
                    y1 = int(i / scale)
                    x2 = int((j + window_size) / scale)
                    y2 = int((i + window_size) / scale)
                    bboxes.append((x1, y1, x2, y2, cls))

        scale /= 1.3  # 缩小尺度

    return bboxes

def get_landmarks(img, bbox, o_model):
    """使用O-Net获取人脸关键点"""
    x1, y1, x2, y2 = bbox[:4]
    # 裁剪人脸区域
    face_img = img.crop((x1, y1, x2, y2))
    # 调整为O-Net输入尺寸(48x48)
    face_img = face_img.resize((48, 48), Image.BILINEAR)
    # 转换为张量
    tensor = ToTensor()(face_img).unsqueeze(0)
    
    # 模型预测
    with torch.no_grad():
        landmarks = o_model(tensor).numpy()[0]
    
    # 关键点坐标转换回原图坐标系
    face_width = x2 - x1
    face_height = y2 - y1
    landmarks = landmarks.reshape(5, 2)  # 重塑为5个点(x,y)
    
    # 转换坐标
    for i in range(5):
        # O-Net输出的是相对坐标，需要转换为绝对坐标
        landmarks[i][0] = x1 + landmarks[i][0] * face_width 
        landmarks[i][1] = y1 + landmarks[i][1] * face_height 
    
    return landmarks.astype(int)

def show_bboxes_with_landmarks(img_path, bboxes, landmarks_list):
    """绘制人脸框和关键点"""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, score = bbox
        # 绘制红色矩形框，线宽2
        draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)
        
        # # 绘制关键点
        landmarks = landmarks_list[i]
        # 5个点分别用不同颜色标记
        colors = ['blue', 'green', 'yellow', 'purple', 'orange']
        # 点的标签（左眼、右眼、鼻子、左嘴角、右嘴角）
        labels = ['LE', 'RE', 'N', 'LM', 'RM']
        
        for j, (x, y) in enumerate(landmarks):
            # 绘制圆点

            draw.ellipse([(x-1, y-1), (x+1, y+1)], fill=colors[j])
    
    return img

def nms(boxes, overlap_threshold=0.2, mode='union'):
    """非极大值抑制，去除重叠框"""
    if len(boxes) == 0:
        return []
        
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if mode == 'min':
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]
    
    kept_boxes = boxes[keep]

    return kept_boxes

if __name__ == '__main__':
    img_path = r"D:\systemDir\Desktop\OIP-C.webp"
    
    # 1. 检测人脸候选框
    bboxes = detect(img_path)
    print(f"原始检测框数量: {len(bboxes)}")
    
    # 2. 应用NMS去除重叠框
    filtered_bboxes = nms(bboxes)
    print(f"NMS后保留框数量: {len(filtered_bboxes)}")
    
    # 3. 加载O-Net模型
    o_model = O_Net()
    o_model.load_state_dict(torch.load(r"D:\systemDir\Desktop\experiment\CNN\myMTCNN\model_O_1.pth"))
    o_model.eval()
    
    # 4. 获取每个人脸的关键点
    img = Image.open(img_path).convert('RGB')
    landmarks_list = []
    for bbox in filtered_bboxes:
        landmarks = get_landmarks(img, bbox, o_model)
        landmarks_list.append(landmarks)
    
    # 5. 绘制并显示结果
    result_img = show_bboxes_with_landmarks(img_path, filtered_bboxes, landmarks_list)
    result_img.show()
    
    # 可选：保存结果图片
    # result_img.save("detection_result.jpg")