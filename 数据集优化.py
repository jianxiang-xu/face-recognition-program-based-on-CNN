import os
import numpy as np
from PIL import Image
import random

def adjust_skin_tone(img):
    """保持肤色比例的同时允许亮度随机浮动（兼容所有PIL版本）"""
    img = img.convert('RGB')  # 统一转为RGB模式
    arr = np.array(img)  # 先转为数组获取数据类型
    
    # 通过数组数据类型判断位深度和最大像素值
    if arr.dtype == np.uint8:
        max_val = 255
        dtype = np.uint8  # 8位图像
    
    arr = arr.astype(np.float32)  # 转为float32用于计算
    
    # 目标肤色的RGB比例关系（核心）
    r_ratio = 0.85
    g_ratio = 0.68
    b_ratio = 0.48
    total_ratio = r_ratio + g_ratio + b_ratio
    
    # 计算原图的平均亮度（用于确定基准）
    avg_brightness = np.mean(arr) / max_val
    
    # 生成随机亮度系数（0.9-1.2倍浮动）
    brightness_factor = random.uniform(0.9, 1.2)
    target_brightness = avg_brightness * brightness_factor * total_ratio
    
    # 计算各通道目标值（保持比例 + 随机亮度）
    r_target = (r_ratio / total_ratio) * target_brightness * max_val
    g_target = (g_ratio / total_ratio) * target_brightness * max_val
    b_target = (b_ratio / total_ratio) * target_brightness * max_val
    
    # 计算各通道调整系数
    r_mean = np.mean(arr[..., 0])
    g_mean = np.mean(arr[..., 1])
    b_mean = np.mean(arr[..., 2])
    
    r_coeff = r_target / (r_mean + 1e-6)
    g_coeff = g_target / (g_mean + 1e-6)
    b_coeff = b_target / (b_mean + 1e-6)
    
    # 应用系数并限制范围
    arr[..., 0] *= r_coeff
    arr[..., 1] *= g_coeff
    arr[..., 2] *= b_coeff
    arr = np.clip(arr, 0, max_val).astype(dtype)
    
    return Image.fromarray(arr)

def batch_process(folder):
    counter=14000
    if not os.path.isdir(folder):
        print(f"错误：{folder} 不是有效文件夹")
        return
        
    for name in os.listdir(folder)[counter:]:
        counter+=1
        if name.lower().endswith(('.jpg', '.jpeg')):
            path = os.path.join(folder, name)
            try:
                with Image.open(path) as img:
                    new_img = adjust_skin_tone(img)
                    new_img.save(path)
            except Exception as e:
                print(f"处理 {name} 失败：{str(e)}")
        if counter%100==0:
            print(f"已处理：{counter}张图片")


import os

def delete_first_n_jpg_files(folder_path, n):
    """
    删除指定文件夹中的前n个jpg文件
    
    参数:
        folder_path (str): 文件夹路径
        n (int): 要删除的文件数量
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个文件夹")
        return
    
    # 检查n是否为正整数
    if not isinstance(n, int) or n <= 0:
        print(f"错误: 请输入正整数作为要删除的文件数量，当前输入为: {n}")
        return
    
    # 获取文件夹中所有jpg文件
    jpg_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        # 检查是否为文件且以.jpg结尾（不区分大小写）
        if os.path.isfile(file_path) and file.lower().endswith('.jpg'):
            jpg_files.append(file_path)
    
    # 按文件名排序（可以根据需要修改排序方式）
    jpg_files.sort()
    
    # 检查文件数量是否足够
    if len(jpg_files) < n:
        print(f"警告: 文件夹中只有 {len(jpg_files)} 个jpg文件，少于要删除的 {n} 个")
        n = len(jpg_files)
        if n == 0:
            print("没有jpg文件可删除")
            return
    
    # 显示待删除的文件
    print(f"\n即将删除以下 {n} 个文件:")
    for i in range(n):
        print(f"{i+1}. {jpg_files[i]}")
    
    # 确认删除
    confirm = input("\n确定要删除这些文件吗？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消删除操作")
        return
    
    # 执行删除
    deleted_count = 0
    for i in range(n):
        try:
            os.remove(jpg_files[i])
            deleted_count += 1
            print(f"已删除: {jpg_files[i]}")
        except Exception as e:
            print(f"删除失败 {jpg_files[i]}: {str(e)}")
    
    print(f"\n操作完成，成功删除 {deleted_count} 个文件")

# if __name__ == "__main__":
#     # 在这里设置要操作的文件夹路径和要删除的文件数量
#     folder_to_clean = r"D:\systemDir\Desktop\experiment\CNN\dataset\nonface_24x24_0"  # 替换为你的文件夹路径
#     num_files_to_delete = 30000                  # 替换为要删除的文件数量
    
#     delete_first_n_jpg_files(folder_to_clean, num_files_to_delete)

if __name__ == "__main__":
    folder = r"D:\systemDir\Desktop\experiment\CNN\dataset\nonface_24x24_0"
    batch_process(folder)
    print("全部处理完成")