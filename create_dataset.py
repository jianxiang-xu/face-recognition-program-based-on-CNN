from os import path as osp
import os
import pickle
from PIL import Image
import PIL
import numpy as np
import random

DEBUG = False
# 记录处理状态的文件路径
PROGRESS_FILE = "RNet_ldmk_progress.pkl"


def load_progress():
    """加载已处理的图片记录"""
    if osp.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'rb') as f:
            return pickle.load(f)
    return set()


def save_progress(processed):
    """保存已处理的图片记录"""
    with open(PROGRESS_FILE, 'wb') as f:
        pickle.dump(processed, f)


def create_pnet_data_txt_parser(txt_path, img_dir):
    """解析标注文件，获取图片路径和人脸框信息"""
    if osp.exists(txt_path):
        img_faces = []
        with open(txt_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        line_counter = 0
        while line_counter < len(lines):
            img_path = lines[line_counter]
            faces_num = int(lines[line_counter + 1])
            faces_pos = []
            
            for i in range(faces_num):
                face_info = lines[line_counter + 2 + i].split()
                face_pos = list(map(int, face_info[:4]))
                faces_pos.append(face_pos)
            
            real_img_path = osp.join(img_dir, img_path)
            if osp.exists(real_img_path):
                try:
                    with Image.open(real_img_path) as img:
                        img.verify()
                    img_faces.append([real_img_path, faces_pos])
                    if DEBUG:
                        print(f"已添加有效图片: {real_img_path}")
                except Exception as e:
                    if DEBUG:
                        print(f"无效图片 {real_img_path}: {e}")
            else:
                print(f"警告: 图片路径不存在 {real_img_path}")
                
            line_counter += (2 + faces_num)
        
        return img_faces
    else:
        print(f"警告: 标注文件不存在 {txt_path}")
        return []


def crop_and_save_face(img_path, face_boxes, face_save_dir, crop_size=24):
    """截取人脸区域并保存"""
    try:
        with Image.open(img_path) as img:
            img_np = np.array(img)
            img_width, img_height = img.size
            saved_count = 0
            
            for i, box in enumerate(face_boxes):
                x1, y1, w, h = box

                if w < 24 or h < 24:
                    if DEBUG:
                        print(f"跳过过小的人脸（宽={w}, 高={h}）：{img_path}")
                    continue
                x2 = min(x1 + w, img_width)
                y2 = min(y1 + h, img_height)
                x1 = max(0, x1)
                y1 = max(0, y1)
                
                if x2 > x1 and y2 > y1:
                    crop_img_np = img_np[y1:y2, x1:x2, :]
                    crop_img = Image.fromarray(crop_img_np)
                    crop_img = crop_img.resize((crop_size, crop_size), resample=PIL.Image.BILINEAR)
                    
                    img_name = osp.splitext(osp.basename(img_path))[0]
                    save_path = osp.join(face_save_dir, f"{img_name}_face_{i}.jpg")
                    # 检查是否已存在，避免重复保存
                    if not osp.exists(save_path):
                        crop_img.save(save_path)
                    saved_count += 1
        
        return saved_count
    except Exception as e:
        if DEBUG:
            print(f"处理人脸时出错 {img_path}: {e}")
        return 0


def generate_non_face_regions(img_width, img_height, face_boxes, crop_size=24, num_non_face=1):
    """生成不与人脸重叠的非人脸区域"""
    non_face_regions = []
    max_attempts = 100
    
    for _ in range(num_non_face):
        attempts = 0
        while attempts < max_attempts:
            x1 = random.randint(0, img_width - crop_size)
            y1 = random.randint(0, img_height - crop_size)
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            
            overlap = False
            for (fx1, fy1, fw, fh) in face_boxes:
                fx2 = fx1 + fw
                fy2 = fy1 + fh
                
                if not (x2 < fx1 or x1 > fx2 or y2 < fy1 or y1 > fy2):
                    overlap = True
                    break
            
            if not overlap:
                non_face_regions.append((x1, y1, x2, y2))
                break
            
            attempts += 1
        
        if attempts >= max_attempts and DEBUG:
            print(f"警告: 无法为图片生成足够的非人脸区域")
    
    return non_face_regions


def crop_and_save_non_face(img_path, face_boxes, non_face_save_dir, crop_size=24, num_per_face=1):
    """截取非人脸区域并保存"""
    try:
        with Image.open(img_path) as img:
            img_np = np.array(img)
            img_width, img_height = img.size
            total_non_face = len(face_boxes) * num_per_face
            non_face_regions = generate_non_face_regions(
                img_width, img_height, face_boxes, crop_size, total_non_face
            )
            
            saved_count = 0
            img_name = osp.splitext(osp.basename(img_path))[0]
            
            for i, (x1, y1, x2, y2) in enumerate(non_face_regions):
                save_path = osp.join(non_face_save_dir, f"{img_name}_nonface_{i}.jpg")
                # 检查是否已存在，避免重复保存
                if not osp.exists(save_path):
                    crop_img_np = img_np[y1:y2, x1:x2, :]
                    crop_img = Image.fromarray(crop_img_np)
                    crop_img.save(save_path)
                saved_count += 1
        
        return saved_count
    except Exception as e:
        if DEBUG:
            print(f"处理非人脸时出错 {img_path}: {e}")
        return 0


def create_dataset(txt_path, img_dir, face_save_dir, non_face_save_dir, crop_size=24):
    """生成人脸和非人脸数据集（支持断点续传）"""
    os.makedirs(face_save_dir, exist_ok=True)
    os.makedirs(non_face_save_dir, exist_ok=True)
    
    # 加载已处理的图片记录
    processed_imgs = load_progress()
    print(f"已处理 {len(processed_imgs)} 张图片，将从断点继续...")
    
    # 解析标注文件
    img_faces = create_pnet_data_txt_parser(txt_path, img_dir)
    if not img_faces:
        print("没有有效的图片数据，无法生成数据集")
        return
    
    total_face = 0
    total_non_face = 0
    new_processed = 0  # 记录本次处理的新图片数量
    
    # 处理每张图片（跳过已处理的）
    for img_path, face_boxes in img_faces:
        # 用图片路径作为唯一标识，判断是否已处理
        if img_path in processed_imgs:
            if DEBUG:
                print(f"已处理，跳过图片: {img_path}")
            continue
        
        if DEBUG:
            print(f"处理图片: {img_path}，人脸数量: {len(face_boxes)}")
        
        # 保存人脸图片
        face_count = crop_and_save_face(img_path, face_boxes, face_save_dir, crop_size)
        total_face += face_count
        
        # 保存非人脸图片
        non_face_count = crop_and_save_non_face(
            img_path, face_boxes, non_face_save_dir, crop_size, num_per_face=1
        )
        total_non_face += non_face_count
        
        # 标记为已处理
        processed_imgs.add(img_path)
        new_processed += 1
        
        # 每处理10张图片保存一次进度（可根据需要调整频率）
        if new_processed % 10 == 0:
            save_progress(processed_imgs)
            print(f"已临时保存进度，本次新增处理 {new_processed} 张图片")
    
    # 全部处理完成后保存最终进度
    save_progress(processed_imgs)
    
    print(f"数据集生成完成！")
    print(f"本次新增处理 {new_processed} 张图片")
    print(f"累计保存人脸图片: {total_face} 张")
    print(f"累计保存非人脸图片: {total_non_face} 张")
    print(f"总处理进度: {len(processed_imgs)}/{len(img_faces)} 张图片")


def landmark_dataset_txt_parser(txt_path, img_dir):
    """
    :param txt_path:
    :param img_dir:
    :return: [absolute_img_path,[x1,x2,y1,y2],(x,y)of[left_eye,right_eye,nose,mouse_left, mouse_right]]
    """
    if osp.exists(txt_path):
        # *** img_faces shape :[img_path,[faces_num, 4]]
        img_faces = []
        with open(txt_path, 'r') as f:
            l = []
            lines = list(map(lambda line: line.strip().split('\n'), f))
            # lines[[str],[str],[]...]
            lines = [i[0].split(' ') for i in lines]
            # lines [[path_str,pos_str]...]
            for line in lines:
                # 将路径中的'\'替换为'/'
                img_path = line[0].replace('\\', '/')
                faces_pos = [int(i) for i in line[1:5]]
                # 标注为 左右眼，嘴，左右嘴角
                landmark = [float(i) for i in line[5:]]
                real_img_path = osp.join(img_dir, img_path)
                # if DEBUG: print(real_img_path)
                # if DEBUG: print(osp.exists(real_img_path), Image.open(real_img_path).verify())
                if osp.exists(real_img_path):
                    try:
                        Image.open(real_img_path).verify()
                        img_faces.append([real_img_path, faces_pos, landmark])
                        if DEBUG: print('Valid image')
                    except Exception:
                        if DEBUG: print('Invalid image')
                else:
                    print("*** warning:image path invalid")

        # for i in img_faces: print(i)
        return img_faces
    else:
        print('*** warning:WILDER_FACE txt file not exist!')

def create_rnet_data(save_dir_name='R_Net_dataset', crop_size=48):
    # 新增：进度记录相关配置
    PROGRESS_FILE = "RNet_ldmk_progress.pkl"  # 单独的进度文件，避免冲突
    
    def load_progress():
        """加载已处理的图片记录"""
        if osp.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'rb') as f:
                return pickle.load(f)
        return set()
    
    def save_progress(processed):
        """保存已处理的图片记录"""
        with open(PROGRESS_FILE, 'wb') as f:
            pickle.dump(processed, f)

    def get_name_from_path(img_path):
        return osp.splitext(osp.split(img_path)[1])[0]

    def make_dir(save_dir):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

    def crop_img(img_np, crop_box, crop_size):
        crop_img_np = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :]
        crop_img = Image.fromarray(crop_img_np)
        crop_img = crop_img.resize((crop_size, crop_size), resample=PIL.Image.BILINEAR)
        return crop_img

    def cal_landmark_offset(box, ldmk):
        if ldmk is None:
            return []
        else:
            minx, miny = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ldmk_offset = [(ldmk[i] - [minx, miny][i % 2]) / float([w, h][i % 2]) for i in range(len(ldmk))]
            return ldmk_offset

    def txt_to_write(path, ldmk_offset):
        s = f'{path} '
        s += ' '.join(f'{i}' for i in ldmk_offset)
        s += '\n'
        return s
    
    # 加载已处理的图片记录
    processed_imgs = load_progress()
    print(f"已处理 {len(processed_imgs)} 张图片，将从断点继续...")
    
    # 解析标注文件
    img_faces = landmark_dataset_txt_parser(
        txt_path=r"D:\systemDir\Desktop\experiment\CNN\cnn_facepoint_dataset\trainImageList.txt",
        img_dir=r"D:\systemDir\Desktop\experiment\CNN\cnn_facepoint_dataset"
    )
    if not img_faces:
        print("没有有效的图片数据，无法生成数据集")
        return
    
    output_path = r"D:\systemDir\Desktop\experiment\CNN\dataset\ldmk_faces"
    make_dir(output_path)
    txt_path = osp.join(output_path, f'{save_dir_name}.txt')
    
    # 新增：如果是新启动的任务，清空旧的标注文件（避免重复写入）
    if not processed_imgs:
        with open(txt_path, 'w') as f:
            pass  # 清空文件
    
    txt = open(txt_path, 'a')
    new_processed = 0  # 记录本次处理的新图片数量
    
    for img_face in img_faces:
        img_path = img_face[0]
        # 用图片路径作为唯一标识，判断是否已处理
        if img_path in processed_imgs:
            if DEBUG:
                print(f"已处理，跳过图片: {img_path}")
            continue
        
        try:
            img_name = get_name_from_path(img_path)
            save_dir = osp.join(output_path, f"{img_name}.jpg")  # 补充.jpg后缀，解决无后缀问题
            faces = np.array(img_face[1])
            faces = np.expand_dims(faces, 0)
            faces[:, :] = faces[:, (0, 2, 1, 3)]
            faces = faces[0]
            
            ldmk = [int(i) for i in img_face[2]]
            with Image.open(img_path) as img:  # 使用with语句确保图片正确关闭
                img_np = np.array(img)

            img_box = crop_img(img_np=img_np, crop_box=faces, crop_size=crop_size)
            img_box.save(save_dir)  # 保存带.jpg后缀的图片
            ldmk_offset = cal_landmark_offset(faces, ldmk)
            txt.write(txt_to_write(osp.relpath(save_dir, osp.split(txt_path)[0]), ldmk_offset))
            
            # 标记为已处理
            processed_imgs.add(img_path)
            new_processed += 1
            
            # 每处理10张图片保存一次进度
            if new_processed % 10 == 0:
                save_progress(processed_imgs)
                print(f"已临时保存进度，本次新增处理 {new_processed} 张图片")
                
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
            continue  # 出错时跳过当前图片，继续处理下一张
    
    # 全部处理完成后保存最终进度
    save_progress(processed_imgs)
    txt.close()
    
    print(f"数据集生成完成！")
    print(f"本次新增处理 {new_processed} 张图片")
    print(f"总处理进度: {len(processed_imgs)}/{len(img_faces)} 张图片")

if __name__ == '__main__':
    # 配置路径（请根据实际情况修改）
    TXT_PATH = r"D:\systemDir\Desktop\experiment\CNN\widerface_dataset\wider_face_split\wider_face_train_bbx_gt.txt"
    IMG_DIR = r"D:\systemDir\Desktop\experiment\CNN\widerface_dataset\WIDER_train\images"
    FACE_SAVE_DIR = r"D:\systemDir\Desktop\experiment\CNN\dataset\face_24x24_1"
    NON_FACE_SAVE_DIR = r"D:\systemDir\Desktop\experiment\CNN\dataset\nonface_24x24_0"
    
    try:
        create_rnet_data(save_dir_name='R_Net_dataset', crop_size=48)
    except KeyboardInterrupt:
        # 捕获Ctrl+C中断信号，确保进度被保存
        print("\n检测到中断信号，正在保存进度...")
        save_progress(load_progress())
        print("进度已保存，下次运行将从断点继续")