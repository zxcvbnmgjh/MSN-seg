# 预测的是中心裁剪的部分
import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from transunet_official_net.vit_seg_modeling import VisionTransformer as ViT_seg 
from transunet_official_net.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main(args):
    weights_path = "/data2/gaojiahao/1_mid_brain_Segmentation/3_TransUNet-plpa/save_weights/best_model.pth"
    input_folder = "/data2/gaojiahao/1_mid_brain_Segmentation/0_midbrain_data/test/images"  # 输入文件夹路径
    output_folder = "/data2/gaojiahao/1_mid_brain_Segmentation/3_TransUNet-plpa/predict_results"

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 检查权重文件是否存在
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(input_folder), f"input folder {input_folder} not found."

    mean = (0.12311059, 0.12312306, 0.12311852)
    std = (0.23592656, 0.23593814, 0.23593553)

    # 获取设备
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 创建模型
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    model.to(device)

    # 加载预训练权重
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # 图像预处理
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 列出输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    if not image_files:
        print(f"No images found in the input folder: {input_folder}")
        return

    model.eval()  # 进入验证模式
    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(input_folder, img_name)
            try:
                # 1. 加载原始图像
                original_img = Image.open(img_path).convert('RGB')
                original_width, original_height = original_img.size
                print(f"Processing {img_name}: Original size {original_width}x{original_height}")

                # 2. 中心裁剪到512x512
                target_size = args.img_size # 512
                # 计算裁剪的边界
                left = (original_width - target_size) // 2
                top = (original_height - target_size) // 2
                right = left + target_size
                bottom = top + target_size
                
                # 执行裁剪
                # 注意：PIL的crop方法是 (left, top, right, bottom)
                cropped_img = original_img.crop((left, top, right, bottom))
                print(f"  Center cropped to {target_size}x{target_size}")

                # 3. 对裁剪后的图像进行预处理和推理
                img = data_transform(cropped_img)
                img = torch.unsqueeze(img, dim=0)  # 添加 batch 维度

                t_start = time_synchronized()
                output = model(img.to(device))
                t_end = time_synchronized()

                # 4. 处理输出
                prediction = output['out'].argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                prediction[prediction == 1] = 255  # 将前景像素值设置为 255（白色）

                # 5. 生成输出文件名并保存
                base_name, ext = os.path.splitext(img_name)
                output_name = f"{base_name}_mask{ext}"
                output_path = os.path.join(output_folder, output_name)

                # 6. 保存预测结果
                # 注意：此时的prediction是512x512的，直接保存
                mask = Image.fromarray(prediction)
                mask.save(output_path)

                print(f"Processed {img_name}, inference time: {t_end - t_start:.4f}s, saved to {output_path}")

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--data-path", default="../", help="DRIVE root")
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--device", default="cuda:3", help="training device")
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument("--amp", default=False, type=bool, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--resume', default='/data2/gaojiahao/1_mid_brain_Segmentation/3_TransUNet-plpa/save_weights/best_model.pth', help='resume from checkpoint')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)