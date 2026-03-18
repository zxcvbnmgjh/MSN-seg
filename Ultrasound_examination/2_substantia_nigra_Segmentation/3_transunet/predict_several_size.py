# 预测之前先裁剪，裁剪到512*512之后进行模型预测，然后有返回到原始尺寸
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

    # 图像预处理（注意：这里不包含resize，因为我们要自己做中心裁剪）
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

                # 2. 创建一个与原始图像大小相同的全黑mask（作为最终结果的容器）
                full_mask = np.zeros((original_height, original_width), dtype=np.uint8)

                # 3. 中心裁剪
                target_size = args.img_size # 512
                left = (original_width - target_size) // 2
                top = (original_height - target_size) // 2
                right = left + target_size
                bottom = top + target_size

                # 检查是否需要裁剪（图像是否大于512x512）
                if original_width >= target_size and original_height >= target_size:
                    # 执行中心裁剪
                    cropped_img = original_img.crop((left, top, right, bottom))
                    print(f"  Center cropped to {target_size}x{target_size}")
                else:
                    # 如果原始图像比512x512小，则先填充再裁剪
                    # 计算需要填充的像素
                    pad_left = max((target_size - original_width) // 2, 0)
                    pad_top = max((target_size - original_height) // 2, 0)
                    pad_right = max(target_size - original_width - pad_left, 0)
                    pad_bottom = max(target_size - original_height - pad_top, 0)
                    
                    # 使用transforms进行填充（填充0，即黑色）
                    pad_transform = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)
                    padded_img = pad_transform(original_img)
                    # 再进行中心裁剪，确保是512x512
                    cropped_img = padded_img.crop((left, top, right, bottom))
                    print(f"  Padded and cropped to {target_size}x{target_size}")

                # 4. 对裁剪后的图像进行预处理和推理
                img_tensor = data_transform(cropped_img)
                img_tensor = torch.unsqueeze(img_tensor, dim=0)  # 添加 batch 维度

                t_start = time_synchronized()
                output = model(img_tensor.to(device))
                t_end = time_synchronized()

                # 5. 处理输出
                prediction = output['out'].argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                prediction[prediction == 1] = 255  # 将前景像素值设置为 255（白色）

                # 6. 将512x512的预测结果粘贴回全黑mask的对应位置
                # 注意：prediction是二维数组，需要与full_mask的切片匹配
                full_mask[top:bottom, left:right] = prediction

                # 7. 生成输出文件名并保存
                base_name, ext = os.path.splitext(img_name)
                output_name = f"{base_name}_mask{ext}"
                output_path = os.path.join(output_folder, output_name)

                mask_image = Image.fromarray(full_mask)
                mask_image.save(output_path)

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
    args = parse_args() # 注意：需要调用parse_args()来获取参数
    main(args) # 传入args参数