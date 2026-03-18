import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from nets.segformer import SegFormer


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/2_segformer/save_weights/5fold/fold_4/best_model.pth"   # 最优权重地址
    input_folder = "/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/0_sn_data/test/fold4/images"  # 输入文件夹路径
    output_folder = "/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/2_segformer/predict_results/test/fold4"  # 输出文件夹路径

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 检查权重文件是否存在
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(input_folder), f"input folder {input_folder} not found."

    mean = (0.1677215,0.1677722,0.16774629)
    std=(0.22441734,0.22444354,0.22439982)

    # 获取设备
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 创建模型并加载权重
    model = SegFormer(num_classes=classes + 1, phi='b3', pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

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
                # 加载图像
                original_img = Image.open(img_path).convert('RGB')
                img = data_transform(original_img)
                img = torch.unsqueeze(img, dim=0)  # 添加 batch 维度

                # 推理
                t_start = time_synchronized()
                output = model(img.to(device))
                t_end = time_synchronized()

                # 处理输出
                prediction = output['out'].argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                prediction[prediction == 1] = 255  # 将前景像素值设置为 255（白色）

                # 生成输出文件名
                base_name, ext = os.path.splitext(img_name)  # 分离文件名和扩展名
                output_name = f"{base_name}_mask{ext}"  # 添加 _mask 后缀
                output_path = os.path.join(output_folder, output_name)

                # 保存预测结果
                mask = Image.fromarray(prediction)
                mask.save(output_path)

                print(f"Processed {img_name}, inference time: {t_end - t_start:.4f}s, saved to {output_path}")

            except Exception as e:
                print(f"Error processing {img_name}: {e}")


if __name__ == '__main__':
    main()