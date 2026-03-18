import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import deeplabv3_resnet50


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # ------------------------------- 配置参数 ------------------------------- #
    aux = False  # 推理时不需要辅助分类器
    classes = 1  # 二分类：背景 + 前景（中脑）
    
    # 模型权重路径
    weights_path = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/4_deeplabv3/save_weights/5fold/fold_4/best_model.pth"
    
    # 输入图像文件夹路径
    input_folder = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/test/fold4/images"
    
    # 输出分割结果文件夹路径
    output_folder = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/4_deeplabv3/predict_results/test/fold4"
    
    # 确保路径存在
    assert os.path.exists(weights_path), f"权重文件不存在: {weights_path}"
    assert os.path.exists(input_folder), f"输入文件夹不存在: {input_folder}"
    
    os.makedirs(output_folder, exist_ok=True)

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ------------------------------- 创建模型 ------------------------------- #
    model = deeplabv3_resnet50(aux=aux, num_classes=classes + 1)
    
    # 加载权重（过滤 aux_classifier）
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    filtered_weights = {k: v for k, v in weights_dict.items() if "aux_classifier" not in k}
    model.load_state_dict(filtered_weights)
    model.to(device)
    model.eval()

    print(f"模型权重已加载: {weights_path}")

    # ------------------------------- 预处理变换 ------------------------------- #
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.17225, 0.17229316, 0.17226526),
            std=(0.22833531, 0.22835146, 0.22830528)
        )
    ])

    # ------------------------------- 遍历文件夹进行推理 ------------------------------- #
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    total_time = 0.0
    processed_count = 0

    with torch.no_grad():
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(image_extensions):
                input_path = os.path.join(input_folder, filename)
                
                # ✅ 修改点：在原文件名后添加 "_mask" 后缀
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"{name}_mask{ext}")

                try:
                    # 加载图像
                    original_img = Image.open(input_path).convert("RGB")
                    original_size = original_img.size  # (width, height)

                    # 预处理
                    img = data_transform(original_img)
                    img = img.unsqueeze(0).to(device)  # 增加 batch 维度并送入设备

                    # 推理
                    t_start = time_synchronized()
                    output = model(img)
                    t_end = time_synchronized()

                    inference_time = t_end - t_start
                    total_time += inference_time

                    # 生成分割 mask
                    prediction = output['out'].argmax(1).squeeze(0).to("cpu").numpy().astype(np.uint8)
                    mask = np.zeros(prediction.shape, dtype=np.uint8)
                    mask[prediction == 1] = 255  # 前景设为白色

                    # 保存为灰度图（保持原始图像尺寸）
                    mask_img = Image.fromarray(mask, mode='L')
                    mask_img.save(output_path)
                    
                    print(f"已处理: {filename} | 推理耗时: {inference_time:.3f}s | 保存至: {output_path}")
                    processed_count += 1

                except Exception as e:
                    print(f"处理 {filename} 时出错: {e}")
                    continue

    # ------------------------------- 总结 ------------------------------- #
    if processed_count > 0:
        avg_time = total_time / processed_count
        print(f"\n✅ 所有图像处理完成！")
        print(f"共处理 {processed_count} 张图像")
        print(f"平均推理时间: {avg_time:.3f}s/张")
        print(f"结果保存在: {output_folder}")
    else:
        print("⚠️ 未找到任何支持的图像文件。")


if __name__ == '__main__':
    main()