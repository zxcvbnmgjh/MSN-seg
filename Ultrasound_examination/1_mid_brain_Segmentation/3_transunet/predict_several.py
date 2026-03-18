import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from transunet_official_net.vit_seg_modeling import VisionTransformer as ViT_seg 
from transunet_official_net.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# -------------------------------
# 配置参数（直接在代码中指定）
# -------------------------------
# 模型权重路径
WEIGHTS_PATH = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/3_transunet/save_weights/5fold/fold_4/best_model.pth"

# 输入图像文件夹（所有图像必须是 512x512）
INPUT_FOLDER = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/test/fold4/images"

# 输出预测结果文件夹
OUTPUT_FOLDER = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/3_transunet/predict_results/test/fold4"

# 模型参数（根据训练时设置）
IMG_SIZE = 512
NUM_CLASSES = 2
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
VIT_NAME = "R50-ViT-B_16"
VIT_PATCHES_SIZE = 16
N_SKIP = 3

# 数据标准化参数（训练时计算得到）
MEAN = (0.17225,0.17229316 ,0.17226526)
STD = (0.22833531, 0.22835146, 0.22830528)


def time_synchronized():
    """同步 GPU 时间以准确测量推理耗时"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 检查文件路径
    assert os.path.exists(WEIGHTS_PATH), f"权重文件不存在: {WEIGHTS_PATH}"
    assert os.path.exists(INPUT_FOLDER), f"输入文件夹不存在: {INPUT_FOLDER}"

    print(f"使用设备: {DEVICE}")
    print(f"加载权重: {WEIGHTS_PATH}")
    print(f"输入图像路径: {INPUT_FOLDER}")
    print(f"输出预测路径: {OUTPUT_FOLDER}")

    # -------------------------------
    # 构建模型
    # -------------------------------
    config_vit = CONFIGS_ViT_seg[VIT_NAME]
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = N_SKIP

    if 'R50' in VIT_NAME:
        grid_size = IMG_SIZE // VIT_PATCHES_SIZE
        config_vit.patches.grid = (grid_size, grid_size)

    model = ViT_seg(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # -------------------------------
    # 加载权重
    # -------------------------------
    checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 设置为评估模式

    print("✅ 模型加载完成")

    # -------------------------------
    # 图像预处理
    # -------------------------------
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # -------------------------------
    # 获取图像列表
    # -------------------------------
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"❌ 在 {INPUT_FOLDER} 中未找到图像文件")
        return

    print(f"共发现 {len(image_files)} 张图像，开始推理...")

    # -------------------------------
    # 推理循环
    # -------------------------------
    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(INPUT_FOLDER, img_name)
            try:
                # 1. 加载图像（已为 512x512，无需裁剪）
                image = Image.open(img_path).convert('RGB')
                w, h = image.size

                # 可选：检查尺寸是否正确
                if w != IMG_SIZE or h != IMG_SIZE:
                    print(f"⚠️ 跳过 {img_name}：尺寸为 {w}x{h}，期望 {IMG_SIZE}x{IMG_SIZE}")
                    continue

                # 2. 预处理
                img_tensor = data_transform(image)  # [C, H, W]
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # [1, C, H, W]

                # 3. 推理
                t_start = time_synchronized()
                output = model(img_tensor)
                t_end = time_synchronized()

                # 4. 后处理：获取预测结果
                if isinstance(output, dict):
                    logits = output['out']
                else:
                    logits = output

                # 分类后处理
                if NUM_CLASSES > 1:
                    pred_mask = logits.argmax(dim=1).squeeze(0)  # 多类分割
                else:
                    pred_mask = (torch.sigmoid(logits) > 0.5).squeeze(0).squeeze(0)  # 二分类

                # 转为 numpy 并映射到 0/255
                pred_array = pred_mask.cpu().numpy().astype(np.uint8)
                pred_array = pred_array * 255  # 前景设为白色

                # 5. 保存预测 mask
                base_name, ext = os.path.splitext(img_name)
                save_name = f"{base_name}_mask{ext}"
                save_path = os.path.join(OUTPUT_FOLDER, save_name)

                result_img = Image.fromarray(pred_array, mode='L')  # 单通道灰度图
                result_img.save(save_path)

                print(f"✅ {img_name} | 推理耗时: {t_end - t_start:.4f}s | 已保存: {save_path}")

            except Exception as e:
                print(f"❌ 处理 {img_name} 时出错: {e}")

    print("✅ 所有图像推理完成！")


if __name__ == "__main__":
    main()