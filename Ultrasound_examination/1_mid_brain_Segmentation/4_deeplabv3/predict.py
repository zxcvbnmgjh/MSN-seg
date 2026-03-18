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
    aux = False  # 推理时不需要辅助分类器
    classes = 1  # 二分类：背景 + 前景（中脑）
    weights_path = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/4_deeplabv3/save_weights/5fold/fold_3/best_model.pth"
    img_path = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/full/images_512/midbrain001_image.png"
    
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 创建模型
    model = deeplabv3_resnet50(aux=aux, num_classes=classes + 1)  # 2 个类别

    # 加载权重（移除 aux_classifier 权重）
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    # 过滤掉 aux_classifier 的参数
    filtered_weights = {k: v for k, v in weights_dict.items() if "aux_classifier" not in k}
    model.load_state_dict(filtered_weights)
    model.to(device)

    # 加载图像
    original_img = Image.open(img_path).convert('RGB')  # 确保是 3 通道

    # 预处理：调整大小、转 Tensor、归一化
    data_transform = transforms.Compose([
        # transforms.Resize(520),  # 与训练时一致
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.17225,0.17229316 ,0.17226526),
                            std = (0.22833531, 0.22835146, 0.22830528))
    ])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)  # 增加 batch 维度

    # 推理模式
    model.eval()
    with torch.no_grad():
        # 初始化模型（可选）
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        # 推理
        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("Inference time: {:.3f}s".format(t_end - t_start))

        # 获取预测结果
        prediction = output['out'].argmax(1).squeeze(0)  # (H, W)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)

        # 二值化：0 → 背景（黑），1 → 前景（白）
        mask = np.zeros(prediction.shape, dtype=np.uint8)
        mask[prediction == 1] = 255  # 前景设为白色

        # 转为 PIL 图像并保存
        mask_img = Image.fromarray(mask, mode='L')  # 'L' 表示单通道灰度图
        mask_img.save("test_result_mask.png")
        print("Segmentation result saved as 'test_result_mask.png'")

    # （可选）可视化原图和叠加图
    # 可以用 matplotlib 或 OpenCV 实现，这里省略


if __name__ == '__main__':
    main()