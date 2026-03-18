import numpy as np
import torch
from PIL import Image
from transunet_official_net.vit_seg_modeling import VisionTransformer as ViT_seg
from transunet_official_net.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import transforms as T


class SegmentationPresetTest:
    def __init__(self, crop_size,mean=(0.16666081,0.16667844,0.16667177), std=(0.26213387,0.26214503,0.26214192)):
        trans = [T.RandomCrop(crop_size)]
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(mean=(0.16666081,0.16667844,0.16667177), std=(0.26213387,0.26214503,0.26214192)):
    crop_size = 512
    return SegmentationPresetTest(crop_size,mean=mean, std=std)
def sliding_window_predict(model, image, window_size=512, stride=None, device='cuda'):
    """
    使用滑动窗口对任意尺寸图像进行预测，并融合重叠区域。
    
    参数:
        model: 分割模型（已加载权重）
        image: 输入图像 (H, W, C)，np.array，RGB格式
        window_size: 窗口大小，默认 512
        stride: 步长，默认为 window_size（无重叠），建议设置为 window_size // 2 实现重叠融合
        device: 推理设备 ('cuda' or 'cpu')
        
    返回:
        full_mask: 与原图等大的分割结果 (H, W)
    """
    if stride is None:
        stride = window_size  # 默认无重叠

    H, W = image.shape[0], image.shape[1]
    full_mask = np.zeros((H, W), dtype=np.float32)  # 存储最终 mask
    full_count = np.zeros((H, W), dtype=np.float32)  # 记录每个像素被预测的次数（用于平均）

    model.eval()
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                # 确保窗口不越界
                x_end = min(x + window_size, W)
                y_end = min(y + window_size, H)
                img_patch = image[y:y_end, x:x_end]

                # 如果 patch 尺寸小于 window_size，则 padding
                pad_x = window_size - img_patch.shape[1]
                pad_y = window_size - img_patch.shape[0]
                img_patch_padded = np.pad(img_patch, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')

                # 图像预处理（根据你的模型输入方式调整）
                img_tensor = torch.from_numpy(img_patch_padded).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)

                # 模型预测
                pred = model(img_tensor)  # 假设输出 shape: [1, 1, H, W] 或 [1, num_classes, H, W]
                pred = pred['out']

                if isinstance(pred, (tuple, list)):
                    pred = pred[0]

                pred = torch.sigmoid(pred).cpu().numpy()[0, 0]  # 假设是二分类任务
                pred = pred[:img_patch.shape[0], :img_patch.shape[1]]

                # 更新 full_mask 和 full_count
                full_mask[y:y_end, x:x_end] += pred
                full_count[y:y_end, x:x_end] += 1

        # 平均融合重叠区域
        full_count[full_count == 0] = 1e-6  # 防止除零
        full_mask = full_mask / full_count

    # 二值化处理（可选）
    full_mask = (full_mask > 0.5).astype(np.uint8) * 255

    return full_mask


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="TransUNet 推理脚本")
    parser.add_argument("--data-path", default="../", help="数据集根目录")
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument('--img_size', type=int, default=512, help='网络输入图像大小')
    parser.add_argument("--pretrained", action='store_true', help="是否使用预训练模型")
    parser.add_argument("--device", default="cuda:3", help="推理设备")
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='选择ViT模型')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT Patch 大小')
    parser.add_argument('--n_skip', type=int, default=3, help='跳接层数')
    parser.add_argument('--resume', default='/data2/gaojiahao/1_mid_brain_Segmentation/3_TransUNet-plpa/save_weights/best_model.pth', help='resume from checkpoint')
    parser.add_argument('--input-image', required=True, type=str, help='输入图像路径')
    parser.add_argument('--output-path', required=True, type=str, help='输出mask保存路径')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # 图像读取（使用 PIL 替代 cv2）
    raw_img = Image.open(args.input_image).convert('RGB')  # 自动转为 RGB
    raw_img = np.array(raw_img)  # shape: (H, W, 3)

    # 模型配置
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size)
        )
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    device = args.device if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 加载模型权重
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # 滑动窗口预测
    pred_mask = sliding_window_predict(
        model,
        raw_img,
        window_size=512,
        stride=256,  # 50% overlap
        device=device
    )

    # 保存结果（使用 PIL 替代 cv2.imwrite）
    output_image = Image.fromarray(pred_mask)
    output_image.save(args.output_path)
    print(f"✅ 分割完成，结果已保存至：{args.output_path}")