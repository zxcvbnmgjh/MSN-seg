import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import UNet


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def main():
    classes = 1  # foreground classes (exclude background)
    weights_path = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/1_unet/save_weights/5fold/fold_0/best_model.pth"
    input_folder = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/Test(512*512)groundtruth/fold1/images"
    output_folder = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/1_unet/predict_results/predicts/fold0_0"

    os.makedirs(output_folder, exist_ok=True)
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(input_folder), f"input folder {input_folder} not found."

    mean = (0.17225, 0.17229316, 0.17226526)
    std = (0.22833531, 0.22835146, 0.22830528)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # 和训练/测试一致：CenterCrop(512) + ToTensor + Normalize
    data_transform = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    if not image_files:
        print(f"No images found in: {input_folder}")
        return

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(input_folder, img_name)
            try:
                original_img = Image.open(img_path).convert("RGB")
                img = data_transform(original_img).unsqueeze(0).to(device)  # [1,3,512,512]

                t_start = time_synchronized()
                out = model(img)
                t_end = time_synchronized()

                logits = out["out"] if isinstance(out, dict) and "out" in out else out  # [1,C,H,W]
                pred_label = torch.argmax(logits, dim=1).squeeze(0)  # [H,W], 0/1

                pred_np = pred_label.cpu().numpy().astype(np.uint8)
                pred_mask_255 = (pred_np > 0).astype(np.uint8) * 255  # 0/255

                base_name, _ = os.path.splitext(img_name)
                output_name = f"{base_name}_predict.png"   # 统一输出 png + 命名规范
                output_path = os.path.join(output_folder, output_name)

                Image.fromarray(pred_mask_255).save(output_path)

                print(f"Processed {img_name}, time: {t_end - t_start:.4f}s, saved: {output_path}")

            except Exception as e:
                print(f"Error processing {img_name}: {e}")


if __name__ == "__main__":
    main()
