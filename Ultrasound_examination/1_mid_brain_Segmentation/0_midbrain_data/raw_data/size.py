from PIL import Image

def print_image_resolution(image_path: str):
    with Image.open(image_path) as img:
        width, height = img.size
        print(f"Image: {image_path}")
        print(f"Resolution: {width} x {height}")

# 示例
image_path = "/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/masks/lyl8_mask.png"
print_image_resolution(image_path)
