import os
from pathlib import Path

def rename_images_with_offset(folder_path, offset=280):
    """
    将文件夹中的图像按顺序重命名为 image{offset + index} 的形式。
    例如 offset=34，第一张图 → image35，第二张 → image36，依此类推。

    参数:
        folder_path (str): 图像所在文件夹路径
        offset (int): 偏移量，默认为 34
    """
    # 支持的图像扩展名（不区分大小写）
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"❌ 指定路径不是一个有效文件夹: {folder_path}")

    # 获取所有图像文件，并按名称排序（确保顺序稳定）
    image_files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_files.sort(key=lambda x: x.name)  # 可改为按时间排序，如 x.stat().st_mtime

    if not image_files:
        print("📁 文件夹中未找到任何图像文件。")
        return

    print(f"🔍 找到 {len(image_files)} 个图像文件，开始重命名（偏移量={offset}）...")

    # 为避免重命名过程中覆盖原文件（比如原文件就叫 image35.jpg），
    # 先将所有文件移到临时名称
    temp_files = []
    temp_suffix = ".tmp_renaming"
    for i, f in enumerate(image_files):
        temp_path = f.with_suffix(f.suffix + temp_suffix)
        f.rename(temp_path)
        temp_files.append(temp_path)

    # 现在从临时文件重命名为目标名称
    for idx, temp_file in enumerate(temp_files, start=1):
        new_number = offset + idx  # 34 + 1 = 35, 34 + 2 = 36, ...
        new_name = folder / f"image{new_number}{temp_file.suffix.replace(temp_suffix, '')}.png"
        
        if new_name.exists():
            print(f"⚠️ 警告: {new_name.name} 已存在，跳过 {temp_file.name}")
            # 如果需要，可恢复原名或报错
            continue

        temp_file.rename(new_name)
        print(f"✅ {temp_file.name} → {new_name.name}")

    print("🎉 重命名完成！")

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 👇 修改为你自己的图像文件夹路径
    your_image_folder = "/data2/gaojiahao/SN+project/dataset/preprocess_data/Ⅳ"  # 例如：r"C:\data\images" 或 "/home/user/images"

    try:
        rename_images_with_offset(your_image_folder, offset=722)
    except Exception as e:
        print(f"❌ 发生错误: {e}")