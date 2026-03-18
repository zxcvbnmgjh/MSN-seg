import os
from PIL import Image


def list_png_sorted(folder: str):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def check_and_fix(folder_rgb: str, folder_gray: str, overwrite_rgb_fix: bool = True):
    rgb_files = list_png_sorted(folder_rgb)
    gray_files = list_png_sorted(folder_gray)

    # 结果记录
    gray_in_rgb_folder = []      # 文件夹1中灰度图（将被无损转RGB）
    non_rgb_in_rgb_folder = []   # 文件夹1中非RGB且非灰度（例如 RGBA/P/CMYK/读取失败等）
    non_gray_in_gray_folder = [] # 文件夹2中不是灰度图
    size_mismatch_pairs = []     # 尺寸不一致

    print(f"Folder1(RGB期望): {folder_rgb} | PNG数={len(rgb_files)}")
    print(f"Folder2(灰度期望): {folder_gray} | PNG数={len(gray_files)}\n")

    # -------------------------
    # 1) 检查 folder_rgb：是否都是RGB；若灰度则无损转RGB
    # -------------------------
    for name in rgb_files:
        path = os.path.join(folder_rgb, name)
        try:
            with Image.open(path) as img:
                mode = img.mode
                if mode == "RGB":
                    continue
                elif mode == "L":
                    gray_in_rgb_folder.append(name)
                    # 无损灰度->RGB：R=G=B=L（像素信息不丢失）
                    rgb_img = img.convert("RGB")
                    if overwrite_rgb_fix:
                        rgb_img.save(path)  # 覆盖原文件
                    else:
                        # 若不覆盖，可改为另存：
                        new_path = os.path.join(folder_rgb, os.path.splitext(name)[0] + "_toRGB.png")
                        rgb_img.save(new_path)
                else:
                    non_rgb_in_rgb_folder.append((name, mode))
        except Exception as e:
            non_rgb_in_rgb_folder.append((name, f"读取失败: {e}"))

    # -------------------------
    # 2) 检查 folder_gray：是否都是灰度
    # -------------------------
    for name in gray_files:
        path = os.path.join(folder_gray, name)
        try:
            with Image.open(path) as img:
                mode = img.mode
                if mode != "L":
                    non_gray_in_gray_folder.append((name, mode))
        except Exception as e:
            non_gray_in_gray_folder.append((name, f"读取失败: {e}"))

    # -------------------------
    # 3) 检查尺寸一致性：按排序后一一对应
    # -------------------------
    min_len = min(len(rgb_files), len(gray_files))
    if len(rgb_files) != len(gray_files):
        print(f"⚠️ 警告：两个文件夹 PNG 数量不一致，将仅检查前 {min_len} 对（按排序对应）。\n")

    for i in range(min_len):
        rgb_name = rgb_files[i]
        gray_name = gray_files[i]
        rgb_path = os.path.join(folder_rgb, rgb_name)
        gray_path = os.path.join(folder_gray, gray_name)

        try:
            with Image.open(rgb_path) as img1, Image.open(gray_path) as img2:
                if img1.size != img2.size:
                    size_mismatch_pairs.append((rgb_name, gray_name, img1.size, img2.size))
        except Exception as e:
            size_mismatch_pairs.append((rgb_name, gray_name, "读取失败", str(e)))

    # -------------------------
    # 输出报告
    # -------------------------
    print("========== 检查报告 ==========\n")

    print("【A】文件夹1（期望RGB）中发现的灰度图（已无损转为RGB）：" if overwrite_rgb_fix
          else "【A】文件夹1（期望RGB）中发现的灰度图（已无损转为RGB并另存）：")
    if gray_in_rgb_folder:
        for n in gray_in_rgb_folder:
            print(f"  - {n}")
    else:
        print("  ✓ 未发现灰度图")

    print("\n【B】文件夹1中不是RGB且不是灰度(L)的图片（需要人工确认/处理）：")
    if non_rgb_in_rgb_folder:
        for n, m in non_rgb_in_rgb_folder:
            print(f"  - {n} (mode={m})")
    else:
        print("  ✓ 全部为RGB（或已将灰度无损转RGB）")

    print("\n【C】文件夹2（期望灰度L）中不是灰度的图片：")
    if non_gray_in_gray_folder:
        for n, m in non_gray_in_gray_folder:
            print(f"  - {n} (mode={m})")
    else:
        print("  ✓ 全部为灰度图(L)")

    print("\n【D】按顺序对应的图片尺寸不一致的图片对：")
    if size_mismatch_pairs:
        for a, b, s1, s2 in size_mismatch_pairs:
            print(f"  - {a} vs {b} | folder1 size={s1}, folder2 size={s2}")
    else:
        print("  ✓ 所有对应图片尺寸一致")

    print("\n========== 完成 ==========")


if __name__ == "__main__":
    folder1_rgb = r"/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/full/images"   # 期望RGB（若有灰度将无损转RGB）
    folder2_gray = r"/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/full/masks"  # 期望灰度L

    # overwrite_rgb_fix=True 会覆盖保存灰度图转换后的RGB图（推荐，保证后续训练一致）
    check_and_fix(folder1_rgb, folder2_gray, overwrite_rgb_fix=True)
