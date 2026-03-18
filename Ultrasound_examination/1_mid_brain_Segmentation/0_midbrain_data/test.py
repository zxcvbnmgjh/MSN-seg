import os
from PIL import Image

def compare_images_by_index(folder1, folder2):
    """
    比较两个文件夹中按顺序排列的 PNG 图像文件（索引对齐）是否内容完全一致。
    文件名可以不同，只比较相同索引位置的图像内容。

    :param folder1: 第一个文件夹路径
    :param folder2: 第二个文件夹路径
    """
    # 检查路径
    if not os.path.isdir(folder1):
        print(f"❌ 错误：'{folder1}' 不是一个有效的文件夹路径。")
        return
    if not os.path.isdir(folder2):
        print(f"❌ 错误：'{folder2}' 不是一个有效的文件夹路径。")
        return

    # 获取所有 .png 文件（不区分大小写），并按名称排序以保证顺序一致
    def get_sorted_png_paths(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
        files.sort()  # 排序确保顺序一致（如：a.png, b.png）
        return [os.path.join(folder, f) for f in files]

    paths1 = get_sorted_png_paths(folder1)
    paths2 = get_sorted_png_paths(folder2)

    print(f"📁 '{os.path.basename(folder1)}': 找到 {len(paths1)} 个 PNG 文件")
    print(f"📁 '{os.path.basename(folder2)}': 找到 {len(paths2)} 个 PNG 文件")
    print("-" * 70)

    # 开始逐对比较
    min_count = min(len(paths1), len(paths2))
    match_count = 0
    mismatch_count = 0

    for i in range(min_count):
        path1 = paths1[i]
        path2 = paths2[i]
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)

        print(f"[{i:2d}] 比较: '{name1}' vs '{name2}'", end=" ... ")

        try:
            with Image.open(path1) as img1, Image.open(path2) as img2:
                # 检查尺寸
                if img1.size != img2.size:
                    print("❌ 尺寸不同")
                    mismatch_count += 1
                    continue

                # 检查模式（如 RGB, RGBA, L 等）
                if img1.mode != img2.mode:
                    print("❌ 图像模式不同")
                    mismatch_count += 1
                    continue

                # 获取像素数据并比较
                pixels1 = list(img1.getdata())
                pixels2 = list(img2.getdata())

                if pixels1 == pixels2:
                    print("✅ 像素完全一致")
                    match_count += 1
                else:
                    print("❌ 像素不同")
                    mismatch_count += 1

        except Exception as e:
            print(f"❌ 读取错误: {e}")
            mismatch_count += 1

    # 检查数量不匹配
    if len(paths1) != len(paths2):
        print("-" * 70)
        print("⚠️ 警告：文件数量不一致！")
        if len(paths1) > len(paths2):
            extra = [os.path.basename(p) for p in paths1[min_count:]]
            print(f"    '{os.path.basename(folder1)}' 多出 {len(extra)} 个文件: {extra}")
        else:
            extra = [os.path.basename(p) for p in paths2[min_count:]]
            print(f"    '{os.path.basename(folder2)}' 多出 {len(extra)} 个文件: {extra}")

    print("-" * 70)
    print(f"✅ 匹配: {match_count} 对 | ❌ 不匹配: {mismatch_count} 对")
    total = match_count + mismatch_count
    if total > 0:
        accuracy = 100 * match_count / total
        print(f"📊 一致率: {accuracy:.1f}%")
    else:
        print("📊 无有效配对文件。")


# ============ 使用示例 ============
if __name__ == "__main__":
    # ✅ 方法1：直接写死路径（适合固定任务）
    folder_a = r"/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/0_sn_data/full/masks"
    folder_b = r"/data2/gaojiahao/Ultrasound_examination/2_substantia_nigra_Segmentation/0_sn_data/raw_data/masks"

    # ✅ 方法2：运行时输入（取消下面注释）
    # folder_a = input("请输入第一个文件夹路径：").strip()
    # folder_b = input("请输入第二个文件夹路径：").strip()

    compare_images_by_index(folder_a, folder_b)