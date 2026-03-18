import os
import argparse
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Set

def validate_image_mask_sizes(image_dir: str, mask_dir: str) -> Tuple[int, int, List[str], List[str], List[str]]:
    """
    验证原图和掩码的尺寸一致性
    
    Args:
        image_dir: 原图文件夹路径
        mask_dir: 掩码文件夹路径
    
    Returns:
        Tuple: (匹配数量, 不匹配数量, 不匹配详情列表, 仅原图存在列表, 仅掩码存在列表)
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    
    if not image_dir.exists():
        raise ValueError(f"原图文件夹不存在: {image_dir}")
    if not mask_dir.exists():
        raise ValueError(f"掩码文件夹不存在: {mask_dir}")
    
    # 获取所有PNG文件
    image_files = {f for f in image_dir.glob("*.png") if f.is_file()}
    mask_files = {f for f in mask_dir.glob("*.png") if f.is_file()}
    
    # 提取基础名（去除 _image.png / _mask.png 后缀）
    image_basenames = {
        f.stem.replace('_image', ''): f 
        for f in image_files if f.stem.endswith('_image')
    }
    mask_basenames = {
        f.stem.replace('_mask', ''): f 
        for f in mask_files if f.stem.endswith('_mask')
    }
    
    # 匹配分析
    all_basenames = set(image_basenames.keys()) | set(mask_basenames.keys())
    matched_basenames = set(image_basenames.keys()) & set(mask_basenames.keys())
    
    only_images = set(image_basenames.keys()) - set(mask_basenames.keys())
    only_masks = set(mask_basenames.keys()) - set(image_basenames.keys())
    
    # 验证尺寸
    matched_count = 0
    mismatched_count = 0
    mismatch_details = []
    
    print(f"\n🔍 共找到 {len(image_files)} 张原图，{len(mask_files)} 个掩码")
    print(f"🔗 可匹配的文件对: {len(matched_basenames)}")
    print(f"⚠️  仅原图存在: {len(only_images)} | 仅掩码存在: {len(only_masks)}\n")
    
    for basename in sorted(matched_basenames):
        img_path = image_basenames[basename]
        mask_path = mask_basenames[basename]
        
        try:
            with Image.open(img_path) as img, Image.open(mask_path) as mask:
                img_size = img.size  # (width, height)
                mask_size = mask.size
                
                if img_size == mask_size:
                    matched_count += 1
                else:
                    mismatched_count += 1
                    mismatch_details.append(
                        f"❌ [{basename}] 尺寸不一致 | 原图: {img_size} | 掩码: {mask_size} "
                        f"| 原图路径: {img_path.name} | 掩码路径: {mask_path.name}"
                    )
        except Exception as e:
            mismatched_count += 1
            mismatch_details.append(
                f"⚠️  [{basename}] 读取失败: {str(e)} | 原图: {img_path.name} | 掩码: {mask_path.name}"
            )
    
    # 输出不匹配详情
    if mismatch_details:
        print("📏 尺寸不一致或读取失败的文件:")
        for detail in mismatch_details:
            print(f"  {detail}")
        print()
    
    # 输出缺失文件
    if only_images:
        print(f"🖼️  仅原图存在（{len(only_images)} 个）:")
        for bn in sorted(only_images):
            print(f"  • {image_basenames[bn].name}")
        print()
    
    if only_masks:
        print(f"🎭 仅掩码存在（{len(only_masks)} 个）:")
        for bn in sorted(only_masks):
            print(f"  • {mask_basenames[bn].name}")
        print()
    
    # 总结
    total_pairs = len(matched_basenames)
    print("=" * 70)
    print(f"✅ 尺寸一致: {matched_count} / {total_pairs}")
    print(f"❌ 尺寸不一致或读取失败: {mismatched_count} / {total_pairs}")
    print(f"📊 一致率: {matched_count / total_pairs * 100:.2f}%" if total_pairs > 0 else "📊 无匹配文件对")
    print("=" * 70)
    
    return matched_count, mismatched_count, mismatch_details, \
           [image_basenames[bn].name for bn in only_images], \
           [mask_basenames[bn].name for bn in only_masks]

def main():
    parser = argparse.ArgumentParser(description="验证原图和掩码尺寸一致性")
    parser.add_argument("--image_dir", type=str, default="/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/images", help="原图文件夹路径")
    parser.add_argument("--mask_dir", type=str, default="/data2/gaojiahao/Ultrasound_examination/1_mid_brain_Segmentation/0_midbrain_data/raw_data/masks", help="掩码文件夹路径")
    
    args = parser.parse_args()
    validate_image_mask_sizes(args.image_dir, args.mask_dir)

if __name__ == "__main__":
    main()