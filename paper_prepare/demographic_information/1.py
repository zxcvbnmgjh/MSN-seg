import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# -------------------------------
# 1. 设置绘图风格（使用 seaborn 美化）
# -------------------------------
sns.set_style("whitegrid")  # 更优雅的网格风格
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用支持英文的字体
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['figure.titlesize'] = 16

# -------------------------------
# 2. 读取 Excel 文件
# -------------------------------
file_path = '/data2/gaojiahao/paper_prepare/demographic_information/demographic information.xlsx'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"未找到文件: {file_path}")

# 读取数据（列名在第2行）
df = pd.read_excel(file_path, header=1)
df.dropna(how='all', inplace=True)

print(f"成功读取 {len(df)} 例患者数据。")

# 提取数据
ages = df.iloc[:, 2].dropna().astype(int)           # 年龄
genders = df.iloc[:, 1].dropna()                    # 性别
left_window = df.iloc[:, 3].dropna()                # 左侧颞窗
right_window = df.iloc[:, 4].dropna()               # 右侧颞窗
left_echo = df.iloc[:, 5].dropna()                  # 左侧回声
right_echo = df.iloc[:, 6].dropna()                 # 右侧回声

# -------------------------------
# 3. 数据处理
# -------------------------------
# 颞窗质量映射为英文
window_map = {'模糊': 'Blurred', '欠佳': 'Poor', '尚可': 'Passable', '良好': 'Good'}
left_window_en = left_window.map(window_map)
right_window_en = right_window.map(window_map)

# 🔻 修改：按您要求的顺序定义 order_window
order_window = ['Good', 'Passable', 'Poor', 'Blurred']  # 良好 -> 尚可 -> 欠佳 -> 模糊
left_window_counts = left_window_en.value_counts().reindex(order_window, fill_value=0)
right_window_counts = right_window_en.value_counts().reindex(order_window, fill_value=0)

# 回声等级（保持使用中文罗马数字，与数据一致）
echo_levels = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ']  # 移除了不存在的 'Ⅴ'
echo_left_counts = left_echo.value_counts().reindex(echo_levels, fill_value=0)
echo_right_counts = right_echo.value_counts().reindex(echo_levels, fill_value=0)

# 性别映射
gender_map = {'男': 'Male', '女': 'Female'}
genders_en = genders.map(gender_map)

# -------------------------------
# 4. 获取输出目录
# -------------------------------
output_dir = os.path.dirname(os.path.abspath(file_path))
if output_dir == '':
    output_dir = '.'

# -------------------------------
# 5. 图1: 年龄分布直方图 + KDE
# -------------------------------
plt.figure(figsize=(10, 8))
sns.histplot(ages, bins=15, kde=True, color='skyblue', alpha=0.8, edgecolor='black')
plt.xlabel('Age (years)', fontsize=24)
plt.ylabel('Density', fontsize=24)
#plt.title('Age Distribution of Patients', fontsize=22, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
sns.despine()  # 去除顶部和右侧边框
plt.tight_layout()
age_plot_path = os.path.join(output_dir, 'age_distribution.png')
plt.savefig(age_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 年龄分布图已保存: {age_plot_path}")

# -------------------------------
# 6. 图2: 性别分布饼图
# -------------------------------
plt.figure(figsize=(8, 6))
gender_counts = genders_en.value_counts()
colors = ['#66c2a5', '#fc8d62']  # 温和的绿/橙配色
wedges, texts, autotexts = plt.pie(
    gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
    startangle=90, colors=colors, textprops={'color': 'white', 'fontsize': 12},
    wedgeprops={'edgecolor': 'black', 'linewidth': 1}
)
plt.title('Gender Distribution', fontsize=14, fontweight='bold', pad=20)
# 美化百分比文字
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
plt.tight_layout()
gender_plot_path = os.path.join(output_dir, 'gender_distribution_pie.png')
plt.savefig(gender_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"✅ 性别分布图已保存: {gender_plot_path}")

# -------------------------------
# 7. 图3: 颞窗质量柱状图（修改为左右侧并列，并调整横坐标顺序）
# -------------------------------
plt.figure(figsize=(10, 8))  # 宽度稍增，以容纳更多标签

# 创建并列柱状图的数据
x = np.arange(len(order_window))  # x轴位置
width = 0.35  # 柱子的宽度

# 绘制左右侧的柱子
bars1 = plt.bar(x - width/2, left_window_counts, width, label='Left Side', color='#ff9999', alpha=0.85, edgecolor='black')
bars2 = plt.bar(x + width/2, right_window_counts, width, label='Right Side', color='#66b3ff', alpha=0.85, edgecolor='black')

plt.xlabel('Temporal Window Quality', fontsize=24)
plt.ylabel('Number of Cases', fontsize=24)
# plt.title('Bilateral Temporal Window Quality Distribution', fontsize=22, fontweight='bold', pad=20)
plt.xticks(x, order_window)  # 设置x轴标签，顺序由 order_window 决定
# plt.legend(title='Hemisphere', title_fontsize=24, fontsize=22, loc='upper right')
plt.grid(True, axis='y', alpha=0.3)

# 在柱子上方添加数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontsize=16, fontweight='bold')

sns.despine()
plt.tight_layout()
window_plot_path = os.path.join(output_dir, 'bilateral_temporal_window_quality.png')  # 文件名也更新
plt.savefig(window_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 颞窗质量对比图已保存: {window_plot_path}")

# -------------------------------
# 8. 图4: 回声强度等级柱状图（分左右侧）
# -------------------------------
plt.figure(figsize=(8, 6))
echo_df = pd.DataFrame({
    'Left': echo_left_counts,
    'Right': echo_right_counts
}, index=echo_levels)

ax = echo_df.plot(kind='bar', color=['#ff9999', '#66b3ff'], alpha=0.85,
                  edgecolor='black', width=0.8, figsize=(8, 6))
plt.xlabel('Echo Intensity Level', fontsize=12)
plt.ylabel('Number of Cases', fontsize=12)
plt.title('Bilateral Echo Intensity Distribution', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=0)
plt.legend(title='Hemisphere', title_fontsize=11, fontsize=10, loc='upper right')
plt.grid(True, axis='y', alpha=0.3)

# 在柱子上方添加数值
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=9, fontweight='bold')

sns.despine()
plt.tight_layout()
echo_plot_path = os.path.join(output_dir, 'echo_intensity_distribution.png')
plt.savefig(echo_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 回声强度图已保存: {echo_plot_path}")

# -------------------------------
# 9. 打印统计数据摘要
# -------------------------------
print("\n" + "="*40)
print("📊 数据集统计摘要")
print("="*40)
print(f"总样本量: {len(df)}")
male_count = gender_counts.get('Male', 0)
female_count = gender_counts.get('Female', 0)
print(f"男性: {male_count} ({male_count/len(df)*100:.1f}%)")
print(f"女性: {female_count} ({female_count/len(df)*100:.1f}%)")
print(f"年龄: {ages.mean():.1f} ± {ages.std():.1f} 岁, 范围 {ages.min()}–{ages.max()} 岁")