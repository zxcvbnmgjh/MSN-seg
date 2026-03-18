import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def click_event(event):
    if event.inaxes is None:
        return

    # 获取点击位置的像素坐标 (x, y)
    x, y = int(event.xdata), int(event.ydata)

    # 确保坐标在图像范围内
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        print(f"Clicked at: ({y}, {x})")  # 注意：matplotlib 的 y 是从上到下，但图像索引也是从上到下
        with open("clicked_coordinates_new_005.txt", "a") as f:
            f.write(f"{y},{x}\n")

    # 检查是否是右键点击
    if event.button == 3:  # 右键
        plt.close()
        print("Right-click detected. Exiting...")
        exit()

# ===========================
# 主程序
# ===========================
image_path = "/data2/gaojiahao/paper_prepare/qualitative_comparison_table/midbrian/data_new/005_input.png"  # 替换为你的图像路径

if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit()

# 读取图像
img = np.array(Image.open(image_path).convert("L"))  # 灰度图，也可以用 RGB

# 创建图像显示窗口
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img, cmap='gray')
ax.set_title("Click on the image to record coordinates\nLeft-click: save coordinate | Right-click: exit")
ax.grid(False)  # 不显示网格

# 连接事件
fig.canvas.mpl_connect('button_press_event', click_event)

# 显示图像
plt.show()