import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from nets.segformer import SegFormer


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "/data2/gaojiahao/Ultrasound_examination/Segmentation/segformer-plpa/save_weights/best_model.pth"   #最优权重地址
    # img_path = "./DRIVE/test/images/01_test.tif"
    img_path= "/data2/gaojiahao/Ultrasound_examination/Segmentation/segformer-plpa/midbrain_data/test/images/zyr8_image.png"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    # assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."


    mean=(0.12311059,0.12312306,0.12311852)
    std=(0.23592656,0.23593814,0.23593553)

    # get devices
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model   = SegFormer(num_classes=classes+1, phi='b3', pretrained=False)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

 

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        #prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("zyr8_test_result_B0.png")


if __name__ == '__main__':
    main()