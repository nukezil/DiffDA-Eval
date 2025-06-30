import os
import re
import numpy as np
import torch
from PIL import Image
import math

def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_image_grid(images, images_per_row=3, padding=10, bg_color=(255, 255, 255)):
    """
    将多个 PIL 图像以网格形式拼接成一张大图（保留原始尺寸）

    Args:
        images (List[Image.Image]): 输入图像列表（可异尺寸）
        images_per_row (int): 每行的图片数量
        padding (int): 图像之间的间隔像素
        bg_color (tuple): 背景颜色

    Returns:
        Image.Image: 拼接后的网格图像
    """
    if not images:
        raise ValueError("图像列表不能为空")

    # 预处理：计算行/列信息
    rows = math.ceil(len(images) / images_per_row)

    # 1. 获取每行的图像尺寸信息
    row_heights = []
    row_widths = []

    for row_idx in range(rows):
        row_images = images[row_idx * images_per_row : (row_idx + 1) * images_per_row]
        row_height = max(img.height for img in row_images)
        row_width = sum(img.width for img in row_images) + padding * (len(row_images) - 1)
        row_heights.append(row_height)
        row_widths.append(row_width)

    # 2. 计算总宽高
    grid_width = max(row_widths)
    grid_height = sum(row_heights) + padding * (rows - 1)

    # 3. 创建空白画布
    grid_image = Image.new("RGB", (grid_width, grid_height), color=bg_color)

    # 4. 粘贴每张图像
    y_offset = 0
    for row_idx in range(rows):
        x_offset = 0
        row_images = images[row_idx * images_per_row : (row_idx + 1) * images_per_row]
        max_height = row_heights[row_idx]

        for img in row_images:
            grid_image.paste(img, (x_offset, y_offset))
            x_offset += img.width + padding

        y_offset += max_height + padding

    return grid_image
