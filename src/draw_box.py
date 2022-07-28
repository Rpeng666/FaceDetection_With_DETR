from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import torch
from pprint import pprint


def draw_box(img_tensor: torch.Tensor, probs: torch.Tensor, boxes: torch.Tensor, file_name: str):

    img_tensor = transforms.ToPILImage()(img_tensor)

    img_numpy = np.array(img_tensor, dtype=np.uint8)

    boxes = boxes.tolist()
    
    img = Image.fromarray(img_numpy)
    draw = ImageDraw.Draw(img)
    
    index = probs[:, 1] > 0.5

    pprint(probs.tolist())

    for i, item in enumerate(index):
        if not item:
            continue
        draw.rectangle(boxes[i], outline='blue', width= 3)

    img.save(file_name)