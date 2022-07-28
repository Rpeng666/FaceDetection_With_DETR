import torch
from build_detr import DETR
from datasets import get_dataset
from src.draw_box import draw_box
import random


device = torch.device('cpu')
num_queries = 10

model = DETR(1, num_queries, device).to(device).eval()


model.load_state_dict(torch.load('model.pth'))

trainset, testset = get_dataset()


# for i in range(2000):
#     index = random.randint(0, 2000)
#     a = input('输入: ')
#     print(f'Index为: {index}')

#     img_tensor, label = trainset[index]    
    
#     with torch.no_grad():
#         outputs = model(img_tensor.unsqueeze(0))
    
#     print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)


#     outputs['pred_boxes'] = outputs['pred_boxes'] * 450

#     print(outputs)

#     center_x = outputs['pred_boxes'][:,:,0]
#     center_y = outputs['pred_boxes'][:,:,1]
#     width = outputs['pred_boxes'][:,:,2]
#     height = outputs['pred_boxes'][:,:,3]

#     outputs['pred_boxes'][:,:,0] = center_x - width / 2
#     outputs['pred_boxes'][:,:,1] = center_y - height / 2
#     outputs['pred_boxes'][:,:,2] = center_x + width / 2 
#     outputs['pred_boxes'][:,:,3] = center_y + height / 2

#     draw_box(img_tensor, outputs['pred_logits'].reshape(num_queries,2), outputs['pred_boxes'].reshape(num_queries,4), 'box.png')



for i in range(2000):
    index = random.randint(0, 2000)
    a = input('输入: ')
    print(f'Index为: {index}')

    img_tensor, label = trainset[index]    
    
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))
    
    boxes = outputs['pred_boxes'].reshape(num_queries,4)
    boxes = boxes * 450
    center_x, center_y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    boxes[:, 0] = center_x - w/2
    boxes[:, 1] = center_y - h/2
    boxes[:, 2] = center_x + w/2
    boxes[:, 3] = center_y + h/2


    draw_box(img_tensor, outputs['pred_logits'].reshape(num_queries,2), boxes, 'box.png')
