import torch.nn as nn
import torch
from scipy.optimize import linear_sum_assignment
import torch


def box_cxcywh_to_xyxy(bbox):
    center_x = bbox[:, 0]
    center_y = bbox[:, 1]
    width = bbox[:, 2]
    height = bbox[:, 3]

    new_bbox = torch.zeros_like(bbox)
    new_bbox[:, 0] = center_x - width / 2
    new_bbox[:, 1] = center_y - height / 2
    new_bbox[:, 2] = center_x + width / 2
    new_bbox[:, 3] = center_y + height / 2

    return new_bbox

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) #[size1]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) #[size2]

    boxes1 = boxes1.unsqueeze(1) # size1 x 1 x 4
    lt = torch.max(boxes1[:, :, :2], boxes2[:, :2])  # [size1, size2, 2]
    rb = torch.min(boxes1[:, :, 2:], boxes2[:, 2:])  # [size1, size2, 2]

    wh = (rb - lt).clamp(min=0)  # [size1, size2, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [size1, size2]

    union = area1.unsqueeze(1) + area2 - inter # [size1, size2]

    iou = inter / union
    return iou, union


def generalized_box_iou(bbox1, bbox2):
    iou, union = box_iou(bbox1, bbox2)

    bbox1 = bbox1.unsqueeze(1) # [size1, 1, 4]

    lt = torch.min(bbox1[:, :, [0, 1]], bbox2[:, [0, 1]])  #[size1, size2, 2]
    rb = torch.max(bbox1[:, :, [2, 3]], bbox2[:, [2, 3]])   #[size1, size2, 2]

    wh = (rb - lt).clamp(min = 0)   #[size1, size2, 2]
    area = wh[:, :, 0] * wh[:, :, 1]  # [size1, size2]

    return iou - (area - union) / area  #[size1, size2]


class Loss_fn(nn.Module):
    def __init__(self, cost_weight_dict,  device) -> None:
        super(Loss_fn, self).__init__()
        self.device = device
        self.cost_weight_dict = cost_weight_dict
        self.weight = torch.tensor([0.25, 1], device= self.device)
        self.loss_ce_fn = nn.CrossEntropyLoss(weight= self.weight)
        self.loss_bbox_fn = nn.L1Loss()

    @torch.no_grad()
    def matcher(self, output_class, bbox_pred, target):
        # 计算三种损失：分类损失， 框的位置损失， 框的GIOU损失
        # 分类损失,由于只有一种类别，所以概率的负值就是相应的损失
        result = torch.zeros(output_class.shape[0]*output_class.shape[1], dtype=torch.long) # 用来存储这个batch最后匹配结果
        result.fill_(-1)

        tmp = 0

        for each in range(output_class.shape[0]): # 遍历batch中的每个图片
            each_output_class = output_class[each] # [num_queries, 2]
            each_bbox_pred = bbox_pred[each]    # [num_queries, 4]
            each_target = target[each]['boxes'].to(self.device)    # [?, 4]

            cost_class = - each_output_class[:, torch.ones(each_target.shape[0], dtype= torch.long)] # 分类损失
            cost_bbox = torch.cdist(each_bbox_pred, each_target, p = 1)      # 位置损失
            cost_giou = - generalized_box_iou(box_cxcywh_to_xyxy(each_bbox_pred), box_cxcywh_to_xyxy(each_target)) # GIOU损失

            cost_all = self.cost_weight_dict['loss_ce'] * cost_class + self.cost_weight_dict['loss_bbox'] * cost_bbox + self.cost_weight_dict['loss_giou']*cost_giou # 这里直接选择权重都是1

            index1, index2 = linear_sum_assignment(cost_all.cpu().detach().numpy())
            result[torch.LongTensor(index1 + each * output_class.shape[1])] = torch.LongTensor(index2 + tmp)
            tmp = tmp + each_target.shape[0]

        return result

    def forward(self, output_class, bbox_pred, target):
        # output_class: [ batch, num_queries, 2 (num_classes + 1)]
        # bbox_pred:    [ batch, num_queries, 4 (center_x, center_y, w, h)]
        # target:       list[ {'labels': Tensor([?]), 'boxes': Tensor[?, 4]}, ... ]

        match_result = self.matcher(output_class, bbox_pred, target)

        output_class = output_class.flatten(0, 1) # [batch * num_queries, 2]
        bbox_pred = bbox_pred.flatten(0, 1)       # [batch * num_queries, 4]
        target = torch.concat([ item['boxes'] for item in target ], dim = 0).to(self.device)  # [?, 4]

        # 分类损失
        new_label = torch.zeros(output_class.shape[0], dtype= torch.long, device= self.device)
        new_label[ match_result != -1 ] = 1
        loss_ce = self.loss_ce_fn(output_class, new_label)
        
        # 位置损失
        loss_bbox = self.loss_bbox_fn(target[ match_result[match_result != -1] ], bbox_pred[match_result != -1])

        # GIOU损失
        loss_giou = generalized_box_iou(
            box_cxcywh_to_xyxy(target[ match_result[match_result != -1] ]),
            box_cxcywh_to_xyxy(bbox_pred[match_result != -1])
            )

        loss_giou = (1- torch.diag(loss_giou)).mean()

        return loss_ce ,  loss_bbox , loss_giou