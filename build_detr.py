import torch
import torch.nn as nn
from backbone import build_backbone
from transformer import Transformer
import torch.nn.functional as F


class Pos_embed(nn.Module):
    '''DETR中设置的位置编码'''
    def __init__(self, embed_dim, device) -> None:  # transformer中的embed_dim 的一半  (128)
        super(Pos_embed, self).__init__()
        self.device = device
        self.row_embed = nn.Embedding(num_embeddings= 15, embedding_dim= embed_dim)
        self.col_embed = nn.Embedding(num_embeddings= 15, embedding_dim= embed_dim)

    def forward(self, x):  # [batch_size, 256, 15, 15]
        h, w = x.shape[2], x.shape[3]
        i = torch.arange(w).to(self.device)
        j = torch.arange(h).to(self.device)

        x_embed = self.col_embed(i) #[15, 128]
        y_embed = self.row_embed(j) #[15, 128] 

        pos = torch.concat([
            x_embed.unsqueeze(0).repeat(h, 1, 1),
            y_embed.unsqueeze(1).repeat(1, w, 1)],
            axis = -1)  # [15, 15, 256]

        pos = pos.permute((2, 0, 1))  # [256, 15, 15]
        pos = pos.unsqueeze(0)  # [1, 256, 15, 15]
        pos = pos.repeat(x.shape[0], 1, 1, 1)  # [batch_size, 256, 15, 15]

        return pos


class BboxEmbed(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, int(in_dim / 2))
        self.fc2 = nn.Linear(int(in_dim / 2), int(in_dim / 2))
        self.fc3 = nn.Linear(int(in_dim / 2), out_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class DETR(nn.Module):
    def __init__(self,  num_classes, num_queries, device) -> None:
        super(DETR, self).__init__()

        self.transformer = Transformer()
        embed_dim = self.transformer.embed_dim  # 256

        if embed_dim % 2 != 0:
            raise 'embed_dim必须为偶数!'

        self.pos_embed = Pos_embed(int(embed_dim/2), device = device)

        self.backbone = build_backbone()
        self.conv1 = nn.Conv2d(in_channels= 512, out_channels= 256, kernel_size=1)

        self.query_embed = nn.Embedding(num_queries, embed_dim)

        self.classification = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed = BboxEmbed(embed_dim, 4)
 
        
    def forward(self, x):
        x = self.backbone(x)           #[batch_size, 512, 15, 15]
        x = self.conv1(x)              #[batch_size, 256, 15, 15]
        pos_embed = self.pos_embed(x)  #[batch_size, 256, 15, 15]
        
        out = self.transformer(x, pos_embed, self.query_embed.weight) # [100 (num_queries), batch_size, 512]

        out = out.permute(1, 0, 2) # [batch, num_queries, 256]

        output_class = F.softmax(self.classification(out), dim = -1 )# [ batch, 100, 2 (num_classes + 1)]
        bbox_pred = self.bbox_embed(out)  # [ batch, 100, 4 (center_x, center_y, w, h)]

        out = {
            'pred_logits': output_class,
            'pred_boxes': bbox_pred
        }

        return out



if __name__ == '__main__':
    # testing
    # num_class 为需要检测的目标的类别数量，并不包括背景
    # 默认第0个编号为背景类
    device = torch.device('cpu')

    x = torch.rand([4, 3, 450, 450])
    model = DETR(1, 20, device)

    out = model(x)
    print(out['pred_boxes'])