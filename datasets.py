from sklearn import datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import random
import copy
from pprint import pprint


def resize_img(image: Image, size: tuple):
    # 对图片进行resize，使图片不失真。在空缺的地方进行padding
    old_w, old_h = image.size
    new_w, new_h = size
    scale = min(new_w/old_w, new_h/old_h)
    new_w = int(old_w*scale)
    new_h = int(old_h*scale)

    image = image.resize((new_w, new_h), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    # new_image.paste(image, ((w-new_w)//2, (h-new_h)//2))
    new_image.paste(image, (0, 0))
    return new_image


def detection_collate(batch: list):
    imgs = []  # 里面的每个元素是图像的Tensor
    targets = [] # 里面的每个元素是个dict

    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])

    return torch.stack(imgs), targets


def my_transform(img: Image, boxes: list, size: tuple):
    '''数据集增强
    img:输入的原始图片, boxes 标注框[x1, y1, x2, y2]
    size:最后统一的大小'''
    old_w, old_h = img.size
    boxes = np.array(boxes)

    # 随机水平翻转图片
    if(random.random() > 0.5):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        tmp = copy.deepcopy(boxes[:, 0])  # x1
        boxes[:, 0] = old_w - boxes[:, 2] # x1改变了
        boxes[:, 2] = old_w - tmp # 原来的x1
    
    # 随机垂直翻转
    if(random.random() > 0.5):
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        tmp = copy.deepcopy(boxes[:, 1]) # y1
        boxes[:, 1] = old_h - boxes[:, 3] # y1改变了
        boxes[:, 3] = old_h - tmp  # 原来的y1

    new_h, new_w = size
    max_scale_ratio = min(new_h/ old_h, new_w/ old_w)

    # 随机缩放/扩大图片
    rand_scale_ratio = random.uniform(0.25, max_scale_ratio)
    if(random.random() > 0.5):
        img = img.resize((int(rand_scale_ratio * old_w), int(rand_scale_ratio * old_h)), Image.BICUBIC)
        boxes *=  rand_scale_ratio


    # 随机图片位置（原始图片的左上角坐标)
    new_x = random.randint(0, new_w - img.size[0])
    new_y = random.randint(0, new_h - img.size[1])

      # 标注框也需要相应的改变位置
    boxes[:, 0] += new_x
    boxes[:, 2] += new_x

    boxes[:, 1] += new_y
    boxes[:, 3] += new_y

    # 统一图片大小
    new_img = Image.new('RGB', size, (128,128,128))
    new_img.paste(img, (new_x, new_y))
    img = new_img

    return img, boxes


def change_xy(boxes:list, size:list):
    '''将[x1, y1, x2, y2]的格式改为 [center_x, center_y, width, height], 
    同时归一化到[0,1]之间'''
    for box in boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        box[0] = ((x1 + x2)/2)/size[0]
        box[1] = ((y1 + y2)/2)/size[1]
        box[2] = ((x2 - x1))/size[0]
        box[3] = ((y2 - y1))/size[1]

    return boxes


class MyDataset(Dataset):
    def __init__(self, data, transform: transforms) -> None:
        super().__init__()

        self.data = [] 
        self.label = []  # list of dict

        self.transform = transform
        
        for img_path, label in data:
            img = Image.open(img_path)
            self.data.append(img.copy())
            self.label.append(label) 


    def __getitem__(self, index):
        data, label = my_transform(self.data[index], self.label[index], (450, 450)) # 随机变换
        label = change_xy(label, data.size)

        data = self.transform(data)

        single_label = dict()
        single_label['labels'] = torch.tensor([1 for i in range(len(label))], dtype=torch.long) # 多少个框就多少个类别标签
        single_label['boxes'] = torch.Tensor(label)  # [框的个数, 4]

        return data, single_label

    def __len__(self):
        return len(self.label)


def get_ellipse_param(major_radius, minor_radius, angle):
    a, b = major_radius, minor_radius
    sin_theta = np.sin(-angle)
    cos_theta = np.cos(-angle)
    A = a**2 * sin_theta**2 + b**2 * cos_theta**2
    B = 2 * (a**2 - b**2) * sin_theta * cos_theta
    C = a**2 * cos_theta**2 + b**2 * sin_theta**2
    F = -a**2 * b**2
    return A, B, C, F


def calculate_rectangle(A, B, C, F):
    '''
    椭圆上下外接点的纵坐标值
    '''
    y = np.sqrt(4*A*F / (B**2 - 4*A*C))
    y1, y2 = -np.abs(y), np.abs(y)
    
    '''
    椭圆左右外接点的横坐标值
    '''
    x = np.sqrt(4*C*F / (B**2 - 4*C*A))
    x1, x2 = -np.abs(x), np.abs(x)
    
    return (x1, y1), (x2, y2)


def get_rectangle(major_radius, minor_radius, angle, center_x, center_y):
    A, B, C, F = get_ellipse_param(major_radius, minor_radius, angle)
    p1, p2 = calculate_rectangle(A, B, C, F)
    return (center_x+p1[0], center_y+p1[1]), (center_x+p2[0], center_y+p2[1])


def get_img_and_label():
    imgs_path_label = []
    base_label_path = './data/FDDB-folds/'
    base_img_path = './data/originalPics/'

    for dir in os.listdir(base_label_path):

        if(len(dir) <= 16):
            continue

        file_path = base_label_path + dir

        with open(file_path, 'r', encoding= 'utf-8') as file:
            while(1):
                img_path = file.readline().strip()

                if(img_path == ''):
                    break

                face_cnt = int(file.readline().strip())
                label = []

                for face in range(face_cnt):

                    major_axis_radius, minor_axis_radius,\
                         angle, center_x, center_y = tuple(map(float, file.readline().strip().split(' ')[:-2]))

                    (x1, y1), (x2, y2) = get_rectangle(major_axis_radius, minor_axis_radius, angle, center_x, center_y)

                    label.append([x1, y1, x2, y2])
                
                imgs_path_label.append([base_img_path + img_path +'.jpg', label])
        
    return imgs_path_label

    
def get_dataloader(batch_size):

    imgs_path_label = get_img_and_label()
    train_data, test_data = train_test_split(imgs_path_label, test_size= 0.8, shuffle = True)

    transform = transforms.ToTensor()

    trainset = MyDataset(train_data, transform)
    testset = MyDataset(test_data, transform)

    trainloader = DataLoader(trainset, batch_size= batch_size, shuffle= True, collate_fn = detection_collate)
    testloader  = DataLoader(testset,  batch_size= batch_size, shuffle= True, collate_fn = detection_collate)

    return trainloader, testloader
    

def get_dataset():
    imgs_path_label = get_img_and_label()
    train_data, test_data = train_test_split(imgs_path_label, test_size= 0.2)
    

    transform = transforms.ToTensor()

    trainset = MyDataset(train_data, transform)
    testset = MyDataset(test_data, transform)

    return trainset, testset



if __name__ == '__main__':
    # 测试部分
    trainloader , testloader = get_dataloader(30)
    for i, (data, target) in enumerate(trainloader):
        for j in range(len(target)):
            print(target[j]['boxes'])
    # imgs_path_label = get_img_and_label()
    # train_data, test_data = train_test_split(imgs_path_label, test_size= 0.2)

    # transform = transforms.ToTensor()

    # trainset = MyDataset(train_data, transform)

    # while(1):

    #     # index = int(input('输入：'))

    #     # img_tensor, boxes = trainset[index]
    #     # boxes = boxes['boxes']
    
    #     # img = transforms.ToPILImage()(img_tensor)

    #     # draw = ImageDraw.Draw(img)
        
    #     # for box in boxes:
    #     #     box = box * 450
    #     #     print(box.tolist())
    #     #     centerx, centery, width, height = box.tolist()
    #     #     draw.rectangle([centerx - width/2, centery - height/2, centerx + width/2, centery + height/2], outline='blue')

    #     # img.save('box.png')
      