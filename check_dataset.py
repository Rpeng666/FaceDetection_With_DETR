from PIL import Image
import os

path = './data/originalPics/'

cnt = 0
h_list = []
w_list = []

for dir in os.listdir(path):
    for month in os.listdir(path + dir):
        for day in os.listdir(path + dir +'/'+ month):
            for img_name in os.listdir(path + dir + '/' + month + '/' +day+ '/big'):
                img = Image.open(path + dir + '/' + month + '/' +day+ '/big/' + img_name )
                h, w = img.size

                h_list.append(h)
                w_list.append(w)

print(max(h_list), max(w_list))