# Face Detection With DETR

This is a small project using `DETR` to detect human face. Instead of using [official code](https://github.com/facebookresearch/detr) which is a little hard for me to understand , I simplified some structure (especially the loss function ) and hard code some hyper-parameter. 

## data 

You can download the dataset in [here](http://vis-www.cs.umass.edu/fddb/index.html). After done, unzip it to data folder. 

There should be two folders in `/data/`: `FDDB-folds` and `originalPics` .
