import os
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

path_train = "data/training.csv"
path_test = "data/test.csv"
# 数据包含脸部关键点的位置，以及图像特征

def show(im):
    img = im.reshape(96, 96)
    plt.figure(figsize=(30, 30))
    plt.imshow(img)
    plt.show()
def load(test=False,cols = None):
    '''
    :param test: if test is True,load *test.csv*
    :param col: load subset from all column
    :return : 返回加载好的数据
    '''
    filename = path_test if test else path_train
    df = read_csv(filename)
    # Image 列的像素值由空格分割，将其转化为numpy数组
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    #show(df['Image'][1])
    if cols: df = df[list(cols)+['Image']]

    # print(df.count()) # 统计行数
    df = df.dropna() #将所有存在 NAN 数据的行清除
    # print(df.count())

    X = np.vstack(df['Image'].values)/255. # scale  pixel to [0,1]
    X = X.astype(np.float32)
    if not test: # 如果不是测试集，那么输出y就是除了Image 列之外的所有列
        y = df[df.columns[:-1]].values
        y = (y-48)/48 # scale y to [-1,1]
        y = y.astype(np.float32)
        X,y = shuffle(X,y,random_state = 42) #随机打乱 X,y
    else:
        y = None
    return X,y


if __name__ == "__main__":
    X,y = load()

    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05) #设置子区域的边框间隔
    for i in range(16):
        ax = fig.add_subplot(4,4,i+1,xticks=[],yticks=[])
        img = X[i].reshape(96,96)
        ax.imshow(img,cmap='gray')
        print(y[i][0::2])
        print(y[i][1::2])
        ax.scatter(y[i][0::2]*48+48,y[i][1::2]*48+48,s=10,c='r',marker='x')
    plt.show()
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape, y.min(), y.max()))

