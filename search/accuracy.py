import os
import pickle
import random
import numpy as np
from search.vgg import VGGNet
import scipy.spatial.distance as dist

class Data:
    def __init__(self, type, vector, url):
        self.type=type
        self.vector = vector
        self.url = url
        self.dis=0

#计算欧氏距离
def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))

#计算曼哈顿距离
def manhattanDist(A,B):
    return np.sum(np.abs(A - B))

#计算切比雪夫距离
def chebyshevDist(A,B):
    return np.max(np.abs(A - B))

#计算余弦相似度
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

#计算杰卡德距离
def jaccardDist(x,y):
    up = np.double(np.bitwise_and((x != y), np.bitwise_or(x != 0, y != 0)).sum())
    down = np.double(np.bitwise_or(x != 0, y != 0).sum())
    d1 = (up / down)
    return d1

#计算汉明距离
def hammingDist(x,y):
    smstr = np.nonzero(x - y)
    # print(smstr)  # 不为0 的元素的下标
    sm = np.shape(smstr[0])[0]
    return sm

#获取猫和狗图片地址
def deal_data():
    # path=r'D:\百度网盘下载\kaggle\demo1'
    path=r'D:\百度网盘下载\kaggle\train'
    cats, dogs = [], []
    for i in os.listdir(path):
        image_path=os.path.join(path,i)
        if(i.find('cat')!=-1):
            cats.append(image_path)
        else:
            dogs.append(image_path)
    return cats,dogs

#提取猫和狗的特征值
def train(cats, dogs):
    vgg = VGGNet()
    list = []
    #训练猫数据
    for cat in cats:
        vector = vgg.extract_feat(cat)
        list.append(Data('cat',vector,cat))
        # file.write('cat '+)
    #训练狗数据
    for dog in dogs:
        vector = vgg.extract_feat(dog)
        list.append(Data('dog',vector,dog))
        # file.write()
    pickle.dump(list, open("data.txt", "wb"))
    return list

#按arr 匹配相似度
def math(arr,list_data):
    list=[]
    for data in list_data:
        data.dis=eucliDist(arr,data.vector)
        list.append(data)
    list.sort(key=lambda x:(x.dis))
    return list

#主函数
def main():
    # cats, dogs=deal_data()#获取数据
    # list=train(cats, dogs)#提取特征值
    list=pickle.load(open("data.txt", "rb"))
    for j in range(10):
        rand=random.randint(0,len(list)-1)
        rand_data=list[rand]
        top=math(rand_data.vector,list)#过去相识度最高的
        tot,right=100,0
        for i in range(tot):
            image=top[i]
            if(rand_data.url==image.url):
                image = top[tot]
            if(rand_data.type==image.type):
                right+=1
            # print(image.type,image.dis)
        print("accuracy:",right*100/tot,"%")

if __name__ == '__main__':
    main()