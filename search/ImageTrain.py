import os
from time import *

import numpy as np

from app import db, Image, app_milvus
from search.vgg import VGGNet
from milvus import Milvus, MetricType
from PIL import Image as Im


def create_milvus():
    milvus = app_milvus
    # # 删除
    if(milvus.has_collection('test01')[1]):
        status=milvus.drop_collection(collection_name='test01')
        print(status)
        sleep(5)#等待5s，等待删除完毕

    # 创建 collection 名为 test01， 1*512， 自动创建索引的数据文件大小为 1024 MB，距离度量方式为欧氏距离（L2）的 collection 。
    param = {'collection_name': 'test01', 'dimension': 512, 'index_file_size': 1024, 'metric_type': MetricType.L2}
    milvus.create_collection(param)
    return milvus

#验证图片的完整性
def IsValidImage(pathfile):
    bValid = True
    try:
        Im.open(pathfile).verify()
    except:
        bValid = False
    return bValid


def main():
    begin_time = time()
    vgg = VGGNet()
    milvus=create_milvus()
    db.session.query(Image).delete()#情况表
    db.session.commit()

    url=r'F:\训练图片'
    vectors,ids=[],[]
    cnt=1;
    for root,dirs,files in os.walk(url):

        for file in files:
            #获取文件路径
            path=os.path.join(root,file)
            if(IsValidImage(path)):
                print("----",path)
                vector=vgg.extract_feat(path)
                vectors.append(vector)
                image=Image(cnt,path)
                db.session.add(image)
                ids.append(cnt)
                cnt = cnt + 1
    milvus.insert(collection_name='test01', records=np.array(vectors),ids=ids)
    milvus.flush(collection_name_array=['test01'])

    db.session.commit()

    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time)  # 该循环程序运行时间： 1.4201874732

if __name__ == '__main__':
    main()