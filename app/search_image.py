import time

import numpy as np

from app import Image, app_milvus
from search.vgg import VGGNet

def search(path):
    vgg = VGGNet()
    milvus = app_milvus
    print(path)
    test_vectors = vgg.extract_feat(path)
    # 涉及的向量6个数
    search_param = {'nprobe': 500};
    status, tables = milvus.search(collection_name='test01', query_records=np.array([test_vectors]), top_k=30,params=search_param)
    # print(status)
    # print(type(tables[0][0].id),type(tables[0][0].distance))
    return tables[0]

if __name__ == '__main__':
    search()