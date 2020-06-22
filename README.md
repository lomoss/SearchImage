# 项目介绍：

一个简单以图搜图的demo项目，采用vgg16提取特征值，使用Miluvs向量搜索引擎，只有简单几个文件，没有什么技术含量。

# 项目目录：

```bash
-SearchImage#项目
	-app#Flask目录
		-static#静态资源
		-temp#上传图片临时存放目录
		-templates#HTML
		__init__.py#初始化文件
		data.db#数据库
		routes.py#请求路由
		search_image.py#图片搜索
	-search
		-model#已经训练的模型
		accuracy.py#测试各相识度算法准确率，选用数据集(猫狗大战)
		ImageTrain.py#将图片库图片提取特征值放入milvus、sql
		vgg16.py#模型训练
	manage.py#Flask启动文件
```

# 数据集:

 ** **AwA2-data** (建议)** 、** pascal VOC 2007 2012** 

# 项目运行准备工作

环境：Flask、PIL、flask_sqlalchemy、pymilvus、numpy、keras、tensorflow

milvus安装： https://www.milvus.io/cn/docs/v0.10.0/guides/get_started/install_milvus/cpu_milvus_docker.md 

# 项目运行流程：

```
一、图片预处理：
	准备搜索的图片库
	提取图片库全部图片的特征值
	将特征值和图片地址持久化(存入数据库)
二、用户上传图片：
	将图片保存服务器
	提取该图片的特征值
	运用欧式距离算法与图片库特征值，进行相似度计算，展示相似度最高的前30张图片

```

前置条件环境配置完毕、milvus安装完毕且开启。

1.指定vgg.py模型地址

2.ImageTrain.py训练图片，提取特征

3.运行manage.py

![Image text](https://github.com/lomoss/SearchImage/blob/master/img-folder/0.png?raw=true)

# 项目展示：

![Image text](https://github.com/lomoss/SearchImage/blob/master/img-folder/1.png?raw=true)

![Image text](https://github.com/lomoss/SearchImage/blob/master/img-folder/2.png?raw=true)