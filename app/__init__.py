import os
from datetime import timedelta

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from milvus import Milvus

from search.vgg import VGGNet

app = Flask(__name__)

# 自动重载模板文件
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# 设置静态文件缓存过期时间
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.root_path, 'data.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # 关闭对模型修改的监控

db = SQLAlchemy(app)  # 初始化扩展，传入程序实例 app

class Image(db.Model):  # 表名将会是 image（自动生成，小写处理）
    id = db.Column(db.Integer, primary_key=True)  # 主键
    url = db.Column(db.String(50))  # url

    def __init__(self, id, url):
        self.id = id
        self.url = url

app_milvus = Milvus()
app_milvus.connect(host='localhost', port='19530')


# db.create_all()

from app import routes