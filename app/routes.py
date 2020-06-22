import datetime
import os
import random
import requests

from PIL import Image as Im
from flask import render_template, request, make_response, redirect, url_for

from app import app, Image, search_image


# @app.route('/')
# def hello_world():
#     return 'Hello World!'

@app.route('/')
@app.route('/upload')
def home():
    siz = Image.query.count()
    list = []
    for i in range(13):
        list.append(random.randint(1, siz))
    return render_template("upload.html",list=list)

def create_uuid():  # 生成唯一的图片的名称字符串，防止图片显示时的重名问题
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S");  # 生成当前时间
    randomNum = random.randint(0, 100);  # 生成的随机整数n，其中0<=n<=100
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum);
    uniqueNum = str(nowTime) + str(randomNum);
    return uniqueNum;

basedir = os.path.abspath(os.path.dirname(__file__))
#图片上传
@app.route('/up_image', methods=['post'])
def up_photo():
    root=os.path.join(basedir,'temp')
    print(root)
    if(not os.path.exists(root)):
        os.makedirs(root)

    img = request.files.get('cover')
    name=img.filename
    print(name.split('.')[-1])
    temp=create_uuid()+'.' +name.split('.')[-1]
    file_path = root + '/'+ temp
    img.save(file_path)
    # return render_template('index.html')
    return '/search/'+temp;


#图片展示
@app.route('/search/<string:image>')
def search(image):
    root = os.path.join(basedir, 'temp')
    path=os.path.join(root,image)
    if IsValidImage(path):
        return render_template("image.html",my_list=search_image.search(path))
    else:
        return '请保证图片的完整性！'

#网络图片下载
@app.route('/download',methods=['POST'])
def download():
    root = os.path.join(basedir, 'temp')
    url=request.form.get('url')
    print(url)
    temp=create_uuid() + '.' +url.split('.')[-1]

    # 获取的文本实际上是图片的二进制文本
    img = requests.get(url).content
    # 将他拷贝到本地文件 w 写  b 二进制  wb代表写入二进制文本
    with open(os.path.join(root,temp), 'wb') as f:
        f.write(img)

    return '/search/'+temp

#验证图片的完整性
def IsValidImage(pathfile):
    bValid = True
    try:
        Im.open(pathfile).verify()
    except:
        bValid = False
    return bValid

# show photo
@app.route('/show/<string:file_dir>', methods=['GET'])
def show_photo(file_dir):
    image = Image.query.get(file_dir)
    if request.method == 'GET':
        print(image.url)
        if (image is not None) and IsValidImage(image.url):
            image_data = open(image.url, "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
        else:
            pass
    else:
        pass