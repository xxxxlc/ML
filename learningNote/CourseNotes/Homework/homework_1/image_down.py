"""
从网页：http://vis-www.cs.umass.edu/lfw/alpha_last_G.html 下载以 “M” 开头的的头像图片。如 “Mitzi Gaynor”，
并将图片大小变成 100*100像素，然后旋转45度，批量保存为JPG格式图片
"""

# 导入函数库
import os

# 导入函数库
import re
from urllib.request import urlopen
from bs4 import BeautifulSoup
from PIL import Image


def image_downloader(url, size=None, angel=None):
    '''
    download picture
    url: str
    size: int
    angel: int
    :return: None
    '''

    # 确认图片保存的文件夹存在
    image_dir = "./image_download/"
    if not os.path.exists(image_dir):
        # 如果文件夹不存在，创建文件夹
        os.makedirs(image_dir)

    # 访问输入的网址
    webpage_url = url

    # 检查网页是否被成功打开
    webpage = None
    webpage_opened = True
    # 尝试打开网络，如果没有打开就会报错
    try:
        webpage = urlopen(webpage_url)
    except Exception as e:
        print("Failed to open the webpage")
        print("The error message: \n" + str(e))
        # 将 webpage_opened 置为 False
        webpage_opened = False

    # 网页被打开就会执行以下代码
    if webpage_opened:
        # 解析网页内容
        # 通过 BeautifulSoup 函数库将网页以 "html.parser" 的规则进行解析(.json)
        soup = BeautifulSoup(webpage, "html.parser")
        # 使用 BeautifulSoup 库中 findAll 函数对开头为 M 的图片进行正则搜索，观察图片命名格式为 \image\name\name_0001.jpg
        # 所以可以使用简单的正则表达式：".*\/M.*" 进行搜索(只要 \ 后跟 M 就行)
        image_tages = soup.findAll('img', {"alt": "person image", "src":re.compile('.*\/M.*')})

        # 下载 image_tages 中的图片
        # 记录图片的数量
        image_count = 0
        for img_Tag in image_tages:
            # 获取图片的链接地址
            image_link = webpage_url[0:webpage_url.rfind('/') + 1] + img_Tag['src']
            # 下载图片内容
            image = Image.open(urlopen(image_link))
            # 改变图片的像素为 100 * 100
            if size:
                image = image.resize((size[1], size[1]), Image.ANTIALIAS)
            # 旋转图片
            if angel:
                image = image.rotate(angel)
            # 保存图片
            image_file_name = image_dir + "image_%03d.png" % (image_count, )
            image.save(image_file_name)
            image_count += 1


if __name__ == "__main__":
    image_downloader('http://vis-www.cs.umass.edu/lfw/alpha_last_A.html', [100, 100], 45)





