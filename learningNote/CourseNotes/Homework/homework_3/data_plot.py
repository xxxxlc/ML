'''
plot data
'''

import os
import io
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from PIL import Image

# 请使用 xlrd == 1.2.0 运行文件
# 高版本的 xlrd 会报错


def getdata_xlrd(filename):
    '''
    使用 xlrd 读取文件中的数据
    :param filename: str
    :return: header: list data: array
    '''
    # 创建列表存储excel中的文件数据
    data = []

    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf8 in position 14: invalid start byte
    # with open(filename, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         if not header:
    #             header.append(line)
    #             continue
    #         data.append(line)

    # 使用 xlrd 读取excel文件
    excel = xlrd.open_workbook(filename)
    # 读取excel文件中的 sheet1 中的所有数据
    sheet = excel.sheet_by_name('Sheet1')
    # 读取文件中的标题行
    header = sheet.row_values(0)
    # 读取文件中剩下的数据，存储在 data 中
    for i in range(1, sheet.nrows):
        data.append(sheet.row_values(i))
    # 将 data 转化为 array 数据类型，方便调用
    data = np.array(data)
    # 返回标题和数据
    return header, data

def getdata_pd(filename):
    '''
    使用 pandas 读取数据
    :param filename:
    :return: header, data
    '''
    # 使用 pandas 读取 excel 的数据
    df = pd.read_excel(filename, sheet_name='Sheet1')
    # 读取头部的标题
    header = df.columns
    # for i in range(0, len(header)):
    #     data.append(df[header[i]])
    # 读取剩下的数据
    data = df.iloc[1:]
    # 变成队列以便操作
    data = np.array(data)
    return header, data


def pltimage(header, data):
    '''
    画图
    :param header:
    :param data:
    :return: None
    '''
    # 计算数据的组数，也是画图的数量
    Nimage = len(header)
    # 使用 subplot 画出子图
    figure, axis = plt.subplots(Nimage // 2, 2, figsize=(10, 10))
    # 因为时间项未知，创建横坐标等于数据的长度
    x = range(0, len(data[:, 0]))
    # 调用 font_manager 函数，否则无法在图中显示汉字
    my_font = font_manager.FontProperties(fname="C:/WINDOWS/Fonts/STSONG.TTF")
    # 循环画图
    for i in range(Nimage):
        # 画图
        axis[i // 2][i % 2].plot(x, data[:, i], '-')
        # 图片标题
        axis[i // 2][i % 2].set_title(header[i], fontproperties=my_font)
        # 横坐标
        axis[i // 2][i % 2].set_xlabel('时间', fontproperties=my_font)
        # 纵坐标
        axis[i // 2][i % 2].set_ylabel('℃', fontproperties=my_font)
        # 开启网格
        axis[i // 2][i % 2].grid(True)
    # subplot 自带的图片整理函数
    figure.tight_layout()
    # 显示函数
    plt.show()


if __name__ == '__main__':
    filename = './data.xlsx'
    # header, data = getdata_pd(filename)
    header, data = getdata_xlrd(filename)
    pltimage(header[1:], data[:, 1:])

