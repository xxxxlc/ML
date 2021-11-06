'''
提取数据，输出数据
'''

import os
import numpy as np
import xlsxwriter as xlsx
import xlrd
# 请使用 xlrd == 1.2.0 运行文件
# 高版本的 xlrd 会报错


def getfilename(files_dir):
    '''
    获取符合要求的文件名
    :param files_dir: 文件夹的相对路径 str
    :return: log_files: 符合要求的文件名 list
    '''
    # 获取文件夹下所有的符合条件的数据文件
    all_files = os.listdir(files_dir)
    # 创建列表来储存符合条件的文件名
    log_files = []
    # 循环所有的文件名
    for filename in all_files:
        # 对文件名进行匹配
        # if filename.startswith("runtime_data") and filename.endswith(".log"):
        log_files.append(filename)
    return log_files


def getdata(files_dir, log_files):
    '''

    :param files_dir: 文件夹名 str
    :param log_files: 需要读取文件名的列表 list
    :return: data: 所有文件的数据 list
             header: 文件的表头 list
    '''
    # 数据文件用 with 打不开
    # 因为excel中存在识别不了的字符
    # 读取指定文件夹的内容
    # 将所有文件的数据合并在一个文件中输出

    # 判断表头是否被读取过一次
    # fileheader = False
    # for filename in log_files:
        # 迭代所有的文件，打开指定文件夹内的文件
        # 判断该行是否是标题行
        # data = []
        # is_header = True
        # 使用 with 打开文件
        # with open(files_dir + "/" + filename, "r") as f:
        #     for line in f:
        #         if is_header:
        #             if not fileheader:
        #                 header = line
        #                 fileheader = True
        #             is_header = False
        #             continue
        #         else:
        #             line_data = line.split()
        #             data.append(line_data)

        # return header, data

    # 尝试使用 xlrd
    # header 用于记录表头
    header = None
    # data 用于记录数据
    data = []
    # 遍历所有的文件夹名
    for filename in log_files:
        # 使用 xlrd 打开一个 xlsx 文件
        excel = xlrd.open_workbook(files_dir + '/' + filename)
        # 读取 xlsx 文件中的 sheet1
        sheet = excel.sheet_by_name('sheet1')
        # 判断表头是否被读取
        if not header:
            # 读取表头
            header = sheet.row_values(0)
        # 读取除表头外所有数据
        for i in range(1, sheet.nrows):
            data.append(sheet.row_values(i))
    # 返回表头和数据
    return header, data


def outputfile(filename, header, data):
    '''

    :param filename: str 输出文件名
    :param header: list 表头
    :param data: list 数据
    :return: None
    '''
    # 创建输出对象
    workbook = xlsx.Workbook(filename)
    # 在输出的 xlsx 添加一个表
    worksheet = workbook.add_worksheet()

    # 迭代输入表头
    for col in range(0, len(header)):
        # 表头 col
        worksheet.write(0, col, header[col])

    # 迭代输入数据
    for i in range(1, len(data) + 1):
        for j in range(0, len(data[0])):
            worksheet.write(i, j, data[i - 1][j])
    # 关闭输出文件
    workbook.close()



if __name__ == '__main__':
    # 数据所在的文件夹路径
    files_dir = "./data_files"
    log_files = getfilename(files_dir)
    header, data = getdata(files_dir, log_files)
    outputfile('./' + 'result.xlsx', header, data)


