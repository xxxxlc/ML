'''
肺炎图像处理
'''
import os
import numpy as np
from PIL import Image

class Imageloader:
    def __init__(self, filepaths):

        self.data = []
        self.label = []
        self.getdata(filepaths)

    def getdata(self, filepaths):
        for filepath in filepaths:
            # savepath = self.image_change(filepath)
            savepath = filepath.replace('data_files', 'data_files_modify')
            data, label = self.dataloader(savepath)
            self.data.append(data)
            self.label.append(label)

    def dataloader(self, filepath):
        listing_pic = os.listdir(filepath)
        imgdata = []
        imglabel = []
        for pic in listing_pic:
            if pic != "":
                img = Image.open(filepath + '/' + pic)
                img = np.array(img)
                imgdata.append(img)
                imglabel.append(self.b_or_v(pic))

        return imgdata, imglabel

    def image_change(self, filepath):
        listing_pic = os.listdir(filepath)
        for pic in listing_pic:
            if pic != "":
                img = Image.open(filepath + '/' + pic)
                img = img.convert("L")
                resizeimg = img.resize((300, 300))
                savepath = filepath.replace('data_files', 'data_files_modify')
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                resizeimg.save(savepath + '/' + pic)
        return savepath

    def b_or_v(self, name):
        if "virus" in name:
            return 1
        elif "bacteria" in name:
            return 1
        else:
            return 0


if __name__ == '__main__':
    trainpath_normal = 'data_files/train/NORMAL'
    trainpath_pneumonia = 'data_files/train/PNEUMONIA'
    val_normal = 'data_files/test/NORMAL'
    val_pneumonia = 'data_files/test/PNEUMONIA'
    filepaths = [trainpath_normal,
                 trainpath_pneumonia,
                 val_normal,
                 val_pneumonia]

    image = Imageloader(filepaths)
