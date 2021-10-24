'''
加载图片
'''
import os
import numpy as np

from PIL import Image

class ImageLoader:
    def __init__(self, imagefilename, genderfilename, Num):
        self.imagefilename = imagefilename
        self.genderfilename = genderfilename
        self.malenum = Num[0]
        self.female = Num[1]

        self.img_name_gender = {}
        self.img_array = []
        self.img_array_male = []
        self.img_array_female = []
        self.img_array_standard = None

        self.genderloader()
        self.imgloader()
        self.select()

    def genderloader(self):
        f = open(self.genderfilename, "r")
        first_read = False

        for line in f.readlines():
            if not first_read:
                first_read = True
                continue
            [name, gender] = line.split('\t')
            self.img_name_gender[name] = gender[:-1]

    def imgloader(self):
        for name, gender in self.img_name_gender.items():
            img_name_path = self.imagefilename + name
            images_path = os.listdir(img_name_path)
            for imgpath in images_path:
                img = Image.open(img_name_path + '/' + imgpath)
                img = self.picprocessing(img, [25, 25, 225, 225], [100, 100])
                self.img_array.append([np.array(img), name, gender])
                if gender == 'male':
                    self.img_array_male.append(self.img_array[-1])
                else:
                    self.img_array_female.append(self.img_array[-1])

    def select(self):
        self.img_array_standard = np.array(self.img_array_male[:4000] + self.img_array_female[:1200])

    def check(self, n):
        image = self.img_array_standard[n][0]
        pli_image = Image.fromarray(image)
        pli_image.show()

    def picprocessing(self, img, box, resize):
        img = img.crop(box)
        img = img.resize((resize[0], resize[1]), Image.ANTIALIAS)
        return img


if __name__ == "__main__":
    imagefilename = "./lfw-deepfunneled/"
    genderfilename = "./lfw-deepfunneled-gender.txt"
    M = 4000
    F = 1200
    image = ImageLoader(imagefilename, genderfilename, [M, F])
    image.check(100)