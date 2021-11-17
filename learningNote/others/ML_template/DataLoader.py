'''
数据加载器
'''

import numpy as np


class DataLoader:
    def __init__(self, filepaths):
        self.train_data = []
        self.train_labels = []

        self.test_data = []
        self.test_labels = []

        data, labels = self.readfiledata(filepaths)
        self.dataprocessing(data, labels)

    def readfiledata(self, filepaths):
        pass

    def dataprocessing(self, data, labels):
        self.datachange(data)
        self.labelschange(labels)
        pass

    def datachange(self):
        pass

    def labelschange(self):
        pass



