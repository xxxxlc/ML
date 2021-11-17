'''
卷积神经网络训练对象
'''
import os
import random
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tensorflow

from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from DataLoader import DataLoader


class CNNLoader:
    def __init__(self, image_data, image_labels, load_model):
        self.train_image = []
        self.train_labels = []

        self.test_image = []
        self.test_labels = []

        self.train_num = len(self.train_image)
        self.test_num = len(self.test_image)

        self.input_shape = []
        self.num_classes = 0

        if not load_model:
            model = self.build_model()
            self.model = self.train(model)
        else:
            self.model = keras.models.load_model(load_model)
            self.model.summary()

    def initialization(self):
        # 随机初始化，将数据的顺序打乱
        self.train_image, self.train_labels = self.data_random(self.train_image, self.train_labels)
        self.test_image, self.train_labels = self.data_random(self.test_image, self.test_labels)

        pass

    def data_random(self, data, labels):
        c = list(zip(data, labels))
        random.shuffle(c)
        data, labels = zip(*c)
        return data, labels

    def set_dimension(self, lists, d):
        arr = np.array(lists)
        arr = np.reshape(arr, d)
        return arr

    def data_normalized(self, arr):
        arr = arr.astype('float32')
        return arr / 255.0

    def train_length(self):
        return self.train_image.shape[0]

    def test_length(self):
        return self.test_image.shape[0]

    def convert_to_2(self, lists):
        arr = np.array(lists)
        arr = keras.utils.to_categorical(arr, self.num_classes)
        return arr

    def printout_input(self):
        print('\n-----------------------\ndata are loaded.')
        print('Number of training images: ', len(self.train_image))
        print('Number of testing images:  ', len(self.test_image))
        print('Shape of train_images: ', self.train_image.shape)
        print('Shape of test_images:  ', self.test_image.shape)

    def build_model(self):
        # 卷积神经网络的结构
        model = keras.Sequential(
            [
                keras.Input(shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            ]
        )

        model.summary()

        # 评价指标
        METRICS = [
            # Calculates the number of true positives
            # keras.metrics.TruePositives(name='tp'),
            # Calculates the number of false positives
            # keras.metrics.FalsePositives(name='fp'),
            # Calculates the number of true negatives.
            # keras.metrics.TrueNegatives(name='tn'),
            # Calculates the number of false negatives.
            # keras.metrics.FalseNegatives(name='fn'),
            # Calculates how often predictions match binary labels.
            keras.metrics.BinaryAccuracy(name='accuracy')
            # Computes the precision of the predictions with respect to the labels.
            # keras.metrics.Precision(name='precision'),
            # Computes the recall of the predictions with respect to the labels.
            # keras.metrics.Recall(name='recall'),
            # Approximates the AUC (Area under the curve) of the ROC or PR curves.
            # keras.metrics.AUC(name='auc')
        ]

        # adma = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epslion=1e-7, decay=0, amsgrad=False)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=METRICS
        )

        return model

    def train(self, model):
        checkpoint = ModelCheckpoint(
                filepath=self.h5_file_name(),
                monitor='val_loss',
                verbose=2,
                save_best_only=True,
                mode='min'
        )

        history = model.fit(
                self.train_image,
                self.train_labels,
                batch_size=128,
                epochs=20,
                # 从测试集中划分多少到训练集
                # validation_split=0.1
                # 测试集
                validation_data=(self.test_image, self.test_labels),
                shuffle=True,
                callbacks=[checkpoint]
            )

        self.plot_metrics(history)
        score = model.evaluate(self.test_image, self.test_labels, verbose=2)
        print('-----------------------\nEvaluating the trained model.')
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        return model

    def plot_metrics(self, history):
        metrics = ['loss', 'accuracy']
        for n, metric in enumerate(metrics):
            name = metric
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch, history.history[metric], label='Train')
            plt.plot(history.epoch, history.history['val_'+metric], linestyle='--', label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0, 1])
            else:
                plt.ylim([0, 1])
            plt.legend()

    def h5_file_name(self):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        modelsave_path = "./modelsave/"
        if not os.path.exists(modelsave_path):
            os.makedirs(modelsave_path)
        return modelsave_path + 'model_callback_best' + nowtime + '.h5'

    def model_predict(self, testnum):
        pass



if __name__ == '__main__':
    pass

