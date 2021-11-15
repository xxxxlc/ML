'''
卷积神经网络 CNN
'''
import os

import tensorflow
import numpy as np
import datetime
import matplotlib.pyplot as plt
import random

from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from data_loader_pne import Imageloader


class PneumoniaPicCNN:
    def __init__(self, image_data, image_label, load_model=None):
        self.train_images = image_data[0] + image_data[1]
        self.val_images = image_data[2] + image_data[3]
        self.train_labels = image_label[0] + image_label[1]
        self.val_labels = image_label[2] + image_label[3]

        self.train_num = len(self.train_images)
        self.val_num = len(self.val_images)
        self.num_classes = 2

        self.data_random()
        self.data_standard()
        self.printout_input()

        input_shape = [300, 300, 1]

        if not load_model:
            model = self.build_model(input_shape)

            self.model = self.train(model)
        else:
            self.model = keras.models.load_model(load_model)
            self.model.summary()

        # predict_num = 333
        # self.model_predict_singel_image(self.val_images[predict_num:predict_num + 1],
        #                                 self.val_labels[predict_num:predict_num + 1])
        self.model_predict(self.val_num)

    def data_random(self):
        # 数据随机化
        c = list(zip(self.train_images, self.train_labels))
        np.random.shuffle(c)
        self.train_images, self.train_labels = zip(*c)

        c = list(zip(self.val_images, self.val_labels))
        random.shuffle(c)
        self.val_images, self.val_labels = zip(*c)

    def data_standard(self):
        self.train_images = self.set_dimension(self.train_images, (self.train_num, 300, 300))
        self.val_images = self.set_dimension(self.val_images, (self.val_num, 300, 300))

        self.train_images = self.data_normalized(self.train_images)
        self.val_images = self.data_normalized(self.val_images)

        self.train_images = np.expand_dims(self.train_images, -1)
        self.val_images = np.expand_dims(self.val_images, -1)

        self.train_labels = self.convert_to_2(self.train_labels)
        self.val_labels = self.convert_to_2(self.val_labels)

    def set_dimension(self, lists, d):
        arr = np.array(lists)
        arr = np.reshape(arr, d)
        return arr

    def data_normalized(self, arr):
        arr = arr.astype('float32')
        return arr / 255.0

    def convert_to_2(self, lists):
        arr = np.array(lists)
        arr = keras.utils.to_categorical(arr, self.num_classes)
        return arr

    def printout_input(self):
        print('\n-----------------------\ndata are loaded.')
        print('Number of training images: ', len(self.train_images))
        print('Number of testing images:  ', len(self.val_images))
        print('Shape of train_images: ', self.train_images.shape)
        print('Shape of test_images:  ', self.val_images.shape)

    def build_model(self, input_shape):
        model = keras.Sequential(
            [
                keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2])),
                layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
                layers.MaxPool2D(pool_size=(2, 2)),
                # 防止过拟合
                layers.Dropout(0.25),
                layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Dropout(0.25),
                layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
                layers.Flatten(),
                layers.Dropout(0.25),
                layers.Dense(self.num_classes, activation="softmax")
            ]
        )

        model.summary()

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
            keras.metrics.BinaryAccuracy(name='accuracy'),
            # Computes the precision of the predictions with respect to the labels.
            keras.metrics.Precision(name='precision'),
            # Computes the recall of the predictions with respect to the labels.
            keras.metrics.Recall(name='recall'),
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
                self.train_images,
                self.train_labels,
                batch_size=128,
                epochs=20,
                # 从测试集中划分多少到训练集
                # validation_split=0.1
                # 测试集
                validation_data=(self.val_images, self.val_labels),
                shuffle=True,
                callbacks=[checkpoint]
            )

        self.plot_metrics(history)
        score = model.evaluate(self.val_images, self.val_labels, verbose=2)
        print('-----------------------\nEvaluating the trained model.')
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        self.save_model(model)

        return model

    def plot_metrics(self, history):
        metrics = ['loss', 'accuracy', 'precision', 'recall']
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

    def model_predict_singel_image(self, test_image, test_label):
        prediction_results = self.model.predict(test_image)
        predicted_index = np.argmax(prediction_results)
        test_label = np.argmax(test_label)
        labels = ['female', 'male']
        predicted_index = labels[predicted_index]
        test_label = labels[test_label]
        self.plot_prediction(prediction_results, predicted_index, test_label, test_image[0])

    def plot_prediction(self, prediction_array, predict_label, true_label, img):
        # 准备图
        plt.figure(figsize=(6, 3))
        # 画出图片
        plt.subplot(1, 2, 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)
        plt.subplot(1, 2, 2)
        plt.text(0.2, 0.7, "predict: " + str(predict_label), size=15)
        plt.text(0.2, 0.3, "true: " + str(true_label), size=15)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.show()

    def h5_file_name(self):
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        modelsave_path = "./modelsave/"
        return modelsave_path + 'pne_image_cnn_model_callback_best' + nowTime + '.h5'

    def save_model(self, model):
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        modelsave_path = "./modelsave/"
        if not os.path.exists(modelsave_path):
            os.makedirs(modelsave_path)
        model.save(modelsave_path + 'pne_image_cnn_model' + nowTime + '.h5')

    def model_predict(self, testnum):
        prediction_results = self.model.predict(self.val_images)
        prediction_index = np.argmax(prediction_results, axis=1)
        test_label_index = np.argmax(self.val_labels, axis=1)
        true_predict_num = (sum(prediction_index == test_label_index))
        true_predict_rate = true_predict_num / self.val_labels.shape[0]

        normal_num = 0
        true_predict_normol_num = 0
        true_predict_pneumonia_num = 0
        for i in range(0, len(prediction_index)):
            if test_label_index[i] == 0:
                normal_num += 1
                if prediction_index[i] == 0:
                    true_predict_normol_num += 1
            else:
                if prediction_index[i] == 1:
                    true_predict_pneumonia_num += 1

        true_predict_normol_rate = true_predict_normol_num / normal_num
        true_predict_pneumonia_rate = true_predict_pneumonia_num / (testnum - normal_num)

        print("---------------------------\n")
        print('true_predict_num: ', true_predict_num)
        print('true_predict_rate: ', true_predict_rate)
        print('true_predict_normol_num: ', true_predict_normol_num)
        print('true_predict_pneumonia_num: ', true_predict_pneumonia_num)
        print('true_predict_normol_rate: ', true_predict_normol_rate)
        print('true_predict_pneumonia_rate: ', true_predict_pneumonia_rate)

if __name__ == "__main__":
    trainpath_normal = 'data_files/train/NORMAL'
    trainpath_pneumonia = 'data_files/train/PNEUMONIA'
    val_normal = 'data_files/test/NORMAL'
    val_pneumonia = 'data_files/test/PNEUMONIA'
    filepaths = [trainpath_normal,
                 trainpath_pneumonia,
                 val_normal,
                 val_pneumonia]

    pne_image = Imageloader(filepaths)
    pne_CNN = PneumoniaPicCNN(pne_image.data, pne_image.label,
                              './modelsave/pne_image_cnn_model_callback_best2021-11-14-22-43.h5')