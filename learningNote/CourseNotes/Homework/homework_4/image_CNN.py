"""
卷积神经网络预测图像性别
"""
import os

import tensorflow
import numpy as np
import datetime
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from data_loader import ImageLoader


class GenderPicCNN:
    def __init__(self, img_male, img_female, trainnum, testnum, labels, load_model=None):
        self.train_images = []
        self.test_images = []
        self.train_labels = []
        self.test_labels = []

        self.datasort(img_male, img_female, trainnum, testnum)
        self.data_standard(labels, trainnum, testnum)

        self.printout_input()

        input_shape = [100, 100, 3]

        if not load_model:
            model = self.build_model(input_shape, labels)

            self.model = self.train(model)
        else:
            self.model = keras.models.load_model(load_model)
            self.model.summary()

        predict_num = 333
        self.model_predict_singel_image(self.test_images[predict_num:predict_num + 1],
                                        self.test_labels[predict_num:predict_num + 1])
        self.model_predict(testnum)

    def datasort(self, img_male, img_female, trainnum, testnum):
        self.train_images = np.array(list(np.concatenate((img_male[0:trainnum[0], 0], img_female[0:trainnum[1], 0]))))
        self.train_labels = np.array(list(np.concatenate((img_male[0:trainnum[0], 2], img_female[0:trainnum[1], 2]))))
        self.test_images = np.array(list(np.concatenate((img_male[trainnum[0]:trainnum[0] + testnum[0], 0],
                                                         img_female[trainnum[1]:trainnum[1] + testnum[1], 0]))))
        self.test_labels = np.array(list(np.concatenate((img_male[trainnum[0]:trainnum[0] + testnum[0], 2],
                                                         img_female[trainnum[1]:trainnum[1] + testnum[1], 2]))))

    def data_standard(self, labels, trainnum, testnum):
        self.train_images = self.train_images / 255
        self.test_images = self.test_images / 255

        self.train_labels[0:trainnum[0]] = 1
        self.train_labels[trainnum[0]:] = 0
        self.test_labels[0:testnum[0]] = 1
        self.test_labels[testnum[0]:] = 0
        num_classes = labels
        self.train_labels = keras.utils.to_categorical(self.train_labels, num_classes)
        self.test_labels = keras.utils.to_categorical(self.test_labels, num_classes)

    def printout_input(self):
        print('\n-----------------------\ndata are loaded.')
        print('Number of training images: ', len(self.train_images))
        print('Number of testing images:  ', len(self.test_images))
        print('Shape of train_images: ', self.train_images.shape)
        print('Shape of test_images:  ', self.test_images.shape)

    def build_model(self, input_shape, numclasses):
        model = keras.Sequential(
            [
                keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2])),
                layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Flatten(),
                # layers.Dense(128, activation="softmax"),
                # layers.Dense(32, activation='softmax'),
                layers.Dense(numclasses, activation="softmax")
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
                verbose=1,
                save_best_only=True,
                mode='min'
        )

        history = model.fit(
                self.train_images,
                self.train_labels,
                batch_size=128,
                epochs=25,
                # 从测试集中划分多少到训练集
                # validation_split=0.1
                # 测试集
                validation_data=(self.test_images, self.test_labels),
                shuffle=True,
                callbacks=[checkpoint]
            )

        self.plot_metrics(history)

        score = model.evaluate(self.test_images, self.test_labels, verbose=2)

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
        return modelsave_path + 'image_cnn_model_callback_best' + nowTime + '.h5'

    def save_model(self, model):
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        modelsave_path = "./modelsave/"
        if not os.path.exists(modelsave_path):
            os.makedirs(modelsave_path)
        model.save(modelsave_path + 'image_cnn_model' + nowTime + '.h5')

    def model_predict(self, testnum):
        prediction_results = self.model.predict(self.test_images)
        prediction_index = np.argmax(prediction_results, axis=1)
        test_label_index = np.argmax(self.test_labels, axis=1)
        true_predict_num = (sum(prediction_index == test_label_index))
        true_predict_rate = true_predict_num / self.test_labels.shape[0]
        true_predict_male_num = (sum(prediction_index[0:testnum[0]] == test_label_index[0:testnum[0]]))
        true_predict_female_num = (sum(prediction_index[testnum[0]:] == test_label_index[testnum[0]:]))
        true_predict_male_rate = true_predict_male_num / (testnum[0])
        true_predict_female_rate = true_predict_female_num / testnum[1]

        print("---------------------------\n")
        print('true_predict_num: ', true_predict_num)
        print('true_predict_rate: ', true_predict_rate)
        print('true_predict_male_num: ', true_predict_male_num)
        print('true_predict_female_num: ', true_predict_female_num)
        print('true_predict_male_rate: ', true_predict_male_rate)
        print('true_predict_female_rate: ', true_predict_female_rate)

if __name__ == "__main__":
    imagefilename = "./lfw-deepfunneled/"
    genderfilename = "./lfw-deepfunneled-gender.txt"
    M = 4000
    F = 1200
    image = ImageLoader(imagefilename, genderfilename, [M, F])

    genderCNN = GenderPicCNN(image.img_array_standard[:4000], image.img_array_standard[4000:5200],
                             [3500, 1000], [500, 200], 2)
                             # './modelsave/image_cnn_model2021-10-30-19-48.h5')
    # genderCNN.model_predict()

