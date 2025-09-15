import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pywt
import cv2
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPool2D, Flatten, Dense, Dropout,
                          TimeDistributed, LSTM)
from keras.optimizers import adam_v2
# download dataset
x_train = pd.read_csv("./lab7/MI-EEG-B9T.csv",header=None)
x_test = pd.read_csv("./lab7/MI-EEG-B9E.csv",header=None)
y_train = pd.read_csv("./lab7/2class_MI_EEG_train_9.csv",header=None)
y_test = pd.read_csv("./lab7/2class_MI_EEG_test_9.csv",header=None)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
n_samples_train = len(y_train)
n_samples_test = len(y_test)

print("n_samples_train:", n_samples_train)
print("n_samples_test :", n_samples_test)

# count classes
n_classes = 2

# calculate scalogram CWT

def scalogram_vertical(data, fs, alto, ancho, n_canales, pts_sig):
    #data：包含时间信号的数据帧或数组。
    #fs：采样频率。
    #alto、ancho：最终图像的尺寸（高度和宽度）。
    #n_canales：通道数（如电极或传感器）。
    #ts_sig：每个通道的信号长度。

    dim = (int(np.floor(ancho / 2)), int(np.floor(alto / 2)))  # ancho, alto

    # Wavelet Morlet 3-3
    # frequency 8 - 30 Hz
    scales = pywt.scale2frequency('cmor3-3', np.arange(8, 30.5, 0.5)) / (1 / fs)
    #pywt.scale2frequency 将刻度转换为频率；除以 (1/fs) 则转换为 CWT 的刻度。
    # complex morlet wavelet
    datesets = np.zeros((data.shape[0], int(np.floor(alto / 2)),
                         int(np.floor(ancho / 2))))

    temporal = np.zeros((alto, ancho))
    #datesets：用于存储结果图像的数组。
    #temporal：用于存储当前信号的时间数组。

    for i in range(data.shape[0]):
        for j in range(n_canales):
            sig = data.iloc[i, j * pts_sig:(j + 1) * pts_sig]

            coef, freqs = pywt.cwt(sig, scales, 'cmor3-3',
                                   sampling_period=(1 / fs))

            temporal[j * 45:(j + 1) * 45, :] = abs(coef)

        resized = cv2.resize(temporal, dim, interpolation=cv2.INTER_AREA)
        datesets[i] = resized
        if i % 100 == 0:
            print(i)

    #提取每个信号的部分（sig）。
    #执行 CWT（pywt.cwt）--将信号转换为系数矩阵。
    #提取系数的绝对值（abs(coef) - 振幅频谱。
    #用通道数据填充时间矩阵。
    #最后将图像缩放至所需大小（cv2.resize）并保存。
    return datesets
'''
和CNN 关系
生成的图像（日期集）是 “scalograms ”或 “spectrograms”，可作为卷积神经网络的输入。
这种方法是使用小波变换 + CNN 的典型例子：
小波变换提取时频信息。
由此产生的图像被用作 CNN 的输入，CNN 经过训练可识别各种模式。
'''
x_train = scalogram_vertical(x_train, 250, 135, 1000, 3, 1000)
x_test = scalogram_vertical(x_test, 250, 135, 1000, 3, 1000)

print(x_train.shape)
print(x_test.shape)

x = np.ceil(np.max(x_train)) #x_train最大值四舍五入

# convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= x#归一化
x_test /= x #归一化 保持比例一致
#对训练样本和测试样本使用相同的比例：将两个样本除以相同的数字（x）可确保数据比例的一致性。
#print(x_train[1].shape)
plt.figure()

plt.imshow(x_train[50], aspect='auto')
plt.colorbar()
plt.show()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

print(x_train.shape[1:])
print(x_test.shape)


# 创建神经网络

def CNN_2D(num_filter, size_filter, n_neurons):
    model = Sequential()
    # Добавление сверточного слоя
    model.add(Conv2D(num_filter, kernel_size=size_filter, activation='relu', padding='same',
                     input_shape=x_train.shape[1:]))
    #num_filter - 用于搜索图像特征的过滤器（内核）数量。
    #kernel_size = size_filter - 每个过滤器的大小（例如，(3, 3)）。
    #activation = 'relu'
    #激活 - ReLU
    #激活函数，用于增加非线性。
    #padding = 'same' - 保留卷积后输入数据的维度。
    #input_shape = x_train.shape[1:] - 输入数据的形状，不含批次大小（例如（高度、宽度、通道））。
    # Добавление слоя подвыборки (макс-пулинг)
    model.add(MaxPool2D((2, 2)))
    #将图像的高度和宽度减半。有助于降低计算复杂度，提高抗位移能力。
    # Еще один сверточный слой
    model.add(Conv2D(num_filter, kernel_size=size_filter, activation='relu', padding='same'))
    # Еще слой подвыборки
    model.add(MaxPool2D((2, 2)))
    #第二个卷积层和池化
    #与第一层类似，还有一个卷积层和一个子采样层。

    #这可以让网络学习到更复杂的特征。
    # Преобразование 4D-данных в 1D-формат для полносвязных слоев

    model.add(Flatten())
    #将多维数据（如形状张量（高度、宽度、通道））转换为一维向量，以便输入全连接层。
    # Полносвязный слой с n_neurons нейронами
    model.add(Dense(n_neurons, activation='relu'))
    #处理获得的特征并学习如何将它们组合起来。

    #n_neurons - 该层的神经元数量。
    #activation = 'relu' - 激活 - 非线性。
    # Dropout для предотвращения переобучения
    model.add(Dropout(0.5))
    #在训练过程中随机关闭50 % 的神经元。防止模型训练过度。
    # Выходной слой с количеством нейронов равным числу классов
    model.add(Dense(n_classes, activation='softmax'))
    #神经元的数量等于类别数（n_classes）。使用软最大激活函数，将输出转化为每个类别的概率。
    # Компиляция модели с оптимизатором Adam и функцией потерь для многоклассовой классификации
    optimizer = adam_v2.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #优化器：亚当，学习步长较小。
    #损失函数：稀疏分类交叉熵，适用于整数标签。
    #指标：准确度 - -跟踪训练质量。
    return model


initial = time.time()
array_loss = []
array_acc = []
array_kappa = []

for i in range(5):
  print("Iteration:", i+1)
  model = CNN_2D(4, (3,3), 32)
  #history = model.fit(x_train, y_train, epochs=40, batch_size=36,
  #                   validation_data=(x_test, y_test), verbose=0)
  history = model.fit(x_train, y_train, epochs=70, batch_size=36,
                       validation_split = 0.1, verbose=0)

  test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

  array_loss.append(test_loss)
  print("loss: ", test_loss)
  array_acc.append(test_acc)
  print("accuracy: ", test_acc)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

probabilidades = model.predict(x_test)

y_pred = np.argmax(probabilidades, 1)

# calculate kappa cohen
kappa = cohen_kappa_score(y_test, y_pred)
array_kappa.append(kappa)
print("kappa: ", kappa)
matriz_confusion = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", matriz_confusion)

from sklearn.metrics import ConfusionMatrixDisplay

labels = ["Left MI", "Right MI"]

disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.show()

model.summary()

print("Mean Accuracy: %.4f" % np.mean(array_acc))
print("std: (+/- %.4f)" % np.std(array_acc))
print("Mean Kappa: %.4f" % np.mean(array_kappa))
print("std: (+/- %.4f)" % np.std(array_kappa))
print("Max Accuracy: %.4f" % np.max(array_acc))
print("Max Kappa: %.4f" % np.max(array_kappa))
fin = time.time()
time_elapsed = fin - initial
print("time_elapsed:", int(time_elapsed))
