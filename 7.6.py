import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pywt
import  cv2
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPool2D, Flatten, Dense, Dropout,
                          TimeDistributed, LSTM)
from keras.optimizers import Adam

# 加载数据集
x_train = pd.read_csv("MI-EEG-B9T.csv", header=None)
x_test = pd.read_csv("MI-EEG-B9E.csv", header=None)
y_train = pd.read_csv("2class_MI_EEG_train_9.csv", header=None)
y_test = pd.read_csv("2class_MI_EEG_test_9.csv", header=None)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

n_samples_train = len(y_train)
n_samples_test = len(y_test)

print("n_samples_train : ", n_samples_train)
print("n_samples_test: ", n_samples_test)

# 类别数量
n_classes = 2


# 计算scalogram CWT（连续小波变换）
def scalogram_vertical(data, fs, alto, ancho, n_canales, pts_sig):
    dim = (int(np.floor(ancho / 2)), int(np.floor(alto / 2)))  # 宽度, 高度
    # Morlet小波3-3
    # 频率8-30 Hz
    scales = pywt.scale2frequency('cmor3-3', np.arange(8, 30.5, 0.5)) / (1 / fs)
    # 复杂数morlet小波
    datesets = np.zeros((data.shape[0], int(np.floor(alto / 2)), int(np.floor(ancho / 2))))
    temporal = np.zeros((alto, ancho))
    for i in range(data.shape[0]):
        for j in range(n_canales):
            sig = data.iloc[i, j * pts_sig: (j + 1) * pts_sig]
            coef, freqs = pywt.cwt(sig, scales, 'cmor3-3', sampling_period=(1 / fs))
            temporal[j * 45: (j + 1) * 45, :] = abs(coef)
        resized = cv2.resize(temporal, dim, interpolation=cv2.INTER_AREA)
        datesets[i] = resized
        if i % 100 == 0:
            print(i)
    return datesets


x_train = scalogram_vertical(x_train, 250, 135, 1000, 3, 1000)
x_test = scalogram_vertical(x_test, 250, 135, 1000, 3, 1000)

print(x_train.shape)
print(x_test.shape)

x = np.ceil(np.max(x_train))
# 转换为float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= x
x_test /= x

# print(x_train[1].shape)
plt.figure()
plt.imshow(x_train[50], aspect='auto')
plt.colorbar()
plt.show()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

print(x_train.shape[1:])
print(x_test.shape)


def CNN_2D(num_filter, size_filter, n_neurons):
    model = Sequential()
    model.add(
        Conv2D(num_filter, kernel_size=size_filter, activation='relu', padding='same', input_shape=x_train.shape[1:]))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(num_filter, kernel_size=size_filter, activation='relu', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


array_loss = []
array_acc = []
array_kappa = []

for i in range(5):
    print("Iteration: ", i + 1)
    model = CNN_2D(4, (3, 3), 32)
    # history = model.fit(x_train, y_train, epochs=40, batch_size=36,
    #                     validation_data=(x_test, y_test), verbose=0)
    history = model.fit(x_train, y_train, epochs=70, batch_size=36,
                        validation_split=0.1, verbose=0)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    array_loss.append(test_loss)
    print("loss : ", test_loss)
    array_acc.append(test_acc)
    print("accuracy : ", test_acc)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    probabilidades = model.predict(x_test)
    y_pred = np.argmax(probabilidades, 1)
    # 计算kappa
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    array_kappa.append(cohen_kappa)
    print("kappa : ", cohen_kappa)
    matriz_confusion = confusion_matrix(y_test, y_pred)
    print("confusion matrix :\n", matriz_confusion)

    from sklearn.metrics import ConfusionMatrixDisplay

    labels = ["Left MI", "Right MI"]
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    model.summary()

print("Mean Accuracy : %.4f" % np.mean(array_acc))
print("std : (+/− %.4f)" % np.std(array_acc))
print("Mean Kappa : %.4f" % np.mean(array_kappa))
print("std : (+/− %.4f)" % np.std(array_kappa))
print("Max Accuracy : %.4f" % np.max(array_acc))
print("Max Kappa : %.4f" % np.max(array_kappa))
