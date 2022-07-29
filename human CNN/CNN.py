import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

from sys import path
path.append('..')
import numpy as np
import os
import cv2
from sklearn.datasets import load_breast_cancer # binaryclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

part_human = '/content/drive/MyDrive/AI/human_data_for_classification/Human_1'
part_non_human = '/content/drive/MyDrive/AI/human_data_for_classification/Non-Human_1'

ones_human = np.ones((28*28, 1)).reshape(1, -1)
for file in os.listdir(part_human):
  path_img_human = part_human+'/'+file
  image_human = cv2.imread(path_img_human, 0)
  image_human = cv2.resize(image_human, (28, 28)).reshape(1, -1)
  ones_human = np.vstack((ones_human, image_human))

data_human = np.delete(ones_human, 0, 0)
N, d = data_human.shape


ones_non_human = np.ones((28*28, 1)).reshape(1, -1)
for file in os.listdir(part_non_human):
  path_img_non_human = part_non_human+'/'+file
  image_non_human = cv2.imread(path_img_non_human, 0)

  image_non_human = cv2.resize(image_non_human, (28, 28)).reshape(1, -1)
  ones_non_human = np.vstack((ones_non_human, image_non_human))

data_non_human = np.delete(ones_non_human, 0, 0)

feature_set = np.vstack((data_human,data_non_human))
feature_set = feature_set/255

# label_ones = np.ones((2712, 1))
# label_zeros = np.zeros((2472, 1))

label_ones = np.ones((20, 1))
label_zeros = np.zeros((20, 1))
targets = np.vstack((label_ones, label_zeros))


X_train, X_test, Y_train, Y_test = train_test_split(feature_set, targets, test_size=0.2)
X_train, X_val, Y_train, Y_val = train_test_split(feature_set, targets, test_size=0.2)

# 3. Reshape lại dữ liệu cho đúng kích thước mà keras yêu cầu
X_train = X_train.reshape(X_train.shape[0], 28, 28, -1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# 4. One hot encoding label (Y)
Y_train = np_utils.to_categorical(Y_train, 10)
Y_val = np_utils.to_categorical(Y_val, 10)
Y_test = np_utils.to_categorical(Y_test, 10)
print('Dữ liệu y ban đầu ', Y_train[0])
print('Dữ liệu y sau one-hot encoding ',Y_train[0])

# 5. Định nghĩa model
model = Sequential()

# Thêm Convolutional layer với 32 kernel, kích thước kernel 3*3
# dùng hàm sigmoid làm activation và chỉ rõ input_shape cho layer đầu tiên
model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)))

# Thêm Convolutional layer
model.add(Conv2D(32, (3, 3), activation='sigmoid'))

# Thêm Max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer chuyển từ tensor sang vector
model.add(Flatten())

# Thêm Fully Connected layer với 128 nodes và dùng hàm sigmoid
model.add(Dense(128, activation='sigmoid'))

# Output layer với 10 node và dùng softmax function để chuyển sang xác xuất.
model.add(Dense(10, activation='softmax'))

# 6. Compile model, chỉ rõ hàm loss_function nào được sử dụng, phương thức
# đùng để tối ưu hàm loss function.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 7. Thực hiện train model với data
H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
          batch_size=32, epochs=10, verbose=1)

# 8. Vẽ đồ thị loss, accuracy của traning set và validation set
fig = plt.figure()
numOfEpoch = 10
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()

# 9. Đánh giá model với dữ liệu test set
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

# 10. Dự đoán ảnh
plt.imshow(X_test[2].reshape(28,28), cmap='gray')

y_predict = model.predict(X_test[2].reshape(1,28,28,1))
print('Giá trị dự đoán: ', np.argmax(y_predict))