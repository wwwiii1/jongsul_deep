import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt


#데이터셋 설정
train_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.2,height_shift_range =0.2,
                                   zoom_range=0.2, horizontal_flip =True, vertical_flip = True)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\thswl\\Desktop\\cat_dog\\cat_dog\\train',
    target_size=(224, 224),
    batch_size=128,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.2,height_shift_range =0.2,
                                   zoom_range=0.2, horizontal_flip =True, vertical_flip = True)

test_generator = train_datagen.flow_from_directory(
    'C:\\Users\\thswl\\Desktop\\cat_dog\\cat_dog\\test',
    target_size=(224, 224),
    batch_size=128,
    class_mode='binary')

'''
val_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.2, height_shift_range = 0.2,
                                  zoom_range=0.2, horizontal_flip= True, vertical_flip=True)

val_generator = val_datagen.flow_from_directory(
    'C:\\Projects\\keras_talk\\test_set',
    target_size=(224, 224),
    batch_size=128,
    class_mode='binary')
'''
#알렉스넷 모델 생성
model = Sequential()

#Alexnet - 계층 1 : 11x11 필터를 96개를 사용, strides = 4, 활화화함수 = relu,
#                   입력 데이터 크기 224x224 , 3x3 크기의 풀리계층 사용

model.add(Conv2D(96, (11,11), strides=4, input_shape=(224,224,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))
model.add(BatchNormalization())

#Alexnet - 계층 2 : 5X5 필터를 256개 사용 , strides = 1, 활화화함수 = relu, 3x3 크기의 풀리계층 사용
model.add(ZeroPadding2D(2))
model.add(Conv2D(256,(5,5), strides=1, activation='relu'))

model.add(MaxPooling2D(pool_size=(3,3),strides=2))
model.add(BatchNormalization())

#Alexnet - 계층 3 : 3x3 필터를 384개 사용, strides =1 , 활성화함수 = relu
model.add(ZeroPadding2D(1))
model.add(Conv2D(384,(3,3), strides=1, activation='relu'))


#Alexnet - 계층 4 : 3x3 필터를 384개 사용, strides =1 , 활성화함수 = relu
model.add(ZeroPadding2D(1))
model.add(Conv2D(384,(3,3), strides=1, activation='relu'))


#Alexnet - 계층 5 : 3x3 필터를 256개 사용, strides =1 , 활성화함수 = relu, 3x3 크기의 풀리계층 사용
model.add(ZeroPadding2D(1))
model.add(Conv2D(256,(3,3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))

#계산을 위해서 1차원 배열로 전환
model.add(Flatten())

#Alexnet - 계층 6 : 4096개의 출력뉴런, 활성화함수 = relu
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

#Alexnet - 계층 7 : 4096게의 출력뉴런, 활성화함수 = relu
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

#Alexnet - 계층 8 : 1개의 출력뉴런, 활성화함수 = sigmoid
model.add(Dense(1, activation='sigmoid'))

#학습과정 설정 - 손실함수는 크로스엔트로피, 가중치 검색은 아담
sgd = SGD(lr=0.01,decay=5e-4, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])

model.summary()


#Alexnet - 학습하기
hist = model.fit_generator(train_generator, steps_per_epoch=312, epochs=20)#validation_data=val_generator, validation_steps=1)

#Alexnet - 그래프 그리기
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'],'y',label='train loss')
loss_ax.plot(hist.history['val_loss'],'r',label = 'val loss')

acc_ax.plot(hist.history['acc'],'b',label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
loss_ax.legend(loc='lower left')

plt.show()

#모델 저장하기
model.save('Alexnet2.h5')

#모델 평가하기
print("-------------Evaluate-----------------")
scores = model.evaluate_generator(test_generator,steps=1)
print("%s : %.2f%%" %(model.metrics_names[1],scores[1]*100))


#모델 사용하기
'''
함수를 이용하여 구축
if를 이용한 val_acc를 판단
'''
