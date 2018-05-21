# Part 1 - Building the CNN
#
#  Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
from keras.models import load_model

# Initialising the CNN
from keras.utils import np_utils
import os

classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dropout(0.3))
classifier.add(Dense(units=256,activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units =1, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale=1/255)

training_set = train_datagen.flow_from_directory('C:\\Users\\thswl\\Desktop\\cat_dog\\cat_dog\\train',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:\\Users\\thswl\\Desktop\\cat_dog\\cat_dog\\test',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'binary')

'''
validation_set = validation_datagen.flow_from_directory('C:\\Users\\thswl\\Desktop\\cat_dog\\cat_dog\\validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
'''

classifier.fit_generator(training_set,
                         steps_per_epoch = 80, #10004 // 32,
                         epochs = 25,
                         #validation_data = validation_set,
                         #validation_steps = 2000
                          )

# 6. 모델 저장하기
classifier.save('cnn_dropout_80')

print("--Evaluate--")
scores = classifier.evaluate_generator(test_set, steps=5)
print(scores)
print("%s: %.2f%%" %(classifier.metrics_names[1], scores[1]*100))
print()

output = classifier.predict_generator(test_set, steps=5)
print('--predict--')
print(test_set.class_indices)
print(output)


