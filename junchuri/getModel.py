from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:\\Users\\thswl\\Desktop\\cat_dog\\cat_dog\\test',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'binary')

model = load_model('cnn_three_layer1')

print("--Evaluate--")
scores = model.evaluate_generator(test_set, steps=5)
print(scores)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
print()

output = model.predict_generator(test_set, steps=5)
print('--predict--')
print(test_set.class_indices)
print(output)
#print(np.where(output>=0, 1, 0))


