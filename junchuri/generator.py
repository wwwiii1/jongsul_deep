import numpy as np

# 랜덤시드 고정시키기
np.random.seed(5)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'C:\\Users\\thswl\\PycharmProjects\\junchuri\\train',
        target_size=(24, 24),
        batch_size=3,
        class_mode='categorical')



# 데이터셋 불러오기
data_aug_gen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=90,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.5,
                                  zoom_range=[0.5, 0.5],
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')


img = load_img('C:\\Users\\thswl\\PycharmProjects\\junchuri\\test\\atopi\\2.jpg')

x = img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0

# 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.

for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir='C:\\Users\\thswl\\PycharmProjects\\junchuri\\generateData', save_prefix='tri', save_format='jpg'):
    i += 1
    if i > 2:
        break

