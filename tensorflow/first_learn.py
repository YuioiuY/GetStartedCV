from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
import numpy as np

#region Constants
# размеры наших изображений.
img_width, img_height = 150, 150

train_data_dir = 'data/img_train'
validation_data_dir = 'data/img_val'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 10
batch_size = 16
#endregion

#region Model
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# это конфигурация дополнения, которую мы будем использовать для обучения
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# это конфигурация расширения, которую мы будем использовать для тестирования:
# только масштабирование
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#endregion

#region predict
# - > model.predict(validation_generator)

# shift+alt+a

""" datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = datagen.flow_from_directory(  
         'test',  
         target_size=(img_width, img_height),
         batch_size=batch_size,  
         class_mode=None,  
         shuffle=False)  

test_generator.reset()
   
pred= model.predict(test_generator, steps = 8//batch_size)
predicted_class_indices= np.argmax(pred, axis =-1 )
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(predicted_class_indices)
print (labels)
print (predictions)  """

#endregion

model.save_weights('first_try.weights.h5')