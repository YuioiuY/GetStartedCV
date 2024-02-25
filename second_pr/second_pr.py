#shift+alt+a

#region Полносвязная нейронная сеть для классификации изображений

""" import tensorflow as tf
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.datasets import mnist

# Загрузка и предобработка данных MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0  # Нормализация значений пикселей в диапазоне [0, 1]
test_images = test_images / 255.0

# Создание модели
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Преобразование 2D изображения в 1D
    Dense(128, activation='relu'),  # Полносвязный слой с 128 нейронами и функцией активации ReLU
    Dense(10, activation='softmax')  # Выходной слой с 10 нейронами (по числу классов) и функцией активации softmax
])

# Компиляция модели
model.compile(optimizer='adam',  # Оптимизатор Adam
              loss='sparse_categorical_crossentropy',  # Функция потерь для задачи классификации
              metrics=['accuracy'])  # Метрика качества - точность

# Обучение модели
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)) """

#endregion

#region Сверточная нейронная сеть для классификации изображений

""" import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist

# Загрузка и предобработка данных MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))  # Изменение формы для соответствия входной форме сети
train_images = train_images.astype('float32') / 255  # Нормализация значений пикселей в диапазоне [0, 1]
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# Создание модели
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Сверточный слой с 32 фильтрами и функцией активации ReLU
    MaxPooling2D((2, 2)),  # Слой максимального пулинга
    Flatten(),  # Преобразование 2D векторов в 1D
    Dense(128, activation='relu'),  # Полносвязный слой с 128 нейронами и функцией активации ReLU
    Dense(10, activation='softmax')  # Выходной слой с 10 нейронами (по числу классов) и функцией активации softmax
])

# Компиляция модели
model.compile(optimizer='adam',  # Оптимизатор Adam
              loss='sparse_categorical_crossentropy',  # Функция потерь для задачи классификации
              metrics=['accuracy'])  # Метрика качества - точность

# Обучение модели
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)) """

#endregion