# Импорт необходимых библиотек
from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

# Загрузка данных и преобразование
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28*28)).astype('float32')/255
x_test = x_test.reshape((10000, 28*28)).astype('float32')/255

# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Модель
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)), # Скрытый слой
    layers.BatchNormalization(),   # Пакетная нормализация
    layers.Dense(256, activation='relu'), # Скрытый слой
    layers.BatchNormalization(),   # Пакетная нормализация
    layers.Dense(10, activation='softmax') # Выходной слой
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Построение графиков
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Сторонняя картинка
img = image.load_img("cifra.png", color_mode="grayscale", target_size=(28,28))
plt.imshow(img, cmap="gray")
plt.title("Загруженная картинка")
plt.show()

# Преобразование
img_array = image.img_to_array(img)   
img_array = img_array.reshape((1, 28*28)).astype('float32')/255

# Предсказание
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
print("Предсказанная цифра:", predicted_class)

