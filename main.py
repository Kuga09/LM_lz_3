from programm import model_mnist, my_image

if __name__ == "__main__":
    # Обучаем модель
    model = model_mnist()

    # Проверяем на сторонней картинке
    my_image(model, "cifra.png")