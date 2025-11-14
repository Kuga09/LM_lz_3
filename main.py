# Импорт функций
from programm import model_mnist, my_image

# Функия main
def main():
    model = model_mnist()

    my_image(model, "cifra.png")

if __name__ == "__main__":
    main()