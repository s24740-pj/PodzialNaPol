import numpy as np
import tensorflow as tf
import ssl
from PIL import Image
from matplotlib import pyplot as plt

"""
    Autorzy:
        Kamil Powierza
        Dawid Feister

    Wymagane biblioteki:
        1. numpy
        2. tensorflow
        3. ssl
        4. PIL
        5. matplotlib

    Description:
        W tym problemie używamy dataset [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), który zawiera `60000` czarno-białych zdjęć i `10` klas, każdy po `6000` obrazków o wielkości `28x28`:
            1. "T-shirt/top", 
            2. "Trouser", 
            3. "Pullover", 
            4. "Dress", 
            5. "Coat",
            6. "Sandal", 
            7. "Shirt", 
            8. "Sneaker", 
            9. "Bag", 
            10. "Ankle boot".
            
        Uczymy model rozróżniać powyższe `ubrania`.
        
        Nauczony model zostaje zapisany do pliku `model.keras` by uniknąć ponownego uczenia. 
        
"""


def model_maker(x_train, x_test, y_train, y_test):
    """
        Description:
            Tworzymy model sieci neuronowej do przewidywania obiektu na zdjęciu 28x28.

        Args:
            - x_train (ndarray): Zbiór treningowy. (Wielowymiarowa tablica)
            - x_test (ndarray): Zbiór testowy. (Wielowymiarowa tablica)
            - y_train (Series): Kolumna docelowa dla zbioru treningowego. (Jednowymiarowa tablica)
            - y_test (Series): Kolumna docelowa dla zbioru testowego. (Jednowymiarowa tablica)

        Returns:
            - model: Wytrenoweany model.
    """
    if tf.io.gfile.exists("model.keras"):
        model = tf.keras.models.load_model(
            "model.keras"
        )
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

        model.save("model.keras")

    return model


def load_image(plik):
    """
        Description:
            Załadujemy zdjęcie i przygotowujemy go do formatu przystosowanego do przewidywania, m.in rozmiar 28x28 oraz zmiana kolorów na odwrotne.

        Args:
            - plik: Lokalizacja do zdjęcie .png lub .jpg itp.

        Returns:
            - img_array: Zwraca przeformatowany ciąg liczb odwzorujący zdjęcie, przystosowany do modelu.
    """
    img = Image.open(plik).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape((1, 28, 28))
    """
        Description:
            Zamiana czarny z białym i biały z czarnym.
    """
    img_array = 1 - img_array
    return img_array


def predict_image(model, img_path, class_names):
    """
        Description:
            Przewidujemy co znajduje się na zdjęciu.

        Args:
            - model: Wytrenoweany model.
            - img_path: Ścieżka do zdjęcia.
            - class_names (array): Lista nazw klas.
    """
    new_image = load_image(img_path)
    pred = model.predict(new_image)
    predicted_class = class_names[np.argmax(pred)]
    print(f"Klasa do przewidywania: {img_path} Przewidziana klasa: {predicted_class}")


def main():
    """
        Description:
            Główna funckja.
    """

    """
        Description:
            SSL wyłącza weryfikację certyfikatów SSL w celu umożliwienia pobierania danych przez HTTPS z niezaufanych lub nieprawidłowo skonfigurowanych źródeł.
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    """
        Description:
            Zczytywanie dataset Fashion-MNIST i podzielenie na dane treningowe i testowe oraz dodajemy nazwy kolumn.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    """
        Description:
            Tworzenie modelu.
    """
    model = model_maker(x_train, x_test, y_train, y_test)

    """
        Description:
            Ocena modelu.
    """
    ev = model.evaluate(x_test, y_test)
    print(ev)

    """
        Description:
            Architektura modelu.
    """
    model.summary()

    """
        Description:
            Wizualizacja 10 przykładowych obrazków z bazy.
    """
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])
    # plt.savefig(f"WizualizacjaDanych.png", bbox_inches="tight")
    plt.show()

    """
        Description:
            Przetestowanie modelu na zdjęciach z danych testowych.
    """
    predictions = model.predict(x_test)
    for i in range(5):
        plt.figure(figsize=(5, 3))
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f"Prawdziwa: {class_names[y_test[i]]}, "
                  f"Przewidywana: {class_names[np.argmax(predictions[i])]}")
        plt.axis('off')
        # plt.savefig(f"{class_names[y_test[i]]}_{class_names[np.argmax(predictions[i])]}.png", bbox_inches="tight")
        plt.show()

    """
        Description:
            Załadujemy zdjęcie zewnętrzne i przewidujemy.
    """
    predict_image(model, "ankle_boot.jpg", class_names)
    predict_image(model, "tshirt.jpg", class_names)
    predict_image(model, "dress.jpg", class_names)


if __name__ == '__main__':
    main()