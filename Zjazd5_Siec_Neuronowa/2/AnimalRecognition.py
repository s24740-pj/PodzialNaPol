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
        W projekcie używamy dataset CIFAR10, który zawiera 60000 kolorowych zdjęć i 10 klas, każdy po 6000 obrazków:
            1. "Samolot", 
            2. "Samochód", 
            3. "Ptak", 
            4. "Kot", 
            5. "Jeleń",
            6. "Pies", 
            7. "Żaba", 
            8. "Koń", 
            9. "Statek", 
            10. "Ciężarówka".
        Uczymy model rozróżniać powyższe obiekty i zwierzęta, jednak skupiamy się na zwierzętach.
"""

def model_maker(x_train, x_test, y_train, y_test):
    """
        Description:
            Tworzymy model sieci neuronowej do przewidywania obiektu na zdjęciu 32x32.

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
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
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

        model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

        model.save("model.keras")

    return model

def load_image(plik):
    """
        Description:
            Załadujemy zdjęcie i przygotowujemy go do formatu przystosowanego do przewidywania, m.in rozmiar 32x32.

        Args:
            - plik: Lokalizacja do zdjęcie .png lub .jpg.

        Returns:
            - img_array: Zwraca przeformatowany ciąg liczb odwzorujący zdjęcie, przystosowany do modelu.
    """
    img = Image.open(plik).convert('RGB')
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape((1, 32, 32, 3))
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
            Zczytywanie dataset CIFAR10 i podzielenie na dane treningowe i testowe oraz dodajemy nazwy kolumn.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    class_names = ["Samolot", "Samochód", "Ptak", "Kot", "Jeleń", "Pies", "Żaba", "Koń", "Statek", "Ciężarówka"]

    """
        Description:
            Tworzenie modelu.
    """
    model = model_maker(x_train, y_train, x_test, y_test)

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
        plt.imshow(x_train[i])
        plt.xlabel(class_names[y_train[i][0]])
    # plt.savefig(f"WizualizacjaDanych.png", bbox_inches="tight")
    plt.show()

    """
        Description:
            Przetestowanie modelu na zdjęciach z danych testowych.
    """
    predictions = model.predict(x_test)
    for i in range(5):
        plt.figure(figsize=(5, 3))
        plt.imshow(x_test[i])
        plt.title(f"Prawdziwa: {class_names[y_test[i][0]]}, "
                  f"Przewidywana: {class_names[np.argmax(predictions[i])]}")
        plt.axis('off')
        # plt.savefig(f"{class_names[y_test[i][0]]}_{class_names[np.argmax(predictions[i])]}.png", bbox_inches="tight")
        plt.show()

    """
        Description:
            Załadujemy zdjęcie zewnętrzne i przewidujemy.
    """
    predict_image(model, "zaba.png", class_names)
    predict_image(model, "ptak.jpg", class_names)
    predict_image(model, "kon.jpg", class_names)
if __name__ == '__main__':
    main()