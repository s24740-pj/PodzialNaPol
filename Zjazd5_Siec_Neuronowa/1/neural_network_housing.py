import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
"""
    Autorzy:
        Kamil Powierza
        Dawid Feister

    Wymagane biblioteki:
        1. pandas
        2. numpy
        3. sklearn.model_selection
        4. sklearn.preprocessing
        5. tensorflow
        6. matplotlib.pyplot

    Opis:
        Skrypt implementuje klasyfikację cen mieszkań w Bostonie na podstawie różnych cech, takich jak liczba pokoi, wskaźnik przestępczości itp.
        Dane są przetwarzane, normalizowane, a następnie klasyfikowane przy użyciu sieci neuronowej. Na końcu wyświetlana jest dokładność sieci
        neuronowej oraz jej porównanie z dokładnością drzewa decyzyjnego i modelu SVM.

    Instrukcja użycia:
        1. Upewnij się, że plik "housing.data.txt" jest w odpowiednim miejscu.
        2. Uruchom kod, aby załadować dane, nauczyć sieć neuronową klasyfikować dane i zobaczyć wyniki.
        3. Porównać wyniki modelu sieci neuronowej, drzewa decyzyjnego, SVM.
"""


def load_and_prepare_data(filepath):
    """
        Ładuje dane z pliku, przygotowuje je do klasyfikacji i dzieli na zbiory treningowy oraz testowy.

        Args:
            filepath (str): Ścieżka do pliku z danymi.

        Returns:
            tuple: Zbiory treningowe i testowe (X_train, X_test, y_train, y_test).
    """
    column_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "PRICE"
    ]
    data = pd.read_csv(filepath, delim_whitespace=True, header=None, names=column_names)

    threshold = data['PRICE'].median()
    data['Price_Category'] = (data['PRICE'] >= threshold).astype(int)

    X = data.drop(columns=['PRICE', 'Price_Category'])
    y = data['Price_Category']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    """
        Skaluje dane wejściowe do standaryzowanej postaci.

        Args:
            X_train (numpy.ndarray): Dane treningowe.
            X_test (numpy.ndarray): Dane testowe.

        Returns:
            tuple: Skalowane dane treningowe i testowe (X_train_scaled, X_test_scaled).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def build_model(input_shape):
    """
        Tworzy i kompiluje model sieci neuronowej do klasyfikacji.

        Args:
            input_shape (int): Liczba cech wejściowych.

        Returns:
            tf.keras.Model: Skompilowany model sieci neuronowej.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def plot_training_history(history):
    """
        Wizualizuje historię treningu modelu, pokazując stratę i dokładność na danych treningowych i walidacyjnych.

        Args:
            history (tf.keras.callbacks.History): Historia treningu modelu.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Strata - trening')
    plt.plot(history.history['val_loss'], label='Strata - walidacja')
    plt.title('Strata podczas treningu')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Dokładność - trening')
    plt.plot(history.history['val_accuracy'], label='Dokładność - walidacja')
    plt.title('Dokładność podczas treningu')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()

    plt.tight_layout()
    plt.show()


def classify_samples(model, X_test):
    """
        Klasyfikuje przykładowe dane testowe i wyświetla wyniki.

        Args:
            model (tf.keras.Model): Wytrenowany model.
            X_test (numpy.ndarray): Dane testowe.
    """
    sample_data = X_test[:3]
    sample_predictions = model.predict(sample_data)
    sample_predictions = (sample_predictions > 0.5).astype(int)

    print("Przykładowe dane testowe i ich klasyfikacja:")
    for i, (input_data, prediction) in enumerate(zip(sample_data, sample_predictions)):
        print(f"Dane: {input_data},    Klasyfikacja ceny: {'Wysoka' if prediction[0] == 1 else 'Niska'}")


def main():
    """
        Główna funkcja programu: ładuje dane, trenuje model, ocenia go oraz porównuje z innymi metodami klasyfikacji.

        Sekcje funkcji:
            1. Ładowanie danych: Przygotowanie zbioru danych do modelowania.
            2. Skalowanie danych: Normalizacja cech wejściowych.
            3. Budowa i trening modelu: Tworzenie i trenowanie sieci neuronowej.
            4. Ocena modelu: Obliczanie strat i dokładności na zbiorze testowym.
            5. Porównanie metod: Analiza wyników sieci neuronowej względem drzewa decyzyjnego i SVM.
            6. Wizualizacja: Rysowanie wykresu historii treningu.
            7. Klasyfikacja próbek: Testowanie modelu na przykładach.
    """
    # Sekcja 1: Ładowanie danych
    filepath = "housing.data.txt"
    X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)

    # Sekcja 2: Skalowanie danych
    X_train, X_test = scale_data(X_train, X_test)

    # Sekcja 3: Budowa i trening modelu
    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

    # Sekcja 4: Ocena modelu
    loss, accuracy = model.evaluate(X_test, y_test)
    print("=== Wyniki sieci neuronowej ===")
    print(f"Dokładność na zbiorze testowym: {accuracy:.2f}")
    print(f"Strata na zbiorze testowym: {loss:.2f}")

    # Sekcja 5: Porównanie metod
    decision_tree_accuracy = 0.81
    decision_SVM_accuracy = 0.84
    print(f"Dokładność drzewa decyzyjnego: {decision_tree_accuracy:.2f}")
    print(f"Dokładność modelu SVM: {decision_SVM_accuracy:.2f}")

    if accuracy > decision_tree_accuracy and accuracy > decision_SVM_accuracy:
        print(f"Najdokładniejsza jest sieć neuronowa: {accuracy:.2f}")
    elif decision_tree_accuracy > accuracy and decision_tree_accuracy > decision_SVM_accuracy:
        print(f"Najdokładniejsze jest drzewo decyzyjne: {decision_tree_accuracy:.2f}")
    else:
        print(f"Najdokładniejszy jest model SVM: {decision_SVM_accuracy:.2f}")

    # Sekcja 6: Wizualizacja
    plot_training_history(history)

    # Sekcja 7: Klasyfikacja próbek
    classify_samples(model, X_test)


if __name__ == "__main__":
    """
        Uruchamia główną funkcję programu.
    """
    main()
