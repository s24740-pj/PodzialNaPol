import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
        7. seaborn
        8. sklearn.metrics

    Opis:
        Skrypt implementuje klasyfikację cen mieszkań w Bostonie na podstawie różnych cech, takich jak liczba pokoi, wskaźnik przestępczości, itp.
        Zawiera proces ładowania, przetwarzania, standaryzowania danych, oraz budowy i treningu dwóch modeli sieci neuronowych: małej i dużej sieci. 
        Modele są oceniane na podstawie dokładności i straty na zbiorze testowym. 
        Wyniki są porównywane z dokładnością modeli drzewa decyzyjnego oraz SVM.
        Dodatkowo wyświetlane są:
        - Macierze pomyłek dla obu modeli sieci neuronowych,
        - Wizualizacje procesu treningu (strata i dokładność),
        - Klasyfikacja próbek danych testowych przez oba modele.
"""


def load_and_prepare_data(filepath):
    """
        Ładuje dane z pliku, przygotowuje je do klasyfikacji i dzieli na zbiory treningowy oraz testowy.

        Parametry:
            filepath (str): Ścieżka do pliku z danymi wejściowymi.

        Zwraca:
            tuple: Zbiór danych treningowych (X_train, y_train) i testowych (X_test, y_test).
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

        Parametry:
            X_train (DataFrame): Dane treningowe.
            X_test (DataFrame): Dane testowe.

        Zwraca:
            tuple: Zbiór danych treningowych i testowych po skalowaniu (X_train_scaled, X_test_scaled).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def build_model(input_shape, layers=[32, 16, 8]):
    """
        Tworzy i kompiluje model sieci neuronowej do klasyfikacji.

        Parametry:
            input_shape (int): Liczba cech wejściowych (rozmiar wejścia).
            layers (list): Lista liczby neuronów w poszczególnych warstwach ukrytych.

        Zwraca:
            model (tf.keras.Sequential): Skonstruowany model sieci neuronowej.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_shape,)))

    for units in layers:
        model.add(tf.keras.layers.Dense(units, activation='relu'))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def plot_training_history(history):
    """
        Wizualizuje historię treningu modelu, pokazując stratę i dokładność na danych treningowych i walidacyjnych.

        Parametry:
            history (History): Historia treningu modelu (obiekt zwrócony przez fit).

        Zwraca:
            None
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


def plot_confusion_matrix(y_true, y_pred):
    """
        Rysuje macierz pomyłek na podstawie rzeczywistych i przewidywanych etykiet.

        Parametry:
            y_true (array-like): Rzeczywiste etykiety (wartości docelowe).
            y_pred (array-like): Przewidywane etykiety.

        Zwraca:
            None
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Niska", "Wysoka"], yticklabels=["Niska", "Wysoka"])
    plt.xlabel('Przewidywana')
    plt.ylabel('Rzeczywista')
    plt.title('Macierz pomyłek')
    plt.show()


def classify_samples(model, X_test):
    """
        Klasyfikuje przykładowe dane testowe i wyświetla wyniki.

        Parametry:
            model (tf.keras.Model): Wytrenowany model sieci neuronowej.
            X_test (DataFrame): Zbiór danych testowych do klasyfikacji.

        Zwraca:
            None
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
        Sekcje w funkcji:
        1. Ładowanie danych
        2. Skalowanie danych
        3. Budowa i trening modelu (mała sieć neuronowa)
        4. Budowa i trening modelu (duża sieć neuronowa)
        5. Ocena modeli
        6. Porównanie wyników z innymi metodami klasyfikacji (drzewo decyzyjne, SVM)
        7. Wizualizacja wyników treningu
        8. Ocena wyników przy użyciu macierzy pomyłek
        9. Klasyfikacja próbek testowych
    """
    # Sekcja 1: Ładowanie danych
    filepath = "housing.data.txt"
    X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)

    # Sekcja 2: Skalowanie danych
    X_train, X_test = scale_data(X_train, X_test)

    # Sekcja 3: Budowa i trening modelu (mała sieć neuronowa)
    model_small = build_model(X_train.shape[1], layers=[16, 8])
    history_small = model_small.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

    # Sekcja 4: Budowa i trening modelu (duża sieć neuronowa)
    model_large = build_model(X_train.shape[1], layers=[64, 32, 16])
    history_large = model_large.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

    # Sekcja 5: Ocena modeli
    loss_small, accuracy_small = model_small.evaluate(X_test, y_test)
    loss_large, accuracy_large = model_large.evaluate(X_test, y_test)

    print("=== Wyniki modelu z małą siecią neuronową ===")
    print(f"Dokładność na zbiorze testowym: {accuracy_small:.2f}")
    print(f"Strata na zbiorze testowym: {loss_small:.2f}")

    print("=== Wyniki modelu z dużą siecią neuronową ===")
    print(f"Dokładność na zbiorze testowym: {accuracy_large:.2f}")
    print(f"Strata na zbiorze testowym: {loss_large:.2f}")

    # Sekcja 6: Porównanie wyników z innymi metodami klasyfikacji (drzewo decyzyjne, SVM)
    decision_tree_accuracy = 0.81
    decision_SVM_accuracy = 0.84
    print(f"Dokładność drzewa decyzyjnego: {decision_tree_accuracy:.2f}")
    print(f"Dokładność modelu SVM: {decision_SVM_accuracy:.2f}")

    if accuracy_small > accuracy_large and accuracy_small > decision_tree_accuracy and accuracy_small > decision_SVM_accuracy:
        print(f"Największa dokładność to {accuracy_small:.2f}, uzyskana przez model: Mała sieć neuronowa")
    elif accuracy_large > accuracy_small and accuracy_large > decision_tree_accuracy and accuracy_large > decision_SVM_accuracy:
        print(f"Największa dokładność to {accuracy_large:.2f}, uzyskana przez model: Duża sieć neuronowa")
    elif decision_tree_accuracy > accuracy_small and decision_tree_accuracy > accuracy_large and decision_tree_accuracy > decision_SVM_accuracy:
        print(f"Największa dokładność to {decision_tree_accuracy:.2f}, uzyskana przez model: Drzewo decyzyjne")
    else:
        print(f"Największa dokładność to {decision_SVM_accuracy:.2f}, uzyskana przez model: SVM")

    # Sekcja 7: Wizualizacja wyników treningu
    plot_training_history(history_small)
    plot_training_history(history_large)

    # Sekcja 8: Ocena wyników przy użyciu macierzy pomyłek
    y_pred_small = (model_small.predict(X_test) > 0.5).astype(int)
    y_pred_large = (model_large.predict(X_test) > 0.5).astype(int)

    print("Macierz pomyłek dla małej sieci neuronowej:")
    plot_confusion_matrix(y_test, y_pred_small)

    print("Macierz pomyłek dla dużej sieci neuronowej:")
    plot_confusion_matrix(y_test, y_pred_large)

    # Sekcja 9: Klasyfikacja próbek testowych
    print("Przykładowe dane testowe dla małej sieci:")
    classify_samples(model_small, X_test)
    print("\nPrzykładowe dane testowe dla dużej sieci:")
    classify_samples(model_large, X_test)


if __name__ == "__main__":
    """
        Description:
            Uruchomienie głównej funkcji main.
    """
    main()
