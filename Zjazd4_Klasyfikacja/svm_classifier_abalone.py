import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
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
        5. sklearn.svm
        6. sklearn.metrics
        7. matplotlib.pyplot
        8. seaborn

    Description:
        Projekt polega na klasyfikacji wieku muszli abalonów na podstawie różnych cech fizycznych, takich jak wymiary, wagi i płeć muszli.
        Wartość Rings reprezentuje liczbę pierścieni na muszli, co odpowiada jej wiekowi.
        Poniżej opis parametrów w bazie:
            1. Sex: Płeć muszli. Może przyjąć wartości M = 0 (mężczyzna), F = 1 (kobieta) lub I = 2 (niemowlę).
            2. Length: Długość muszli, mierzona w milimetrach (mm), będąca najdłuższym wymiarem muszli.
            3. Diameter: Średnica muszli, mierzona w milimetrach (mm), w kierunku prostopadłym do długości.
            4. Height: Wysokość muszli wraz z mięsem, mierzona w milimetrach (mm).
            5. Whole weight: Całkowita waga muszli, mierzona w gramach (g), obejmująca muszlę oraz mięso.
            6. Shucked weight: Waga samego mięsa, mierzona w gramach (g), po oddzieleniu od muszli.
            7. Viscera weight: Waga wnętrzności muszli, mierzona w gramach (g), po usunięciu wnętrzności.
            8. Shell weight: Waga muszli, mierzona w gramach (g), po wysuszeniu.
            9. Rings: Liczba pierścieni na muszli. To cecha, którą chcemy przewidywać.

        Na podstawie mediany ilości pierścieni dane są klasyfikowane na dwie grupy: Younger (0) i Older (1).
        Model SVC jest trenowany na przygotowanych danych i wykorzystywane do przewidywania klasy dla nowych danych. 
        Skrypt zawiera także przykładową wizualizację klasyfikacji SVC, oraz ocenę modelu, w tym dokładność klasyfikacji oraz raport z wynikami klasyfikacji.
        
    BONUS:
        Dodatkowo skrypt zawiera rozwiązanie zadania bonusowego w zakresie użycia różnych rodzaji kernel function z różnymi parametrami.
"""
def load_and_prepare_data():
    """
        Wczytuje dane z pliku 'abalone.data', mapuje wartości 'Sex' na liczby,
        oraz tworzy nową kolumnę 'Rings_Class' na podstawie mediany wartości 'Rings'.

        Zwraca:
            data (DataFrame): Przetworzone dane zawierające cechy muszli i klasyfikację wieku.
    """
    columns = [
        "Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight",
        "Viscera weight", "Shell weight", "Rings"
    ]
    data = pd.read_csv("abalone.data", header=None, names=columns)

    sex_mapping = {'M': 0, 'F': 1, 'I': 2}
    data['Sex'] = data['Sex'].map(sex_mapping)

    median_rings = data["Rings"].median()
    data["Rings_Class"] = (data["Rings"] > median_rings).astype(int)

    return data


def split_data(data):
    """
        Dzieli dane na cechy (X) i etykiety (y), a następnie dzieli je na zestaw treningowy i testowy.

        Parametry:
            data (DataFrame): Dane wejściowe z cechami i etykietami.

        Zwraca:
            X_train (ndarray): Zestaw treningowy cech.
            X_test (ndarray): Zestaw testowy cech.
            y_train (ndarray): Zestaw treningowy etykiet.
            y_test (ndarray): Zestaw testowy etykiet.
    """
    X = data.drop(columns=["Rings", "Rings_Class"])
    y = data["Rings_Class"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    """
        Skaluje dane za pomocą StandarScaler, normalizując cechy w zestawach treningowym i testowym.

        Parametry:
            X_train (ndarray): Zestaw treningowy cech.
            X_test (ndarray): Zestaw testowy cech.

        Zwraca:
            X_train_scaled (ndarray): Skalowane dane treningowe.
            X_test_scaled (ndarray): Skalowane dane testowe.
            scaler (StandardScaler): Obiekt skaleru używany do skalowania danych.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train, kernel_type='rbf', C=1, gamma='scale', degree=3):
    """
        Trenuje model SVC (Support Vector Classifier) na podstawie danych treningowych.

        Parametry:
            X_train (ndarray): Zestaw treningowy cech.
            y_train (ndarray): Zestaw treningowy etykiet.
            kernel_type (str): Typ jądra SVC (domyślnie 'rbf').
            C (float): Parametr regularyzacji (domyślnie 1).
            gamma (str): Parametr gamma jądra (domyślnie 'scale').
            degree (int): Stopień jądra wielomianowego (domyślnie 3).

        Zwraca:
            svc (SVC): Wytrenowany model SVC.
    """
    svc = SVC(kernel=kernel_type, C=C, gamma=gamma, degree=degree, random_state=42)
    svc.fit(X_train, y_train)
    return svc


def evaluate_model(svc, X_test, y_test):
    """
        Ocena modelu SVC na podstawie danych testowych, w tym dokładności, raportu klasyfikacji i macierzy pomyłek.

        Parametry:
            svc (SVC): Wytrenowany model SVC.
            X_test (ndarray): Zestaw testowy cech.
            y_test (ndarray): Zestaw testowy etykiet.

        Zwraca:
            accuracy (float): Dokładność modelu.
            report (str): Raport klasyfikacji.
            conf_matrix (ndarray): Macierz pomyłek.
    """
    y_pred = svc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, conf_matrix


def plot_data(data):
    """
        Tworzy wykres rozrzutu przedstawiający zależność między długością muszli a jej wagą,
        z kolorowaniem punktów w zależności od klasy 'Rings_Class'.

        Parametry:
            data (DataFrame): Dane wejściowe z cechami muszli i klasyfikacją wieku.
    """
    hue_labels = {0: "Younger", 1: "Older"}

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=data["Length"], y=data["Whole weight"], hue=data["Rings_Class"],
        palette=["red", "blue"], alpha=0.7, hue_order=[0, 1]
    )
    plt.title("Wizualizacja danych: Length vs Whole weight")
    plt.xlabel("Length")
    plt.ylabel("Whole weight")

    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [hue_labels[int(label)] for label in labels]
    plt.legend(handles, labels, title="Rings_Class", loc="upper left", bbox_to_anchor=(0, 1))

    plt.show()


def classify_example(example_data, scaler, svc):
    """
        Przewiduje klasę dla przykładowych danych, stosując skalowanie i model SVC.

        Parametry:
            example_data (ndarray): Dane do klasyfikacji.
            scaler (StandardScaler): Obiekt skaleru używany do skalowania danych.
            svc (SVC): Wytrenowany model SVC.

        Zwraca:
            predicted_class (ndarray): Przewidywana klasa dla danych wejściowych.
    """
    example_data_scaled = scaler.transform(example_data)
    predicted_class = svc.predict(example_data_scaled)
    return predicted_class

def main():
    """
        Funkcja główna, która wykonuje cały proces: ładowanie danych, trenowanie modelu, ocena i wizualizacja.

        Wywołuje funkcje pomocnicze, przeprowadza eksperymenty z różnymi typami jąder SVM, prezentuje wyniki
        klasyfikacji i wybiera najlepszy model na podstawie dokładności. Dodatkowo przewiduje klasę dla przykładowych
        danych i wyświetla wizualizację zbioru danych.

        Process:
            1. Ładowanie i przygotowanie danych.
            2. Podział danych na zestawy treningowe i testowe.
            3. Skalowanie cech przy użyciu standardyzacji.
            4. Trening modeli SVM dla różnych rodzajów jąder.
            5. Ocena modeli: dokładność, raport klasyfikacji, macierz pomyłek.
            6. Wybór najlepszego modelu na podstawie dokładności.
            7. Przewidywanie klasy dla przykładowych danych.
            8. Wizualizacja danych.

        Returns:
            None: Funkcja nie zwraca żadnej wartości, ale drukuje wyniki klasyfikacji oraz wizualizacje.
    """
    data = load_and_prepare_data()

    X_train, X_test, y_train, y_test = split_data(data)

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
    best_accuracy = 0
    best_kernel = ''

    for kernel in kernel_types:
        svc = train_model(X_train_scaled, y_train, kernel_type=kernel)

        accuracy, report, conf_matrix = evaluate_model(svc, X_test_scaled, y_test)

        print(f"Dokładność z jądrem {kernel}: {accuracy}")
        print(report)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_kernel = kernel

    print(f"Najlepsze jądro to: {best_kernel} z dokładnością: {best_accuracy}")

    example_data = np.array([[0, 0.39, 0.31, 0.1, 0.406, 0.1745, 0.093, 0.125]])
    predicted_class = classify_example(example_data, scaler, svc)
    print(f"\nPrzykładowe dane: {example_data.tolist()}")
    print(f"Przewidywana klasa: {predicted_class.tolist()}")

    plot_data(data)

if __name__ == "__main__":
    """
        Description:
            Uruchomienie głównej funkcji main.
    """
    main()