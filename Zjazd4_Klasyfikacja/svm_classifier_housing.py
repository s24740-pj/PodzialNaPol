from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
"""
    Autorzy:
        Kamil Powierza
        Dawid Feister

    Wymagane biblioteki:
        1. sklearn.svm
        2. sklearn.metrics
        3. sklearn.decomposition
        4. pandas
        5. numpy
        6. matplotlib.pyplot
        7. sklearn.preprocessing
        8. sklearn.model_selection
        9. matplotlib.patches
        10. matplotlib.colors

    Opis:
        Projekt klasyfikacji mieszkań w Bostonie na podstawie cech [szczegółowy opis cech znajduje się w pliku "readme.md"], wykorzystujący SVM (Support Vector Machine).
        Dane są klasyfikowane na dwie grupy: niską (0) i wysoką (1) cenę, w oparciu o medianę wartości mieszkań (MEDV).
        Kod obejmuje wczytanie danych, przygotowanie ich do analizy, budowę modelu SVC, ocenę wyników oraz wizualizację wyników klasyfikacji.

    Instrukcja użycia:
        1. Upewnij się, że plik "housing.data.txt" jest w odpowiednim miejscu.
        2. Uruchom kod, aby załadować dane, przeprowadzić klasyfikację SVM i zobaczyć wyniki.
        3. Możesz także przetestować model na przykładowych danych wejściowych.
"""


def load_and_prepare_data(file_path):
    """
        Wczytuje dane z pliku, przetwarza je i przygotowuje do analizy.

        Parametry:
            file_path (str): Ścieżka do pliku z danymi.

        Zwraca:
            X (DataFrame): Dane wejściowe (cechy).
            y (Series): Wynik (klasa ceny).
    """
    data = pd.read_csv(file_path, sep="\\s+")
    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ]
    data.columns = columns

    median_price = data["MEDV"].median()
    data["PRICE_CLASS"] = (data["MEDV"] > median_price).astype(int)
    X = data.drop(["MEDV", "PRICE_CLASS"], axis=1)
    y = data["PRICE_CLASS"]

    return X, y


def split_and_scale_data(X, y):
    """
        Dzieli dane na zestawy treningowe i testowe oraz je skaluje.

        Parametry:
            X (DataFrame): Dane wejściowe (cechy).
            y (Series): Wynik (klasa ceny).

        Zwraca:
            X_train (ndarray): Zestaw treningowy danych wejściowych.
            X_test (ndarray): Zestaw testowy danych wejściowych.
            y_train (ndarray): Zestaw treningowy wyników.
            y_test (ndarray): Zestaw testowy wyników.
            scaler (StandardScaler): Obiekt skalujący dane.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_svm(X_train, y_train):
    """
        Trenuje model SVM na danych treningowych.

        Parametry:
            X_train (ndarray): Zestaw treningowy danych wejściowych.
            y_train (ndarray): Zestaw treningowy wyników.

        Zwraca:
            clf_svm (SVC): Wytrenowany model SVM.
    """
    clf_svm = SVC(kernel='linear', random_state=42)
    clf_svm.fit(X_train, y_train)
    return clf_svm


def evaluate_model(clf_svm, X_test, y_test):
    """
        Oceny modelu SVM na danych testowych, wyświetlanie dokładności i raportu klasyfikacji.

        Parametry:
            clf_svm (SVC): Wytrenowany model SVM.
            X_test (ndarray): Zestaw testowy danych wejściowych.
            y_test (ndarray): Zestaw testowy wyników.
    """
    y_pred_svm = clf_svm.predict(X_test)
    print("=== Wyniki SVM ===")
    print("Dokładność:", accuracy_score(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm))


def visualize_support_vectors(clf_svm, X, y):
    """
        Wizualizuje wektory wspierające w przestrzeni 2D po redukcji wymiaru za pomocą PCA.

        Parametry:
            clf_svm (SVC): Wytrenowany model SVM.
            X (DataFrame): Dane wejściowe.
            y (Series): Wyniki.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    y_2d = y.to_numpy()

    h = .02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_13d = pca.inverse_transform(grid_points)
    Z = clf_svm.predict(grid_points_13d)

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)

    colors = ['purple', 'yellow']
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_2d, edgecolor='k', marker='o', cmap=mcolors.ListedColormap(colors))

    legend_labels = ['Niska cena (0)', 'Wysoka cena (1)']
    patches = [mpatches.Patch(color='purple', label=legend_labels[0]),
               mpatches.Patch(color='yellow', label=legend_labels[1])]
    plt.legend(handles=patches)

    plt.title("SVC - Wyniki PCA (2D)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


def predict_sample_data(clf_svm, scaler, sample_data):
    """
        Przewiduje klasę na podstawie przykładowych danych wejściowych.

        Parametry:
            clf_svm (SVC): Wytrenowany model SVM.
            scaler (StandardScaler): Obiekt skalujący dane.
            sample_data (list): Przykładowe dane wejściowe do klasyfikacji.

        Zwraca:
            prediction (ndarray): Przewidywana klasa.
    """
    sample_scaled = scaler.transform([sample_data])
    prediction = clf_svm.predict(sample_scaled)
    print("Przewidywana klasa dla danych:", prediction)
    return prediction


if __name__ == "__main__":
    """
        Główna funkcja uruchamiająca skrypt.

        Proces wykonuje następujące kroki:
        1. Wczytuje dane z pliku "housing.data.txt".
        2. Przygotowuje dane, tworząc zmienne wejściowe (X) i wyjściowe (y).
        3. Dzieli dane na zbiory treningowe i testowe oraz dokonuje ich standaryzacji.
        4. Trenuje model klasyfikacji SVM (Support Vector Machine) na danych treningowych.
        5. Ocena modelu na zbiorze testowym, wyświetlenie raportu o dokładności.
        6. Wizualizacja wyników klasyfikacji i wykrytych wektorów nośnych (support vectors).
        7. Dokonuje predykcji na przykładowych danych wejściowych.

        Parametry:
            Brak

        Zwraca:
            Brak
    """
    file_path = "housing.data.txt"
    X, y = load_and_prepare_data(file_path)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

    clf_svm = train_svm(X_train, y_train)
    evaluate_model(clf_svm, X_test, y_test)
    visualize_support_vectors(clf_svm, X_train, y_train)

    # Przykładowe dane
    sample_data = [0.03, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296.0, 15.3, 396.9, 4.98]
    predict_sample_data(clf_svm, scaler, sample_data)
