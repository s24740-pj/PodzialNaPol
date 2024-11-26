from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

"""
    Autorzy:
        Kamil Powierza
        Dawid Feister

    Wymagane biblioteki:
        1. sklearn
        2. sklearn.model_selection
        3. sklearn.preprocessing
        4. sklearn.tree
        5. sklearn.metrics
        6. pandas
        7. matplotlib.pyplot

    Description:
        Projekt polega na klasyfikacji mieszkań w Bostonie na podstawie cech:
            1. CRIM: Wskaźnik przestępczości na mieszkańca w danej miejscowości.
            2. ZN: Procent terenów mieszkalnych przeznaczonych na działki większe niż 25,000 stóp kwadratowych.
            3. INDUS: Procent gruntów przeznaczonych na przemysł nienależący do sektora detalicznego w danej miejscowości.
            4. CHAS: Zmienna binarna wskazująca, czy dany obszar graniczy z rzeką Charles (1 = tak, 0 = nie).
            5. NOX: Stężenie tlenków azotu (w częściach na 10 milionów).
            6. RM: Średnia liczba pokoi w mieszkaniach w danej miejscowości.
            7. AGE: Procent jednostek mieszkalnych posiadających właściciela, które zostały wybudowane przed 1940 rokiem.
            8. DIS: Wazona odległość do pięciu głównych ośrodków zatrudnienia w Bostonie.
            9. RAD: Indeks dostępności do dróg promieniowych.
            10. TAX: Stawka podatku od nieruchomości (wartość nieruchomości na $10,000).
            11. PTRATIO: Stosunek liczby uczniów do nauczycieli w danej miejscowości.
            12. B: Wzór: 1000 * (Bk - 0.63)^2, gdzie Bk to proporcja ludności czarnoskórej w danej miejscowości.
            13. LSTAT: Procent ludności o niższym statusie społecznym.
            14. MEDV: Mediana wartości domów właścicieli w tysiącach dolarów.
            
        Na podstawie mediany ceny mieszkań (MEDV) dane są klasyfikowane na dwie grupy: niską (0) i wysoką (1) cenę. Drzewo decyzyjne jest trenowane na przygotowanych danych i wykorzystywane do 
        przewidywania klasy dla nowych danych. Skrypt zawiera także wizualizację drzewa decyzyjnego oraz ocenę modelu, w tym dokładność klasyfikacji oraz raport z wynikami klasyfikacji.
"""


def load_and_prepare_data(file_path):
    """
    Description:
        Wczytuje dane z pliku CSV i przygotowuje je do analizy.
        Zbiór danych jest uzupełniany o kolumnę klasyfikującą ceny domów na podstawie mediany.

        >> data = pd.read_csv(file_path, sep="\\s+")
            Args:
                - sep="\\s+": Oznacza separator pomiędzy danymi, które są w tym przypadku odzielane spacjami.

    Args:
        - file_path (str): Ścieżka do pliku z danymi.

    Returns:
        - X (DataFrame): Dane wejściowe (wszystkie cechy oprócz MEDV i PRICE_CLASS).
        - y (Series): Kolumna docelowa, czyli PRICE_CLASS, 0 lub 1.
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
    Description:
        Dzieli dane na zbiory treningowe i testowe oraz dokonuje skalowania.

    Args:
        - X (DataFrame): Dane wejściowe.
        - y (Series): Kolumna docelowa.

    Returns:
        - X_train (ndarray): Zbiór treningowy po skalowaniu. (Wielowymiarowa tablica)
        - X_test (ndarray): Zbiór testowy po skalowaniu. (Wielowymiarowa tablica)
        - y_train (Series): Kolumna docelowa dla zbioru treningowego. (Jednowymiarowa tablica)
        - y_test (Series): Kolumna docelowa dla zbioru testowego. (Jednowymiarowa tablica)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Skalowanie danych
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_decision_tree(X_train, y_train):
    """
    Description:
        Trenuje model drzewa decyzyjnego na danych treningowych.

    Args:
        - X_train (ndarray): Zbiór treningowy po skalowaniu. (Wielowymiarowa tablica)
        - y_train (Series): Kolumna docelowa dla zbioru treningowego. (Jednowymiarowa tablica)

    Returns:
        - clf_tree (DecisionTreeClassifier): Wytrenowany model drzewa decyzyjnego.
    """
    clf_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf_tree.fit(X_train, y_train)
    return clf_tree


def evaluate_model(clf_tree, X_test, y_test):
    """
    Description:
        Ocenia działanie modelu na zbiorze testowym.

    Args:
        - clf_tree (DecisionTreeClassifier): Wytrenowany model drzewa decyzyjnego.
        - X_test (ndarray): Zbiór testowy po skalowaniu. (Wielowymiarowa tablica)
        - y_test (Series): Kolumna docelowa dla zbioru testowego. (Jednowymiarowa tablica)
    """
    y_pred_tree = clf_tree.predict(X_test)
    print("=== Wyniki drzewa decyzyjnego ===")
    print("Dokładność:", accuracy_score(y_test, y_pred_tree))
    print(classification_report(y_test, y_pred_tree))


def visualize_decision_tree(clf_tree, feature_names):
    """
    Description:
        Wizualizuje model drzewa decyzyjnego.

    Args:
        - clf_tree (DecisionTreeClassifier): Wytrenowany model drzewa decyzyjnego.
        - feature_names (list): Lista nazw cech.
    """
    plt.figure(figsize=(35, 10))
    tree.plot_tree(clf_tree, filled=True, feature_names=feature_names, class_names=["Low", "High"],  rounded=True, fontsize=10)
    plt.title("Boston House Price Decision Tree")
    # plt.savefig("decision_tree_plot.png", bbox_inches="tight")
    plt.show()


def predict_sample_data(clf_tree, scaler, sample_data, feature_names):
    """
    Description:
        Przewiduje klasę dla przykładowych danych z wymaganym wytrenowanym modelem.

    Args:
        - clf_tree (DecisionTreeClassifier): Wytrenowany model drzewa decyzyjnego.
        - scaler (StandardScaler): Dopasowany scaler do przeskalowania danych.
        - sample_data (list): Przykładowe dane wejściowe.
        - feature_names (list): Lista nazw cech.

    Returns:
        - predicted_class (ndarray): Przewidywana klasa dla przykładowych danych.
    """
    sample_data_df = pd.DataFrame(sample_data, columns=feature_names)
    sample_data_scaled = scaler.transform(sample_data_df)
    predicted_class = clf_tree.predict(sample_data_scaled)

    return predicted_class


# Główna funkcja
def main():
    """
        Description:
            Przykładowe wykorzystanie drzewa decyzyjnego na danych "Boston House Price Dataset".
    """

    """
        Description:
            Podział na zbiory treningowe i testowe oraz skalowanie.
    """
    X, y = load_and_prepare_data("housing.data.txt")

    """
        Description:
            Podział na zbiory treningowe i testowe oraz skalowanie.
    """
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

    """
        Description:
            Trening drzewa decyzyjnego.
    """
    clf_tree = train_decision_tree(X_train, y_train)

    """
        Description:
            Ocena modelu.
    """
    evaluate_model(clf_tree, X_test, y_test)

    """
        Description:
            Wizualizacja drzewa decyzyjnego.
    """
    visualize_decision_tree(clf_tree, X.columns)

    """
        Description:
            Przewidywanie dla przykładowych danych.
    """
    sample_data = [[0.1, 25.0, 5.0, 0, 0.5, 6.0, 70.0, 4.0, 1, 300, 15, 390, 5]]
    predicted_class = predict_sample_data(clf_tree, scaler, sample_data, X.columns)
    print("Przykładowe dane:", sample_data)
    print("Przewidywana klasa:", predicted_class)

if __name__ == "__main__":
    """
        Description:
            Uruchomienie głównej funkcji main.
    """
    main()
