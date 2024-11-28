from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        Drzewo decyzyjne jest trenowane na przygotowanych danych i wykorzystywane do przewidywania klasy dla nowych danych. 
        Skrypt zawiera także wizualizację drzewa decyzyjnego oraz ocenę modelu, w tym dokładność klasyfikacji oraz raport z wynikami klasyfikacji.
"""


def load_and_prepare_data(file_path):
    """
    Description:
        Wczytuje dane z pliku CSV i przygotowuje je do analizy.
        Zbiór danych jest uzupełniany o kolumnę klasyfikującą ilość pierścieni na podstawie mediany.

        >> data = pd.read_csv(file_path, sep=",")
            Args:
                - sep=",": Oznacza separator pomiędzy danymi, które są w tym przypadku odzielane przecinkami.
        >> data['Rings_Class'] = (data['Rings'] > median_rings).astype(int)
            - .astype(int): zamienia 0 na False i 0 na True.

    Args:
        - file_path (str): Ścieżka do pliku z danymi.

    Returns:
        - X (DataFrame): Dane wejściowe (wszystkie cechy oprócz Rings i Rings_Class).
        - y (Series): Kolumna docelowa, czyli Rings_Class, 0 lub 1.
    """
    data = pd.read_csv(file_path, sep=",")
    columns = [
        "Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"
    ]
    data.columns = columns
    data["Sex"] = data["Sex"].map({"M": 0, "F": 1, "I": 2})

    median_rings = data['Rings'].median()
    data['Rings_Class'] = (data['Rings'] > median_rings).astype(int)
    X = data.drop(["Rings", "Rings_Class"], axis=1)
    y = data["Rings_Class"]

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
    plt.figure(figsize=(42, 10))
    tree.plot_tree(clf_tree, filled=True, feature_names=feature_names, class_names=["Younger", "Older"], rounded=True, fontsize=9)
    plt.title("Age of Abalone")
    # plt.savefig("decision_tree_abalone_plot.png", bbox_inches="tight")
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

    if isinstance(sample_data[0][0], str):
        """
            Description:
                Sprawdzamy czy w pierwszej wartości podaliśmy (int) czy (str) i zawsze zwraca (int) w to miejsce.
        """
        sample_data[0][0] = {"M": 0, "F": 1, "I": 2}.get(sample_data[0][0].upper())

    sample_data_df = pd.DataFrame(sample_data, columns=feature_names)
    sample_data_scaled = scaler.transform(sample_data_df)
    predicted_class = clf_tree.predict(sample_data_scaled)

    return predicted_class

def main():
    """
        Description:
            Przykładowe wykorzystanie drzewa decyzyjnego na danych "Abalone".
    """

    """
        Description:
            Podział na zbiory treningowe i testowe oraz skalowanie.
    """
    X, y = load_and_prepare_data("abalone.data")

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

    sample_data = [['m',0.39,0.31,0.1,0.406,0.1745,0.093,0.125]]
    predicted_class = predict_sample_data(clf_tree, scaler, sample_data, X.columns)
    print("Przykładowe dane:", sample_data)
    print("Przewidywana klasa:", predicted_class)

if __name__ == "__main__":
    """
        Description:
            Uruchomienie głównej funkcji main.
    """
    main()
