from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
"""
    Autorzy:
        Kamil Powierza
        Dawid Feister

    Wymagane biblioteki:
        1. sklearn.svm
        2. sklearn.model_selection
        3. sklearn.preprocessing
        4. sklearn.metrics
        5. pandas
        6. matplotlib.pyplot

    Description:
        Stworzyliśmy ankietę wsród graczy.
        Link do ankiety: https://forms.gle/VGM7HoqHYkRwqTNn9
        Zebraliśmy 20 różnych informacji na podstawie, których model będzie w stanie przewidzieć czy osoba odpowiadająca na 12 z nich jest pełnoletnia.
        50% danych do nauczenia modelu są osobami pełnoletnimi a drugie 50% to osoby niepełnoletnie.
        Odpowiedzi nie były weryfikowane, czyt. nie sprawdzaliśmy czy dana osoba faktycznie ma daną ilość sprzętu czy też sprawdzaliśmy wiek dokumentem tożsamości itd.
        Zaufaliśmy że wszystkie 120 osób odpowiedziało prawidłowo.

        Rozwinięcie skrótów (szczegóły) znajdują się w pliku "README.md".

        W samym projekcie przetestowaliśmy różne rodzaje danych, które będą miały dobrą korelację między sobą.
        Prawidłowy dobór danych jest na tyle istotny, że znacząco wpływa na dokładność modeli.
        Zostało to udowodnione i przedstawione na teście 1 oraz teście 2, gdzie dokładność modelu dla każdego z badanych jąder jest znacząco inna.
        Testy różnią się dwoma pytaniami. W drugim teście nie pytamy o płęć ale za to pytamy o platformę, na której najczęściej gra.
        Szukamy odpowiedzi na pytanie czy osoba jest pełnoletnia lub niepełnoletnia.
        

        Test1:
            "SEX", "PREFERRED_GENRE", "USE_MODS", "PLAY_TIME_OF_DAY", "GAME_MODE", "VOICE_CHAT_START", "COMM_METHOD", "GAME_SKILLS", "NOSTALGIC_GAMES", "FIRST_DEVICE", "ICONIC_MOMENTS", "GAME_SOUNDS"
            X = data.drop(["AGE", "AGE_CLASS", 'DEVICE', 'DEVICES_COUNT', 'PLATFORM', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'], axis=1)

            Adult 22 lat
                # Przykładowe dane: [[0, 0, 1, 3, 1, 3, 0, 1, 9, 6, 3, 4]]
                # Przewidywana klasa: [1]


        Tests2:
            "PREFERRED_GENRE", "USE_MODS", "PLATFORM", "PLAY_TIME_OF_DAY", "GAME_MODE", "VOICE_CHAT_START", "COMM_METHOD", "GAME_SKILLS", "NOSTALGIC_GAMES", "FIRST_DEVICE", "ICONIC_MOMENTS", "GAME_SOUNDS"
            X = data.drop(["AGE", "AGE_CLASS", 'SEX', 'DEVICE', 'DEVICES_COUNT', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'], axis=1)

            Adult 22 lat
                # Przykładowe dane: [[0, 1, 0, 3, 1, 3, 0, 1, 9, 6, 3, 4]]
                # Przewidywana klasa: [1]
"""
def load_and_prepare_data(file_path):
    """
        Wczytuje dane z pliku CSV, przetwarza je, a następnie przygotowuje do użycia w modelu.

        Parametry:
            file_path (str): Ścieżka do pliku CSV z danymi wejściowymi.

        Zwraca:
            X (DataFrame): Zestaw danych wejściowych po przetworzeniu (bez kolumn 'AGE', 'AGE_CLASS' oraz innych).
            y (Series): Zestaw danych wyjściowych (klasa wiekowa: pełnoletni/niepełnoletni).
    """
    data = pd.read_csv(file_path, sep=",")
    columns = [
        "AGE",
        "SEX",
        "PREFERRED_GENRE",
        "USE_MODS",
        "DEVICE",
        "DEVICES_COUNT",
        "PLATFORM",
        "PLAY_TIME",
        "PLAY_TIME_OF_DAY",
        "GAME_MODE",
        "VOICE_CHAT_START",
        "FAV_GAME",
        "VISUAL_IMPACT",
        "COMM_METHOD",
        "GAME_SKILLS",
        "FOLLOW_ESPORTS",
        "NOSTALGIC_GAMES",
        "FIRST_DEVICE",
        "ICONIC_MOMENTS",
        "GAME_SOUNDS"
    ]
    data.columns = columns

    bins = [11, 17, float('inf')]
    labels = [0, 1]
    data['AGE_CLASS'] = pd.cut(data['AGE'], bins=bins, labels=labels, right=True)

    # Test 1
    # X = data.drop(["AGE", "AGE_CLASS", 'DEVICE', 'DEVICES_COUNT', 'PLATFORM', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'], axis=1)
    # Test 2
    X = data.drop(["AGE", "AGE_CLASS", 'SEX', 'DEVICE', 'DEVICES_COUNT', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'], axis=1)
    y = data["AGE_CLASS"]

    return X, y


def split_and_scale_data(X, y):
    """
        Dzieli dane na zestawy treningowe i testowe oraz skaluje je za pomocą standaryzacji.

        Parametry:
            X (DataFrame): Dane wejściowe.
            y (Series): Etykiety (klasy wiekowe).

        Zwraca:
            X_train (ndarray): Znormalizowane dane treningowe.
            X_test (ndarray): Znormalizowane dane testowe.
            y_train (Series): Etykiety dla danych treningowych.
            y_test (Series): Etykiety dla danych testowych.
            scaler (StandardScaler): Obiekt skalera, który został użyty do skalowania danych.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_svc_models(X_train, y_train, kernel_params):
    """
        Trenuje modele SVC z różnymi jądrami na danych treningowych.

        Parametry:
            X_train (ndarray): Znormalizowane dane treningowe.
            y_train (Series): Etykiety treningowe.
            kernel_params (dict): Parametry jądra dla różnych typów jądra SVC (np. 'linear', 'poly', itp.).

        Zwraca:
            models (dict): Słownik modeli SVC dla każdego jądra.
    """
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    models = {}
    for kernel in kernels:
        params = kernel_params.get(kernel, {})
        svc = SVC(kernel=kernel, random_state=42, **params)
        svc.fit(X_train, y_train)
        models[kernel] = svc
    return models


def evaluate_models(models, X_test, y_test):
    """
        Ocena wydajności modeli SVC na danych testowych. Oblicza dokładność oraz generuje raport klasyfikacji.

        Parametry:
            models (dict): Modele SVC z różnymi jądrami.
            X_test (ndarray): Znormalizowane dane testowe.
            y_test (Series): Etykiety testowe.

        Zwraca:
            results (dict): Słownik dokładności modeli dla różnych jąder.
    """
    results = {}
    for kernel, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[kernel] = accuracy
        print(f"=== Wyniki SVC z jądrem {kernel} Test 2===")
        print("Dokładność:", accuracy)
        print(classification_report(y_test, y_pred))
    return results


def predict_for_input(models, input_data, scaler):
    """
        Przewiduje klasy dla nowych danych wejściowych na podstawie wytrenowanych modeli.

        Parametry:
            models (dict): Modele SVC z różnymi jądrami.
            input_data (list): Nowe dane wejściowe do przewidzenia.
            scaler (StandardScaler): Obiekt skalera do transformacji danych wejściowych.

        Zwraca:
            predictions (dict): Słownik przewidywanych klas dla różnych jąder SVC.
    """
    input_data_scaled = scaler.transform(input_data)

    predictions = {}
    for kernel, model in models.items():
        prediction = model.predict(input_data_scaled)
        predictions[kernel] = prediction
        print(f"=== Przewidywana klasa dla jądra {kernel} Test 2===")
        print("Przewidywana klasa:", prediction)

    return predictions


def plot_accuracy_comparison(results):
    """
        Tworzy wykres porównujący dokładności modeli SVC z różnymi jądrami.

        Parametry:
            results (dict): Słownik dokładności modeli dla różnych jąder.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(results.keys(), results.values(), color=['blue', 'orange', 'green', 'red'])
    plt.title('Porównanie dokładności dla różnych jąder SVC Test 2')
    plt.xlabel('Jądro')
    plt.ylabel('Dokładność')
    plt.show()


def main():
    """
        Główna funkcja programu, która wykonuje pełny proces od wczytania danych po oceny modeli i przewidywania.

        Funkcja wykonuje następujące kroki:
        1. Wczytuje dane z pliku CSV za pomocą funkcji `load_and_prepare_data()`.
        2. Dzieli dane na zestawy treningowe i testowe oraz skaluje je przy użyciu funkcji `split_and_scale_data()`.
        3. Definiuje parametry jąder dla różnych modeli SVC i trenuje modele za pomocą funkcji `train_svc_models()`.
        4. Oceny skuteczności modeli (dokładność i raport klasyfikacji) na danych testowych są generowane przez funkcję `evaluate_models()`.
        5. Tworzy wykres porównujący dokładności modeli z różnymi jądrami przy pomocy funkcji `plot_accuracy_comparison()`.
        6. Przewiduje klasę (pełnoletność) dla przykładowych danych wejściowych za pomocą funkcji `predict_for_input()`.

        Parametry:
            Brak.

        Zwraca:
            Brak.

        Opis:
            Funkcja `main()` jest punktem wyjścia dla całego programu. Wykonuje wszystkie niezbędne kroki do trenowania i oceny modelu, a także do generowania wyników przewidywań dla nowych danych.
            Testuje różne modele SVC z różnymi jądrami i ocenia ich skuteczność na podstawie danych testowych. Na końcu, użytkownik otrzymuje wyniki w formie tekstowej oraz wizualnej (wykresu porównania dokładności).
    """
    X, y = load_and_prepare_data("Gaming_Dataset.csv")
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

    kernel_params = {
        'linear': {},
        'poly': {'degree': 3, 'coef0': 1},
        'rbf': {'gamma': 'scale'},
        'sigmoid': {'gamma': 'scale', 'coef0': 1}
    }

    models = train_svc_models(X_train, y_train, kernel_params)
    results = evaluate_models(models, X_test, y_test)
    plot_accuracy_comparison(results)

    # Test 1
    # input_data = [[0, 0, 1, 3, 1, 3, 0, 1, 9, 6, 3, 4]]
    # Test 2
    input_data = [[0, 1, 0, 3, 1, 3, 0, 1, 9, 6, 3, 4]]
    predict_for_input(models, input_data, scaler)


if __name__ == "__main__":
    """
        Description:
            Uruchomienie głównej funkcji main.
    """
    main()
