import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dropout
"""
    Autorzy:
        Kamil Powierza
        Dawid Feister

    Wymagane biblioteki:
        1. pandas
        2. sklearn.model_selection
        3. sklearn.preprocessing
        4. tensorflow
        5. matplotlib.pyplot
        6. sklearn.utils
        7. tensorflow.keras.layers

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
        Szukamy odpowiedzi na pytanie czy osoba jest pełnoletnia lub niepełnoletnia.
"""

def load_and_prepare_data(filepath):
    """
        Wczytuje i przetwarza dane z pliku CSV.

        - Tworzy kolumnę `AGE_CLASS`, klasyfikując wiek na podstawie przedziałów.
        - Usuwa wybrane kolumny i przekształca dane wejściowe na zmienne kategoryczne (one-hot encoding).
        - Dzieli dane na zestawy treningowe i testowe.
        - Tasuje dane treningowe dla większej losowości.

        Args:
            filepath (str): Ścieżka do pliku CSV z danymi.

        Returns:
            tuple: Zawiera cztery elementy:
                - X_train (pd.DataFrame): Dane wejściowe do treningu.
                - X_test (pd.DataFrame): Dane wejściowe do testowania.
                - y_train (pd.Series): Etykiety klas dla danych treningowych.
                - y_test (pd.Series): Etykiety klas dla danych testowych.
    """
    column_names = [
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
    data = pd.read_csv(filepath, sep=",", header=None, names=column_names)

    bins = [11, 17, float('inf')]
    labels = [0, 1]
    data['AGE_CLASS'] = pd.cut(data['AGE'], bins=bins, labels=labels, right=True).astype(int)

    X = data.drop(["AGE", "AGE_CLASS", 'DEVICE', 'DEVICES_COUNT', 'PLATFORM', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'], axis=1)
    y = data["AGE_CLASS"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    X_train, y_train = shuffle(X_train, y_train, random_state=3)

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """
        Standaryzuje dane wejściowe, przekształcając je na rozkład o średniej 0 i odchyleniu standardowym 1.

        Args:
            X_train (pd.DataFrame or np.array): Dane wejściowe do treningu.
            X_test (pd.DataFrame or np.array): Dane wejściowe do testowania.

        Returns:
            tuple: Zawiera dwa elementy:
                - X_train_scaled (np.array): Standaryzowane dane treningowe.
                - X_test_scaled (np.array): Standaryzowane dane testowe.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def build_model(input_shape):
    """
        Buduje i kompiluje model sieci neuronowej do klasyfikacji binarnej.

        - Sieć zawiera warstwy w pełni połączone, normalizację batchową oraz regularizację Dropout.
        - Wykorzystuje funkcję aktywacji ReLU w ukrytych warstwach i sigmoidalną w warstwie wyjściowej.

        Args:
            input_shape (int): Liczba cech w danych wejściowych.

        Returns:
            tf.keras.Model: Skonstruowany i skompilowany model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def plot_training_history(history):
    """
        Rysuje wykresy strat i dokładności dla zbiorów treningowego i walidacyjnego.

        Args:
            history (tf.keras.callbacks.History): Obiekt zawierający historię treningu sieci neuronowej.
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
        Klasyfikuje kilka przykładowych danych testowych, wykorzystując wytrenowany model.

        - Wyświetla wartości wejściowe i ich przewidywane klasy.

        Args:
            model (tf.keras.Model): Wytrenowany model.
            X_test (np.array): Dane wejściowe do testowania.

        Prints:
            Przykładowe dane testowe i ich przewidywane klasy.
    """
    sample_data = X_test[:3]
    sample_predictions = model.predict(sample_data)
    sample_predictions = (sample_predictions > 0.5).astype(int)

    print("Przykładowe dane testowe i ich klasyfikacja:")
    for i, prediction in enumerate(sample_predictions):
        print(f"Dane testowe: {sample_data[i]}, Predykcja: {prediction[0]}")


def main():
    """
        Główna funkcja programu realizująca proces przetwarzania danych, trenowania modelu sieci neuronowej
        oraz oceny wyników.

        Kroki:
        1. Wczytuje dane z pliku `Gaming_Dataset.csv` i przygotowuje cechy wejściowe oraz etykiety klas.
        2. Dzieli dane na zbiory treningowy i testowy oraz przeprowadza skalowanie cech.
        3. Buduje model sieci neuronowej, trenuje go i ocenia na zbiorze testowym.
        4. Wizualizuje wyniki treningu i klasyfikuje przykładowe dane testowe.

        Wyniki są wyświetlane w terminalu oraz na wykresach.

        Args:
            Brak argumentów wejściowych.

        Returns:
            None
    """
    filepath = "Gaming_Dataset.csv"
    X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)

    print("Wymiary danych treningowych:", X_train.shape)
    print("Wymiary danych testowych:", X_test.shape)

    X_train, X_test = scale_data(X_train, X_test)

    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=64, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test)
    print("=== Wyniki sieci neuronowej ===")
    print(f"Dokładność na zbiorze testowym: {accuracy:.2f}")
    print(f"Strata na zbiorze testowym: {loss:.2f}")

    plot_training_history(history)

    classify_samples(model, X_test)


if __name__ == "__main__":
    """
        Description:
            Uruchomienie głównej funkcji main.
    """
    main()
