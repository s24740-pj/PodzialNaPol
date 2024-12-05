from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Todo
# Readme:
#   Wizualizacja korelacji
#   Przykład danych RAW i przerobione przygotowane do modelu
#   Link do ankiety
#   Wykresy z ankiety wyeksportować
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
        8. seaborn

    Description:
        Stworzyliśmy ankietę wsród graczy.
        Link do ankiety: https://forms.gle/VGM7HoqHYkRwqTNn9
        Zebraliśmy 20 różnych informacji na podstawie, których model będzie w stanie przewidzieć czy osoba odpowiadająca na 12 z nich jest pełnoletnia.
        50% danych do nauczenia modelu są osobami pełnoletnimi a drugie 50% to osoby niepełnoletnie.
        Odpowiedzi nie były weryfikowane, czyt. nie sprawdzaliśmy czy dana osoba faktycznie ma daną ilość sprzętu czy też sprawdzaliśmy wiek dokumentem tożsamości itd.
        Zaufaliśmy że wszystkie 120 osób odpowiedziało prawidłowo.
        
        Rozwinięcie skrótów:
        1. AGE: Ile masz lat? (Podaj wiek)
            [Integer]
        2. SEX: Jaka jest Twoja płeć? 
            0: Mężczyzna
            1: Kobieta
            2: Inna
        4. PREFERRED_GENRE: Jaki gatunek gier preferujesz?
            0: Strzelanki
            1: Battle Royale
            2: Gry RPG; Souls Like
            3: Gry strategiczne
            4: Symulatory
            5: Gry logiczne
            6: Gry przygodowe
            7: Inne
        5. USE_MODS: Czy używasz modów do gier?
            0: Tak, często
            1: Okazjonalnie
            2: Nie
        6. DEVICE: Na jakim sprzęcie grasz najczęściej?
            0: PC
            1: Konsola (PS/Xbox)
            2: Mobilne urządzenia
            3: Inne
        7. DEVICES_COUNT: Z ilu sprzętów korzystasz do grania? (Podaj ilość)
            [Integer] Przedział od 0 do 10, gdzie 10 to 10 lub więcej.
        8. PLATFORM: Na jakiej platformie grasz najczęściej?
            0: Steam
            1: Origin
            2: Epic Games
            3: Inne
        9. PLAY_TIME: Ile czasu dziennie średnio spędzasz na grach (Podaj w godzinach)?
            [Integer] Przedział od 0 do 10, gdzie 10 to 10 lub więcej.
        10. PLAY_TIME_OF_DAY: O jakiej porze dnia preferujesz granie?
            0: Rano
            1: Po południe
            2: Wieczór
            3: Późna noc
        11. GAME_MODE: Jaki tryb gry preferujesz?
            0: Gra lokalna (Singleplayer)
            1: Gra online (Multiplayer)
        12. VOICE_CHAT_START: Kiedy zacząłeś/aś używać czatu głosowego w grach komputerowych?
            0: Nigdy
            1: Przed 2000
            2: 2000–2010
            3: 2010–2020
            4: Po 2020
        13. FAV_GAME: Jaką gre lubisz najbardziej? (Wybierz jedną)
            0: Counter Strike (2, Global Offensive)
            1: Valorant
            2: Fortnite
            3: Minecraft
            4: League of Legends
            5: The Sims 4
            6: FIFA (np. FIFA 24)
            7: Call of Duty
            8: Roblox
            9: GTA V
            10: Mario
            11: Zelda
            12: Pokémon
            13: Inna
        14. VISUAL_IMPACT: Która z poniższych gier wywarła na Tobie największe wrażenie wizualne? (Wybierz jedną)
            0: Mario
            1: The Legend of Zelda: Breath of the Wild
            2: Final Fantasy
            3: Half-Life 2
            4: Vrchat
            5: Fortnite
            6: The Witcher 3: Wild Hunt
            7: Horizon Zero Dawn
            8: Cyberpunk 2077
            9: GTA 5
            10: Inne
        15. COMM_METHOD: W jakiej formie najczęściej komunikujesz się z innymi podczas grania?
            0: Discord
            1: TeamSpeak
            2: Inne
        16. GAME_SKILLS: Jak oceniasz swoje umiejętności w grach komputerowych?
            0: Bardzo zaawansowane
            1: Średnio zaawansowane
            2: Podstawowe
            3: Brak umiejętności
        17. FOLLOW_ESPORTS: Czy śledzisz esport lub zawody gamingowe?
            0: Tak, regularnie
            1: Tak, sporadycznie
            2: Nie, nie interesuje mnie to
        18. NOSTALGIC_GAMES: W skali od 0-10 jak bardzo lubisz gry nostalgiczne/stare (np. Quake, Diablo 1, Tetris, Mario, itp.)
            [Integer]
        19. FIRST_DEVICE: Jaki był twój pierwszy sprzęt, na którym zagrałeś/aś swoją pierwszą grę (Jeżeli nie dokładnie to bardzo podobny).
            0: Atari
            1: NES/SNES
            2: PlayStation 1/2
            3: Xbox (Original/360)
            4: Game Boy/DS
            5: Współczesna konsola (PS4/PS5, Xbox One/Series, itp.)
            6: PC (przed 2005 rokiem)
            7: PC (po 2005 roku)
            8: Smartfon/Telefon (przed 2005 rokiem)
            9: Smartfon/Telefon (po 2005 roku)
        20. ICONIC_MOMENTS: Które z tych kultowych momentów związanych z grami pamiętasz, jako najstarsze?
            0: Premiera Pokémon Red/Blue
            1: Hype wokół gier w czasach ery Y2K
            2: Premiera Halo 3
            3: Szał na Minecrafta w latach 2010
            4: Fortnite World Cup
        21. GAME_SOUNDS: Który z tych dźwięków lub utworów z gier najbardziej utkwił Ci w pamięci?
            0: Dźwięk zbierania monety z Mario
            1: Motyw przewodni z "The Legend of Zelda"
            2: Intro z "Halo"
            3: Motyw przewodni z "Tetrisa"
            4: „Still Alive” z "Portal"
            5: Dźwięk Creepera z "Minecraft"
            6: Dźwięk z "Roblox" "Death Sound"
            7: Motyw przewodni z "The Elder Scrolls V: Skyrim" (Dragonborn)
            8: „Megalovania” z "Undertale"
            9: Dźwięk piosenki Victory Royale z "Fortnite"
            10: Motyw przewodni z "The Last of Us" (melancholijna gitara)
            11: Muzyka z lobby CSGO
        
        W samym projekcie przetestowaliśmy różne rodzaje danych, które będą miały dobrą korelację między sobą na tyle by osiągnąć około 80% dokładności.
        Zapytaiśmy pare prawdziwych osób do przetestowania naszego modelu oraz postawiliśmy ChatGPT na to by odpowiedział na pytania jako osoba pełnoletnia i niepełnoletnia.
        Sprawdziliśmy to na dwóch testach, różnią się one dwoma pytaniami. W drugim teście nie pytamy o płęć ale za to pytamy o platformę, na której najczęściej gra.
        Wszystkim odpowiadającym również ufamy że odpowiadają poprawnie.
        
        Test1:
            "SEX", "PREFERRED_GENRE", "USE_MODS", "PLAY_TIME_OF_DAY", "GAME_MODE", "VOICE_CHAT_START", "COMM_METHOD", "GAME_SKILLS", "NOSTALGIC_GAMES", "FIRST_DEVICE", "ICONIC_MOMENTS", "GAME_SOUNDS"
            X = data.drop(["AGE", "AGE_CLASS", 'DEVICE', 'DEVICES_COUNT', 'PLATFORM', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'], axis=1)
            
            Prawdziwe osoby:
                Kid 12 lat
                    # Przykładowe dane: [["M", 1, 2, 2, 1, 4, 0, 2, 8, 3, 3, 0]]
                    # Przewidywana klasa: [1]
                    # Dodatkowe informacje: Przypadek tej osoby jest taki że odpowiedział że grał swoje pierwsze gry na xbox 360 gdzie nie do końca jest ta konsola już w użytku, tymabrdziej w jego bardzo młodym wieku.
                    # Tak samo odpowiedział że dźwięk monety z mario najbardziej utkwił w jego pamięci, gdzie jest to jednak gra bardzo bardzo stara i nie jest już grana w tym pokoleniu jak było to w bardzo wczesnych latach 2000.
                    # Również odpowiedział że pamięta jako najstarsze kultowe wydarzenie "Szał na minecrafta w latach 2010" mimo że urodził się dopiero 2 lata później w 2012 roku.
                Adult 21 lat
                    # Przykładowe dane: [["M", 2, 1, 1, 1, 3, 0, 1, 7, 2, 3, 7]]
                    # Przewidywana klasa: [1]
                Adult 21 lat
                    # Przykładowe dane: [["M", 2, 1, 3, 1, 3, 0, 1, 6, 6, 3, 5]]
                    # Przewidywana klasa: [1]
                Adult 22 lat
                    # Przykładowe dane: [["M", 0, 1, 3, 1, 3, 0, 1, 9, 6, 3, 4]]
                    # Przewidywana klasa: [1]
                
                
            AI (ChatGPT):
                Kid 13 lat
                    # Przykładowe dane: [["M", 1, 1, 2, 0, 3, 1, 1, 6, 6, 3, 6]]
                    # Przewidywana klasa: [0]
                Kid 14 lat
                    # Przykładowe dane: [["M", 0, 2, 3, 0, 4, 0, 0, 7, 8, 4, 7]] 
                    # Przewidywana klasa: [0]
                Kid 15 lat
                    # Przykładowe dane: [["M", 3, 0, 2, 1, 4, 0, 1, 8, 7, 3, 5]]
                    # Przewidywana klasa: [1]
                Kid 16 lat
                    # Przykładowe dane: [["K", 6, 1, 3, 0, 3, 0, 0, 6, 9, 3, 9]]
                    # Przewidywana klasa: [0]
                Kid 17 lat
                    # Przykładowe dane: [["K", 5, 0, 2, 1, 4, 0, 2, 6, 7, 4, 9]]
                    # Przewidywana klasa: [1]
                Adult 18 lat
                    # Przykładowe dane: [["M", 1, 2, 3, 0, 4, 1, 1, 6, 6, 3, 6]]
                    # Przewidywana klasa: [0]
                Adult 22 lat
                    # Przykładowe dane: [["K", 2, 1, 3, 1, 4, 0, 0, 7, 8, 4, 7]]
                    # Przewidywana klasa: [0]
                Adult 25 lat
                    # Przykładowe dane: [["M", 3, 0, 2, 1, 4, 0, 1, 8, 7, 3, 5]]
                    # Przewidywana klasa: [1]
                Adult 27 lat
                    # Przykładowe dane: [["K", 0, 1, 1, 0, 3, 1, 2, 6, 9, 3, 7]]
                    # Przewidywana klasa: [1]
                Adult 30 lat
                    # Przykładowe dane: [["M", 2, 3, 3, 0, 3, 0, 2, 8, 8, 3, 6]]
                    # Przewidywana klasa: [0]
            
        Tests2:
            "PREFERRED_GENRE", "USE_MODS", "PLATFORM", "PLAY_TIME_OF_DAY", "GAME_MODE", "VOICE_CHAT_START", "COMM_METHOD", "GAME_SKILLS", "NOSTALGIC_GAMES", "FIRST_DEVICE", "ICONIC_MOMENTS", "GAME_SOUNDS"
            X = data.drop(["AGE", "AGE_CLASS", 'SEX', 'DEVICE', 'DEVICES_COUNT', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'], axis=1)
            
            Prawdziwe osoby:
                Kid 12 lat
                    # Przykładowe dane: [[1, 2, 0, 2, 1, 4, 0, 2, 8, 3, 3, 0]]
                    # Przewidywana klasa: [1]
                    # Dodatkowe informacje: Przypadek tej osoby jest taki że odpowiedział że grał swoje pierwsze gry na xbox 360 gdzie nie do końca jest ta konsola już w użytku, tymabrdziej w jego bardzo młodym wieku.
                    # Tak samo odpowiedział że dźwięk monety z mario najbardziej utkwił w jego pamięci, gdzie jest to jednak gra bardzo bardzo stara i nie jest już grana w tym pokoleniu jak było to w bardzo wczesnych latach 2000.
                    # Również odpowiedział że pamięta jako najstarsze kultowe wydarzenie "Szał na minecrafta w latach 2010" mimo że urodził się dopiero 2 lata później w 2012 roku.
                Adult 21 lat
                    # Przykładowe dane: [[2, 3, 0, 1, 1, 3, 0, 1, 7, 2, 3, 7]]
                    # Przewidywana klasa: [1]
                Adult 21 lat
                    # Przykładowe dane: [[2, 3, 0, 3, 1, 3, 0, 1, 6, 6, 3, 7]]
                    # Przewidywana klasa: [1]
                Adult 22 lat
                    # Przykładowe dane: [[0, 1, 0, 3, 1, 3, 0, 1, 9, 6, 3, 4]]
                    # Przewidywana klasa: [1]
            
            AI (ChatGPT):
                Kid 13 lat
                    # Przykładowe dane: [[1, 1, 0, 2, 0, 3, 1, 1, 6, 6, 3, 6]]
                    # Przewidywana klasa: [0]
                Kid 14 lat
                    # Przykładowe dane: [[0, 2, 2, 3, 0, 4, 0, 0, 7, 8, 4, 7]] 
                    # Przewidywana klasa: [0]
                Kid 15 lat
                    # Przykładowe dane: [[3, 0, 2, 2, 1, 4, 0, 1, 8, 7, 3, 5]]
                    # Przewidywana klasa: [1]
                Kid 16 lat
                    # Przykładowe dane: [[6, 1, 0, 3, 0, 3, 0, 0, 6, 9, 3, 9]]
                    # Przewidywana klasa: [0]
                Kid 17 lat
                    # Przykładowe dane: [[5, 0, 2, 2, 1, 4, 0, 2, 6, 7, 4, 9]]
                    # Przewidywana klasa: [1]
                Adult 18 lat
                    # Przykładowe dane: [[1, 2, 0, 3, 0, 4, 1, 1, 6, 6, 3, 6]]
                    # Przewidywana klasa: [0]
                Adult 22 lat
                    # Przykładowe dane: [[2, 1, 0, 3, 1, 4, 0, 0, 7, 8, 4, 7]]
                    # Przewidywana klasa: [0]
                Adult 25 lat
                    # Przykładowe dane: [[2, 0, 1, 2, 1, 4, 0, 1, 8, 7, 3, 5]]
                    # Przewidywana klasa: [1]
                Adult 27 lat
                    # Przykładowe dane: [[0, 1, 3, 1, 0, 3, 1, 2, 6, 9, 3, 7]]
                    # Przewidywana klasa: [1]
                Adult 23 lat
                    # Przykładowe dane: [[2, 3, 0, 3, 0, 3, 0, 2, 8, 8, 3, 6]]
                    # Przewidywana klasa: [0]
        
        Poniżej kombinacje informacji, które nie są potrzebne do klasyfikowania czy osoba jest pełnoletnia.
        10 Najlepszych modeli:
            # 1. Kombinacja: ('SEX', 'DEVICE', 'DEVICES_COUNT', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'), Dokładność: 0.8750
            # 2. Kombinacja: ('USE_MODS', 'DEVICE', 'DEVICES_COUNT', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'), Dokładność: 0.8750
            # 3. Kombinacja: ('USE_MODS', 'DEVICES_COUNT', 'PLATFORM', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'), Dokładność: 0.8750
            # 4. Kombinacja: ('USE_MODS', 'DEVICES_COUNT', 'PLAY_TIME', 'PLAY_TIME_OF_DAY', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'), Dokładność: 0.8750
            # 5. Kombinacja: ('USE_MODS', 'DEVICES_COUNT', 'PLAY_TIME', 'VOICE_CHAT_START', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'), Dokładność: 0.8750
            # 6. Kombinacja: ('DEVICE', 'DEVICES_COUNT', 'PLATFORM', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'), Dokładność: 0.8750
            # 7. Kombinacja: ('DEVICE', 'DEVICES_COUNT', 'PLAY_TIME', 'PLAY_TIME_OF_DAY', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'), Dokładność: 0.8750
            # 8. Kombinacja: ('DEVICE', 'DEVICES_COUNT', 'PLAY_TIME', 'VOICE_CHAT_START', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'), Dokładność: 0.8750
            # 9. Kombinacja: ('DEVICES_COUNT', 'PLATFORM', 'PLAY_TIME', 'PLAY_TIME_OF_DAY', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'), Dokładność: 0.8750
            # 10.Kombinacja: ('DEVICES_COUNT', 'PLATFORM', 'PLAY_TIME', 'VOICE_CHAT_START', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'), Dokładność: 0.8750
"""


def load_and_prepare_data(file_path):
    """
    Description:
        Wczytuje dane z pliku CSV i przygotowuje je do analizy.
        Zbiór danych jest uzupełniany o kolumnę klasyfikującą wiek pełnoletni lub niepełnoletni.

        >> data = pd.read_csv(file_path, sep=",")
            Args:
                - sep=",": Oznacza separator pomiędzy danymi, które są w tym przypadku odzielane przecinkami.

        >> bins = [11, 17, float('inf')]
            - Ustalamy granice przedziału wiekowego 11-17 i 18-nieskońconość.
        >> labels = [0, 1]
            - Ustawiamy odpowiednio do przedziału 0 lub 1.
        >> data['AGE_CLASS'] = pd.cut(data['AGE'], bins=bins, labels=labels, right=True)
            - Dodajemy kolumnę AGE_CLASS i przypisujemy mu wartości z tego przedziału.

    Args:
        - file_path (str): Ścieżka do pliku z danymi.

    Returns:
        - X (DataFrame): Dane wejściowe.
        - y (Series): Kolumna docelowa, czyli AGE_CLASS, 0 lub 1.
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
    X = data.drop(["AGE", "AGE_CLASS", 'DEVICE', 'DEVICES_COUNT', 'PLATFORM', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'], axis=1)
    # Test 2
    # X = data.drop(["AGE", "AGE_CLASS", 'SEX', 'DEVICE', 'DEVICES_COUNT', 'PLAY_TIME', 'FAV_GAME', 'VISUAL_IMPACT', 'FOLLOW_ESPORTS'], axis=1)
    y = data["AGE_CLASS"]

    print(data)
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
    plt.figure(figsize=(25, 10))
    tree.plot_tree(clf_tree, filled=True, feature_names=feature_names, class_names=["Kid", "Adult"], rounded=True, fontsize=9)
    plt.title("Pełnoletność graczy")
    # plt.savefig("decision_tree_gaming_plot.png", bbox_inches="tight")
    plt.show()

def visualize_correlationMatrix(df):
    """
        Description:
            Wizualizuje macierz korelacji.

        Args:
            - df (DataFrame) - dane do sprawdzenia korelacji np. X.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Macierz korelacji')
    # plt.savefig("correlationMatrix_Gaming_plot.png", bbox_inches="tight")
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
        sample_data[0][0] = {"M": 0, "K": 1, "I": 2}.get(sample_data[0][0].upper())

    sample_data_df = pd.DataFrame(sample_data, columns=feature_names)
    sample_data_scaled = scaler.transform(sample_data_df)
    predicted_class = clf_tree.predict(sample_data_scaled)

    return predicted_class

def main():
    """
        Description:
            Przykładowe wykorzystanie drzewa decyzyjnego na danych "Gaming_Dataset".
    """

    """
        Description:
            Podział na zbiory treningowe i testowe oraz skalowanie.
    """
    X, y = load_and_prepare_data("Gaming_Dataset.csv")

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
            Wizualizacja macierzy korelacji.
    """
    visualize_correlationMatrix(X)

    """
        Description:
            Przewidywanie dla przykładowych danych.
    """

    sample_data = [["K", 6, 1, 3, 1, 4, 0, 3, 4, 7, 4, 9]]
    predicted_class = predict_sample_data(clf_tree, scaler, sample_data, X.columns)
    print("Przykładowe dane:", sample_data)
    print("Przewidywana klasa:", predicted_class)

if __name__ == "__main__":
    """
        Description:
            Uruchomienie głównej funkcji main.
    """
    main()
