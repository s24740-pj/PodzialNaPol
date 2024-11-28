## Table of Contents
- [Zjazd 1 | Gra: Podzial na pol](#zjazd-1--gra-podzial-na-pol)
- [Zjazd 2 | FuzzyLogic: Ocena ryzyka jazdy + animacja (zastosowanie w praktyce)](#zjazd-2--fuzzylogic-ocena-ryzyka-jazdy--animacja-zastosowanie-w-praktyce)
- [Zjazd 3 | Clustering: Rekomendacja filmow](#zjazd-3--clustering-rekomendacja-filmow)
- [Zjazd 4 | Klasyfikacja](#zjazd-4--klasyfikacja)

---

# Zjazd 1 | Gra: Podzial na pol

**Folder:** `Zjazd1_PodzialNaPol`

## Wymagane biblioteki
1. math
2. matplotlib
3. easyAI
4. tabulate

## Zasady:
1. Gra zaczyna się od ustalonej liczby (np. 1000)
2. Gracze na zmianę dzielą aktualną liczbę przez **2**, **3** lub **4** (Wynik dzielenia jest zawsze zaokrąglany w dół)
3. Gracz, który nie może wykonać poprawnego podziału, gdy liczba wynosi **1**, przegrywa grę
4. Jeśli gracz wybierze dzielnik, który spowodowałby wynik równy **zero**, przegrywa
5. Gra trwa, dopóki jeden z graczy nie zmusi przeciwnika do sytuacji, w której nie można wykonać ruchu

## Przykład gry:
- Początkowa liczba: **100**
- Gracz 1 dzieli przez **4**: wynik to **25**
- Gracz 2 dzieli przez **3**: wynik to **8**
- Gra trwa, aż jeden z graczy nie będzie w stanie wykonać ruchu.

## Screenshoty z gry
### Gamplay
![Screenshot of the game](Zjazd1_PodzialNaPol/Screenshots/Gameplay.png)
### Historia gry
![Screenshot of the game](Zjazd1_PodzialNaPol/Screenshots/HistoryOutput.png)
### Wykres historii gry
![Screenshot of the game](Zjazd1_PodzialNaPol/Screenshots/HistoryGraph.png)

---

# Zjazd 2 | FuzzyLogic: Ocena ryzyka jazdy + animacja (zastosowanie w praktyce)

**Folder:** `Zjazd2_WarunkiJazdy_FuzzyLogic`

## Wymagane biblioteki
1. numpy
2. skfuzzy as fuzz
3. skfuzzy import control as ctrl
4. networkx
5. matplotlib.pyplot
6. pygame
7. sys

## Problem:
- Ocena ryzyka jazdy na podstawie widoczności, intensywności opadów i natężenia ruchu.
- Animacja przejazdu samochodu na podstawie oceny ryzyka jazdy.

## Screenshoty z FuzzyLogic
### Przykłady wystąpień ryzyka
![Screenshot of fuzzyLogic](Zjazd2_WarunkiJazdy_FuzzyLogic/Screenshots/ryzyko1.png)
![Screenshot of fuzzyLogic](Zjazd2_WarunkiJazdy_FuzzyLogic/Screenshots/ryzyko2.png)
![Screenshot of fuzzyLogic](Zjazd2_WarunkiJazdy_FuzzyLogic/Screenshots/ryzyko3.png)
### Przykłady wywołania systemu - zrzut ekranu
![Screenshot of fuzzyLogic](Zjazd2_WarunkiJazdy_FuzzyLogic/Screenshots/przykladowe_wywolanie1.png)
![Screenshot of fuzzyLogic](Zjazd2_WarunkiJazdy_FuzzyLogic/Screenshots/przykladowe_wywolanie2.png)
![Screenshot of fuzzyLogic](Zjazd2_WarunkiJazdy_FuzzyLogic/Screenshots/przykladowe_wywolanie3.png)

## Nagranie animacji
![Video of fuzzyLogic](Zjazd2_WarunkiJazdy_FuzzyLogic/Video/animacja_ryzyka_trzy_warianty.mp4)

---

# Zjazd 3 | Clustering: Rekomendacja filmow

**Folder:** `Zjazd3_Rekomendacja_filmów`

## Wymagane biblioteki
1. numpy
2. argparse
3. json
4. sklearn
5. scipy

## Problem:
System rekomendacji i antyrekomendacji filmów z wykorzystaniem klasteryzacji i metod podobieństw.

Moduł zawiera implementację systemu rekomendacji filmowych, który grupuje użytkowników na podstawie wspólnych filmów
i kategorii, a następnie generuje rekomendacje i antyrekomendacje dla wybranego użytkownika. System uwzględnia różne
metody obliczania podobieństwa między użytkownikami, takie jak odległość Euklidesowa i korelacja Pearsona, oraz
wykorzystuje algorytm k-średnich (k-means) do grupowania użytkowników.

## Screenshoty
### Rekomendacji
![Screenshot of recommendation](Zjazd3_Rekomendacja_filmów/Screenshots/rekomendacja1.png)
![Screenshot of recommendation](Zjazd3_Rekomendacja_filmów/Screenshots/rekomendacja2.png)
![Screenshot of recommendation](Zjazd3_Rekomendacja_filmów/Screenshots/rekomendacja3.png)

### Rekomendacji z opisem i datą wydania (Bonus)
Wykorzystanie API [TMDB](https://www.themoviedb.org)
```terminal
python rekomendacje.py "Dawid Feister" films_ratings.json --method euclidean --clusters 3 --api <api>
```
![Screenshot of recommendation](Zjazd3_Rekomendacja_filmów/Screenshots/opis_rekomendacji.png)

### Porównanie rekomendacji dwoma metrykami (Bonus)
Porównywanie dwóch metryk `uclidean` oraz `pearson`
```terminal
python rekomendacje.py "Dawid Feister" films_ratings.json --method euclidean --clusters 3 --compare
python rekomendacje.py "Paweł Czapiewski" films_ratings.json --method euclidean --clusters 3 --compare
```
![Screenshot of recommendation](Zjazd3_Rekomendacja_filmów/Screenshots/porownanie_rekomendacji.png)

---

# Zjazd 4 | Klasyfikacja

**Folder:** `Zjazd4_Klasyfikacja`

## Problem Boston House Price
Projekt polega na klasyfikacji mieszkań w Bostonie na podstawie cech:
- `CRIM`: Wskaźnik przestępczości na mieszkańca w danej miejscowości.
- `ZN`: Procent terenów mieszkalnych przeznaczonych na działki większe niż 25,000 stóp kwadratowych.
- `INDUS`: Procent gruntów przeznaczonych na przemysł nienależący do sektora detalicznego w danej miejscowości.
- `CHAS`: Zmienna binarna wskazująca, czy dany obszar graniczy z rzeką Charles (1 = tak, 0 = nie).
- `NOX`: Stężenie tlenków azotu (w częściach na 10 milionów).
- `RM`: Średnia liczba pokoi w mieszkaniach w danej miejscowości.
- `AGE`: Procent jednostek mieszkalnych posiadających właściciela, które zostały wybudowane przed 1940 rokiem.
- `DIS`: Wazona odległość do pięciu głównych ośrodków zatrudnienia w Bostonie.
- `RAD`: Indeks dostępności do dróg promieniowych.
- `TAX`: Stawka podatku od nieruchomości (wartość nieruchomości na $10,000).
- `PTRATIO`: Stosunek liczby uczniów do nauczycieli w danej miejscowości.
- `B`: Wzór: 1000 * (Bk - 0.63)^2, gdzie Bk to proporcja ludności czarnoskórej w danej miejscowości.
- `LSTAT`: Procent ludności o niższym statusie społecznym.
- `MEDV`: Mediana wartości domów właścicieli w tysiącach dolarów.

[Źródło](https://machinelearningmastery.com/standard-machine-learning-datasets/)

### Drzewo decyzyjne
Na podstawie mediany ceny mieszkań `MEDV` dane są klasyfikowane na dwie grupy: `niską (0)` i `wysoką (1)` cenę. `Drzewo decyzyjne` jest trenowane na przygotowanych danych i wykorzystywane do `przewidywania klasy dla nowych danych`. Skrypt zawiera także `wizualizację drzewa decyzyjnego` oraz `ocenę modelu`, w tym `dokładność klasyfikacji` oraz `raport z wynikami klasyfikacji`.

### Wizualizacja Drzewa Decyzyjnego
![decision_tree_plot_boston_house_price.png](Zjazd4_Klasyfikacja/Screenshots/decision_tree_plot_boston_house_price.png)
### Dokładność drzewa i przykładowe użycie
![decision_tree_terminal_boston_house_price.png](Zjazd4_Klasyfikacja/Screenshots/decision_tree_terminal_boston_house_price.png)

### SVM (SVC)
Na podstawie mediany ceny mieszkań `MEDV` dane są klasyfikowane na dwie grupy: `niską (0)` i `wysoką (1)` cenę. `SVM` jest trenowany na przygotowanych danych i wykorzystywany do `przewidywania klasy dla nowych danych`. Skrypt zawiera także `wizualizację SVC` oraz `ocenę modelu`, w tym `dokładność klasyfikacji` oraz `raport z wynikami klasyfikacji`.

### Wizualizacja SVM
![SVM_plot_boston_house_price.png](Zjazd4_Klasyfikacja/Screenshots/SVM_plot_boston_house_price.png)
### Dokładność drzewa i przykładowe użycie
![SVM_terminal__boston_house_price.png](Zjazd4_Klasyfikacja/Screenshots/SVM_terminal__boston_house_price.png)

## Problem przewidywania wieku Abalone
Projekt polega na `klasyfikacji wieku muszli abalonów` na podstawie różnych `cech fizycznych`, takich jak `wymiary`, `wagi` i `płeć muszli`.\
Wartość `Rings` reprezentuje `liczbę pierścieni` na muszli, co odpowiada jej wiekowi.\
Poniżej opis parametrów w bazie:
- `Sex`: Płeć muszli. Może przyjąć wartości M = 0 (mężczyzna), F = 1 (kobieta) lub I = 2 (niemowlę).
- `Length`: Długość muszli, mierzona w milimetrach (mm), będąca najdłuższym wymiarem muszli.
- `Diameter`: Średnica muszli, mierzona w milimetrach (mm), w kierunku prostopadłym do długości.
- `Height`: Wysokość muszli wraz z mięsem, mierzona w milimetrach (mm).
- `Whole weight`: Całkowita waga muszli, mierzona w gramach (g), obejmująca muszlę oraz mięso.
- `Shucked weight`: Waga samego mięsa, mierzona w gramach (g), po oddzieleniu od muszli.
- `Viscera weight`: Waga wnętrzności muszli, mierzona w gramach (g), po usunięciu wnętrzności.
- `Shell weight`: Waga muszli, mierzona w gramach (g), po wysuszeniu.
- `Rings`: Liczba pierścieni na muszli. To cecha, którą chcemy przewidywać.

[Źródło](https://archive.ics.uci.edu/dataset/1/abalone)

### Drzewo decyzyjne
Na podstawie `mediany` ilości pierścieni dane są `klasyfikowane` na dwie grupy: `Younger (0)` i `Older (1)`, definiując wiek Abalone.\
`Drzewo decyzyjne` jest trenowane na przygotowanych danych i wykorzystywane do przewidywania klasy dla nowych danych. \
Skrypt zawiera także `wizualizację` drzewa decyzyjnego oraz `ocenę modelu`, w tym `dokładność klasyfikacji` oraz `raport z wynikami klasyfikacji`.

Płeć w danych są przeliczane w postaci `"M": 0, "F": 1, "I": 2`.

### Wizualizacja Drzewa Decyzyjnego
![decision_tree_abalone_plot.png](Zjazd4_Klasyfikacja/Screenshots/decision_tree_plot_abalone.png)
### Dokładność Drzewa Decyzyjnego i przykładowe użycie
Dane mogą być podane w postaci:
```terminal
    [['0',0.39,0.31,0.1,0.406,0.1745,0.093,0.125]]
```
Lub
```terminal
    [['m',0.39,0.31,0.1,0.406,0.1745,0.093,0.125]]
```
![decision_tree_terminal_abalone.png](Zjazd4_Klasyfikacja/Screenshots/decision_tree_terminal_abalone.png)