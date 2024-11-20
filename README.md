## Table of Contents
- [Gra: Podzial na pol](#gra-podzial-na-pol)
- [FuzzyLogic: Ocena ryzyka jazdy + animacja (zastosowanie w praktyce)](#fuzzylogic-ocena-ryzyka-jazdy--animacja-zastosowanie-w-praktyce)
- [Clustering: Rekomendacja filmow](#clustering-rekomendacja-filmow)

---

## Gra: Podzial na pol
## Zjazd 1

**Folder:** `Zjazd1_PodzialNaPol`

### Wymagane biblioteki
1. math
2. matplotlib
3. easyAI
4. tabulate

### Zasady:
1. Gra zaczyna się od ustalonej liczby (np. 1000)
2. Gracze na zmianę dzielą aktualną liczbę przez **2**, **3** lub **4** (Wynik dzielenia jest zawsze zaokrąglany w dół)
3. Gracz, który nie może wykonać poprawnego podziału, gdy liczba wynosi **1**, przegrywa grę
4. Jeśli gracz wybierze dzielnik, który spowodowałby wynik równy **zero**, przegrywa
5. Gra trwa, dopóki jeden z graczy nie zmusi przeciwnika do sytuacji, w której nie można wykonać ruchu

### Przykład gry:
- Początkowa liczba: **100**
- Gracz 1 dzieli przez **4**: wynik to **25**
- Gracz 2 dzieli przez **3**: wynik to **8**
- Gra trwa, aż jeden z graczy nie będzie w stanie wykonać ruchu.

## Screenshot z gry
### Gamplay
![Screenshot of the game](Zjazd1_PodzialNaPol/Screenshots/Gameplay.png)
### Historia gry
![Screenshot of the game](Zjazd1_PodzialNaPol/Screenshots/HistoryOutput.png)
### Wykres historii gry
![Screenshot of the game](Zjazd1_PodzialNaPol/Screenshots/HistoryGraph.png)

---

## FuzzyLogic: Ocena ryzyka jazdy + animacja (zastosowanie w praktyce)
## Zjazd 2

**Folder:** `Zjazd2_WarunkiJazdy_FuzzyLogic`

### Wymagane biblioteki
1. numpy
2. skfuzzy as fuzz
3. skfuzzy import control as ctrl
4. networkx
5. matplotlib.pyplot
6. pygame
7. sys

### Problem:
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

## Clustering: Rekomendacja filmow
## Zjazd 3

**Folder:** `Zjazd3_Rekomendacja_filmów`

### Wymagane biblioteki
1. numpy
2. argparse
3. json
4. sklearn
5. scipy

### Problem:
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
![Screenshot of recommendation](Zjazd3_Rekomendacja_filmów/Screenshots/opis_rekomendacji.png)