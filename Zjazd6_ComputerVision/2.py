import cv2 as cv
import numpy as np

"""
    Autorzy:
        Kamil Powierza
        Dawid Feister

    Wymagane biblioteki:
        1. cv2
        2. numpy

    Description:
        Główna pętla programu, która odczytuje klatki z kamery, konwertuje je na przestrzeń kolorów HSV
        i analizuje obecność kolorów zielonego i czerwonego, rysując odpowiednie wskazówki na obrazie w postaci celowników.
    
"""
"""
    Description: 
        Uruchomienie kamery
"""
cap = cv.VideoCapture(0)

while True:
    """
        Description: 
            Odczytanie jednej klatki z kamery.
    """
    _, frame = cap.read()

    """
        Description:
            Konwersja obrazu z przestrzeni kolorów BGR na HSV.
    """
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    """
        Description:
            Definiowanie zakresu kolorów dla zielonego koloru w przestrzeni HSV.
    """
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([80, 255, 255])

    """
        Description:
            Definiowanie zakresu kolorów dla czerwonego koloru w przestrzeni HSV (dwa zakresy, aby uwzględnić pełny zakres czerwieni).
    """
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 100])
    red_upper2 = np.array([180, 255, 255])

    """
        Description:
            Tworzenie masek dla zielonego i czerwonego koloru w obrazie.
    """
    green_mask = cv.inRange(hsv_frame, green_lower, green_upper)
    red_mask1 = cv.inRange(hsv_frame, red_lower1, red_upper1)
    red_mask2 = cv.inRange(hsv_frame, red_lower2, red_upper2)

    """
        Description:
            Łączenie dwóch masek czerwonego koloru w jedną.
    """
    red_mask = cv.bitwise_or(red_mask1, red_mask2)

    """
        Description:
            Znajdowanie konturów na masce zielonego koloru w celu odnajdywania gdzie narysować celownik.
    """
    contours, _ = cv.findContours(green_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    """
        Description:
            Iteracja po wykrytych konturach.
    """
    for contour in contours:
        """
            Jeśli kontur jest wystarczająco duży (ma minimalny rozmiar), obliczamy jego środek
            i rysujemy większy celownik w miejscu wykrycia zielonego koloru.
            
            100 to minimalny rozmiar obszaru.
        """
        if cv.contourArea(contour) > 100:
            """
                Description:
                    Obliczanie momentów konturu, które pozwolą znaleźć środek konturu.
            """
            M = cv.moments(contour)
            if M["m00"] > 0:
                """
                    Description:
                        Obliczanie współrzędnych środka konturu.
                        
                        cx - Współrzędne X środka.
                        cy - Współrzędne Y środka.
                        
                        Moment geometryczny (m00) to suma wartości pikseli wewnątrz konturu.
                        W przypadku masek binarnych (czarno-białych) oznacza liczbę białych pikseli w konturze (obszar).
                        
                        m10 i m01 reprezentują sumy ważone współrzędnych X i Y.
                        
                        Aby obliczyć środek musimy sume ważoną współrzędnej podzielić na sume wartości pikseli wewnątrz konturu
                """
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                """
                    Description:
                        Rysowanie celownika na środku konturu.
                """
                """
                    Description:
                        Rysowanie okręgu.
                """
                cv.circle(frame, (cx, cy), 20, (0, 255, 0), 3)
                """
                    Description:
                        Rysowanie linii poziomej.
                """
                cv.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 3)
                """
                    Description:
                        Rysowanie linii pionowej.
                """
                cv.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 3)

                """
                    Description:
                        Dodanie tekstu na obrazie, że wykryto zielony kolor.
                """
                cv.putText(frame, "Zielony - Strzelam!", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    """
        Description:
            Sprawdzanie, czy wykryto wystarczająco dużo czerwonego koloru na obrazie.
    """
    if cv.countNonZero(red_mask) > 100:
        """
            Description:
                Jeśli na obrazie znajduje się wystarczająca ilość czerwonego koloru,
                dodajemy tekst, że nie strzelamy.
        """
        cv.putText(frame, "Czerwony - Nie strzelam!", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    """
        Description:
            Wyświetlanie wyników w postaci obrazu z kamery oraz masek dla zielonego i czerwonego koloru.
    """
    cv.imshow("frame", frame)
    cv.imshow("Green Mask", green_mask)
    cv.imshow("Red Mask", red_mask)

    """
        Description:
            Zakończenie programu po naciśnięciu klawisza ESC.
    """
    if cv.waitKey(1) == 27:
        break

"""
    Description:
        Zwolnienie zasobów kamery i zamknięcie wszystkich okien.
"""
cap.release()
cv.destroyAllWindows()