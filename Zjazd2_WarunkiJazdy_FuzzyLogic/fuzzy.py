import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import pygame
import sys
from skfuzzy import control as ctrl

def ocena_ryzyka_jazdy(widocznosc_v, intensywnosc_opadow_v, natezenie_ruchu_v):
    """
        Autorzy:
            Kamil Powierza
            Dawid Feister

        Wymagane biblioteki:
            1. numpy
            2. skfuzzy as fuzz
            3. skfuzzy import control as ctrl
            4. networkx
            5. matplotlib.pyplot
            6. pygame
            7. sys

        Description:
            Ocena ryzyka jazdy na podstawie widoczności, intensywności opadów i natężenia ruchu.

        Parametry:
            widocznosc (float): Wartość widoczności w zakresie od 0 do 100.
                0 - Najgorsza widoczność.
                100 - Najlepsza widoczność.
            intensywnosc_opadow (float): Wartość intensywności opadów w zakresie od 0 do 100.
                0 - Brak opadów.
                100 - Intensywne opady.
            natezenie_ruchu (float): Wartość natężenia ruchu w zakresie od 0 do 100.
                0 - Brak natężenia ruchu, puste drogi.
                100 - Duże natężenie ruchu, wiele aut na drodze.

        Zwraca:
            float: Poziom ryzyka jazdy w zakresie od 0 do 100.
                0 - Małe ryzyko wypadku.
                100 - Duże ryzyko wypadku.
    """
    """
        Description:
            Tworzenie zmiennych rozmytych [Antecedents] oraz zmiennej wyjściowej [Consequent] 
            dla systemu rozmytego, które będą wykorzystywane w ocenie ryzyka.
    """
    widocznosc = ctrl.Antecedent(np.arange(0, 101, 0.1), 'widocznosc')
    intensywnosc_opadow = ctrl.Antecedent(np.arange(0, 101, 0.1), 'intensywnosc_opadow')
    natezenie_ruchu = ctrl.Antecedent(np.arange(0, 101, 0.1), 'natezenie_ruchu')
    ryzyko = ctrl.Consequent(np.arange(0, 101, 0.1), 'ryzyko')
    """
        Description:
            Użycie funkcji automf do automatycznego przypisania trzech funkcji przynależności 
            [poor, average, good] dla zmiennych wejściowych.
    """
    intensywnosc_opadow.automf(3)
    widocznosc.automf(3)
    natezenie_ruchu.automf(3)
    """
        Description:
            Tworzenie funkcji przynależności dla zmiennej wyjściowej 'ryzyko' 
            przy użyciu funkcji trójkątnej (trimf) dla trzech poziomów ryzyka: niski, średni, wysoki.
    """
    ryzyko['poor'] = fuzz.trimf(ryzyko.universe, [0, 0, 50])
    ryzyko['average'] = fuzz.trimf(ryzyko.universe, [0, 50, 100])
    ryzyko['good'] = fuzz.trimf(ryzyko.universe, [50, 100, 100])
    """
        Description:
            Ustalanie reguł rozmytych, które definiują, jak poziomy zmiennych wejściowych 
            wpływają na poziom ryzyka. 
    """
    rule1 = ctrl.Rule(widocznosc['good'] | intensywnosc_opadow['poor'] | natezenie_ruchu['poor'], ryzyko['poor'])
    rule2 = ctrl.Rule(widocznosc['average'] | intensywnosc_opadow['average'] | natezenie_ruchu['average'], ryzyko['average'])
    rule3 = ctrl.Rule(widocznosc['poor'] | intensywnosc_opadow['good'] | natezenie_ruchu['good'], ryzyko['good'])
    """
        Description:
            Inicjalizacja systemu kontrolnego zdefiniowanego wcześniej regułami.
    """
    ryzyko_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    ryzyko_sim = ctrl.ControlSystemSimulation(ryzyko_ctrl)
    """
        Description:
            Ustawienie wartości wejściowych dla zmiennych rozmytych na podstawie przekazanych 
            parametrów funkcji.
    """
    ryzyko_sim.input['widocznosc'] = widocznosc_v
    ryzyko_sim.input['intensywnosc_opadow'] = intensywnosc_opadow_v
    ryzyko_sim.input['natezenie_ruchu'] = natezenie_ruchu_v
    """
        Description:
            Wykonanie obliczeń w systemie rozmytym, wyświetlenie wyniku oraz wizualizacja wyników.
    """
    ryzyko_sim.compute()
    ryzyko.view(sim=ryzyko_sim)
    plt.show()

    print(f"Wynik ryzyka jazdy: {ryzyko_sim.output['ryzyko']:.2f}")
    """
        Zwraca:
            float: Obliczona wartość ryzyka jazdy w zakresie od 0 do 100, na podstawie
            symulacji logiki rozmytej. Im wyższa wartość, tym większe ryzyko jazdy:
                0 - Bardzo niskie ryzyko.
                100 - Bardzo wysokie ryzyko.
    """
    return ryzyko_sim.output['ryzyko']

def animacja_car(widocznosc, intensywnosc_opadow, natezenie_ruchu):
    """
        Description:
            Funkcja animuje przejazd samochodu na podstawie oceny ryzyka jazdy.

        Parametry:
            widocznosc (float): Wartość widoczności.
            intensywnosc_opadow (float): Wartość intensywności opadów.
            natezenie_ruchu (float): Wartość natężenia ruchu.
    """
    ryzyko = ocena_ryzyka_jazdy(widocznosc, intensywnosc_opadow, natezenie_ruchu)
    """
        Description:
            Ustalenie czasu przejazdu oraz koloru samochodu w zależności od poziomu ryzyka:
            -dla dobrych warunków, pojazd ma kolor zielony i pokonuje trasę w 3 sekundy, 
            -dla umiarkowanych warunków, pojazd ma kolor pomarańczowy i pokonuje trasę w 6 sekund,
            -dla trudnych warunków, pojazd przyjmuje kolor czerwony i pokojue trasę w 9 sekund.
    """
    if ryzyko < 33:
        czas_przejazdu = 3000
        car_color = (0, 255, 0)
    elif ryzyko < 66:
        czas_przejazdu = 6000
        car_color = (255, 165, 0)
    else:
        czas_przejazdu = 9000
        car_color = (255, 0, 0)
    """
        Description:
            Funkcja inicjalizuje Pygame, ustawia rozmiar okna na 860x150 pikseli,
            ustawia tytuł okna na 'Animacja samochodu' oraz tworzy obiekt zegara,
            który będzie kontrolował liczbę klatek na sekundę w animacji.
    """
    pygame.init()
    screen = pygame.display.set_mode((860, 150))
    pygame.display.set_caption('Animacja samochodu')
    clock = pygame.time.Clock()
    """
        Description:
            Ustala położenie samochodu oraz jego wymiary, w tym szerokość i wysokość
            nadwozia, a także promień kół. Parametry te są wykorzystywane do rysowania
            samochodu podczas animacji.

        Parametry:
            car_x (int): Położenie samochodu na osi X (początkowo ustawione na 0).
            car_width (int): Szerokość samochodu w pikselach.
            car_height (int): Wysokość samochodu w pikselach.
            wheel_radius (int): Promień kół samochodu w pikselach.            
    """
    car_x = 0
    car_width = 60
    car_height = 30
    wheel_radius = 8
    """
        Description:
            Pętla animacyjna, która jest odpowiedzialna za rysowanie samochodu na ekranie oraz 
            aktualizowanie jego stanu w czasie rzeczywistym. Oblicza czas animacji, 
            monitorując, jak długo samochód przemieszcza się na osi X, 
            aż osiągnie określoną pozycję (800 pikseli).

        Zawartość:
            - Inicjalizacja zmiennej `start_ticks` do śledzenia czasu rozpoczęcia animacji.
            - Wykonywanie się pętli, dopóki położenie samochodu na osi X jest mniejsze niż 800,
              co oznacza, że samochód jeszcze nie dotarł do końca drogi.

        Zwraca:
            None: Funkcja działa w trybie animacji, nie zwraca wartości, lecz wykonuje 
            animację w czasie rzeczywistym.
    """
    start_ticks = pygame.time.get_ticks()
    while car_x < 800:
        """
            Description:
                Obsługuje zdarzenia Pygame w trakcie animacji, umożliwiając interakcję 
                użytkownika z aplikacją. Sprawdza, czy wystąpiło zdarzenie zamknięcia 
                okna, co pozwala na bezpieczne zakończenie aplikacji.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        """
            Description:
                Oblicza czas, jaki upłynął od rozpoczęcia animacji, i na jego podstawie aktualizuje
                położenie samochodu na osi x. Umożliwia to płynne przesuwanie 
                samochodu wzdłuż ekranu w czasie rzeczywistym.
        """
        elapsed_time = pygame.time.get_ticks() - start_ticks
        if elapsed_time < czas_przejazdu:
            car_x = (elapsed_time / czas_przejazdu) * 800
        """
            Description:
                Funkcja wypełnia tło kolorem, rysuje drogę, samochód oraz jego szczegóły.
                Wyświetla także nagłówek z informacją o czasie przejazdu i rysuje granice okna.
                Wszystkie elementy są dodawane do ekranu przy użyciu funkcji Pygame, co tworzy
                animację ruchu samochodu.

            Elementy:
                - Wypełnienie ekranu tłem o kolorze zielonym.
                - Rysowanie szarego prostokąta, symbolizującego drogę.
                - Wyświetlanie tekstu informacyjnego o czasie przejazdu w górnej części ekranu.
                - Rysowanie granicznych linii po bokach ekranu.
                - Rysowanie samochodu, jego kół, lamp oraz okien w zadanym położeniu.
        """
        screen.fill((41, 138, 64))

        pygame.draw.rect(screen, (128, 128, 128), (0, 80, 860, 30))

        font = pygame.font.SysFont('Arial', 20)
        naglowek_text = f'Samochód przejechał dystans w {czas_przejazdu / 1000} sekund. Ryzyko wyniosło: {round(ryzyko, 2)}'
        naglowek_surface = font.render(naglowek_text, True, (0, 0, 0))
        naglowek_rect = naglowek_surface.get_rect(center=(425, 20))
        screen.blit(naglowek_surface, naglowek_rect)

        pygame.draw.line(screen, (0, 0, 0), (0, 0), (0, 150), 5)
        pygame.draw.line(screen, (0, 0, 0), (860, 0), (860, 150), 5)

        pygame.draw.rect(screen, car_color, (car_x, 100 - car_height, car_width, car_height))

        pygame.draw.circle(screen, (0, 0, 0), (int(car_x + 15), 100), wheel_radius)
        pygame.draw.circle(screen, (0, 0, 0), (int(car_x + car_width - 14), 100), wheel_radius)

        pygame.draw.rect(screen, (255, 255, 0), (car_x + car_width - 1, 105 - (car_height / 2), 5, 5))

        pygame.draw.rect(screen, (0, 0, 255), (car_x + 35, 95 - (car_height / 1.5), 17, 12))
        """
            Description:
                Odświeża zawartość ekranu, aby wyświetlić nowe położenie i stan elementów
                animacji. Używa funkcji `pygame.display.flip()`, aby zaktualizować ekran, 
                a następnie `clock.tick(60)` do ograniczenia liczby klatek na sekundę do 60, 
                co zapewnia płynność animacji.
        """
        pygame.display.flip()
        clock.tick(60)
    """
        Description:
            Wywołanie funkcji `pygame.quit()` kończy działanie Pygame i zwalnia wszystkie
            zasoby, zamykając okno aplikacji. Jest to ostatni krok w zamknięciu aplikacji,
            gdy animacja została zakończona.
    """
    pygame.quit()
    """
        Uruchomienie programu do oceny ryzyka jazdy i animacji samochodu.

        Description:
            Blok główny programu, który pobiera wartości widoczności, intensywności opadów 
            oraz natężenia ruchu od użytkownika, sprawdzanie poprawności wartości wejściowych,
            a następnie wywołuje funkcję `animacja_car()` z tymi danymi. Umożliwia to obliczenie
            ryzyka jazdy oraz animację przedstawiającą przejazd samochodu na podstawie podanych
            parametrów.

        Parametry:
            Brak (pobierane bezpośrednio od użytkownika przez `input`).
    """
if __name__ == "__main__":
    while True:
        try:
            widocznosc_input = float(input("Podaj widoczność (0[niska] - 100[wysoka]): "))
            intensywnosc_opadow_input = float(input("Podaj intensywność opadów (0[brak] - 100[wysoka]): "))
            natezenie_ruchu_input = float(input("Podaj natężenie ruchu (0[niskie] - 100[wysokie]): "))
            animacja_car(widocznosc_input, intensywnosc_opadow_input, natezenie_ruchu_input)
            break
        except ValueError:
            print("Wprowadzono błędne dane. Upewnij się, że podajesz liczby od 0 do 100.")
