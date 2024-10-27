import math
import matplotlib.pyplot as plt
from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
from tabulate import tabulate

class DivideByHalf(TwoPlayerGame):
    """
    Gra: Podział na pół | Divide by half

    Autorzy:
    Kamil Powierza
    Dawid Feister

    Zasady:
    1. Gra zaczyna się od ustalonej liczby (np. 1000)
    2. Gracze na zmianę dzielą aktualną liczbę przez 2, 3 lub 4 (Wynik dzielenia jest zawsze zaokrąglany w dół)
    3. Gracz, który nie może wykonać poprawnego podziału, gdy liczba wynosi 1, przegrywa grę
    4. Jeśli gracz wybierze dzielnik, który spowodowałby wynik równy zero, przegrywa
    5. Gra trwa, dopóki jeden z graczy nie zmusi przeciwnika do sytuacji, w której nie można wykonać ruchu
    """

    def __init__(self, players=None):
        """
        Description:
            Inicjalizuje nową grę "Podział na pół".
            Ustalamy początkową ilość punktów oraz ilość graczy grających.

        Parameters:
            players (list): Lista graczy. Default: None.
        """
        self.players = players
        self.start_points = 1000
        self.points = self.start_points
        self.current_player = 1

    def show_welcome_message(self):
        """
        Description:
            Wyświetla wprowadzenie do gry i jej zasady.
        """
        print('\n\t\t\tZapraszamy do gry "PODZIAŁ NA PÓŁ", w której zmierzysz się z AI jako swoim przeciwnikiem\n'
              f'{'*' * 130}\n'
              'Zasady:\n'
              f'1. Gra zaczyna się od ustalonej liczby (np. {self.start_points})\n'
              '2. Gracze na zmianę dzielą aktualną liczbę przez 2, 3 lub 4 (Wynik dzielenia jest zawsze zaokrąglany w dół)\n'
              '3. Gracz, który nie może wykonać poprawnego podziału, gdy liczba wynosi 1, przegrywa grę\n'
              '4. Jeśli gracz wybierze dzielnik, który spowodowałby wynik równy 0, przegrywa\n'
              '5. Gra trwa, dopóki jeden z graczy nie zmusi przeciwnika do sytuacji, w której nie można wykonać ruchu\n'
              f'{'*' * 130}\n')

    def possible_moves(self):
        """
        Description:
            Zwraca statyczną listę możliwych ruchów do wykonania. Zawiera ona dzielniki potrzebne do gry.

        Returns:
            possible_moves (list): Lista możliwych dzielników do wyboru.
        """
        return ['2', '3', '4']

    def make_move(self, move):
        """
        Description:
            Funkcja dzieli self.points przez wybrany dzielnik zaokrąglając go w dół, aktualizując przy tym stan self.points.

        Parameters:
            move (str): Wybrany dzielnik przez gracza.

        Returns:
            self.points (int): Aktualny stan punktacji.
        """
        self.points = math.floor(self.points / int(move))
        return self.points

    def win(self):
        """
        Description:
            Sprawdza bool czy aktualny wynik self.points jest równy 1, informując przy tym czy gra została zakończona/wygrana.

        Return:
            bool: True lub False w zależności od if self.points == 1.
        """
        return self.points == 1

    def is_over(self):
        """
        Description:
            Funkcja wywołuje metodę self.win(), która sprawdza czy gra została zakończona.

        Returns:
            bool: True lub False w zależności od self.win().
        """
        return self.win()

    def show(self):
        """
        Description:
            Pokazuje aktualny stan gry, wyświetlając bieżącą punktacje self.points.
        """
        print("Wynik: %d \n" % self.points)

    def scoring(self):
        """
        Description:
            Jest to punktacja przydatna dla AI, by ocenić jak blisko jest zwycięstwa. Zwracamy negatywną wartość ze względu na to że jeżeli mamy 100 punktów to AI myśli że jest daleko od zwycięstwa (1) i
            dostaje punktacje -100. Następna punktacja jeżeli dojdzie do sytuacji 50 punktów to AI dostaje większy scoring czyli -50. Jest to dla AI przydatna informacja aby ocenić jak blisko jest zwycięstwa.
            Im bliżej jest 0 tym jest bliżej wygranej.

        Returns:
            -self.points (int): Ujemna punktacja w postaci oceny bliskości wygranej dla AI.
        """
        return -self.points

    def show_history_table(self, history):
        """
        Description:
            Wyświetla historię gry w formie tabeli z wykorzystaniem tabulate.

        Parametry:
            history (list): Historia ruchów w grze w formacie [(gracz, wybór, wynik przed, wynik po)]
        """
        table_data = []
        for move in history:
            gracz = "Gracz 1" if move[0] == 1 else "Gracz AI"
            wybor = move[1]
            wynik_przed = move[2]
            wynik_po = move[3]
            table_data.append([gracz, wybor, wynik_przed, wynik_po])

        headers = ["Gracz", "Wybór gracza", "Wynik do podzielenia", "Wynik po podziale"]

        print(tabulate(table_data, headers, tablefmt="grid"))

    def plot_game_history(self, history):
        """
        Description:
            Rysuje histogram przedstawiający historię gry.

        Parameters:
            history (list): Historia ruchów w grze w formacie [(gracz, wybór, wynik przed, wynik po)]
        """
        rounds = list(range(len(history) + 1))
        scores = [self.start_points]

        for move in history:
            scores.append(move[3])

        plt.figure(figsize=(12, 6))

        for i in range(len(history)):
            plt.plot([i, i + 1], [scores[i], scores[i + 1]], color='red' if history[i][0] == 1 else 'green')

        for i in range(len(history)):
            plt.fill_between([i, i + 1], [scores[i], scores[i + 1]], color='red' if history[i][0] == 1 else 'green', alpha=0.3)

        plt.xlabel('Runda')
        plt.ylabel('Wynik')
        plt.title('Historia gry')
        plt.xticks(rounds)

        plt.yscale('log')
        ytick_values = [record[2] for record in history]
        divisors = [record[1] for record in history]
        ytick_labels = [f"{val} ({div})" for val, div in zip(ytick_values, divisors)]
        plt.yticks(ytick_values, labels=ytick_labels)

        plt.axhline(1, color='black', linewidth=0.8, linestyle='--')
        plt.grid()
        plt.legend(['Gracz 1', 'Gracz AI'], loc='upper right')
        plt.show()

    def play(self):
        """
        Description:
            Informacja o nazwie gry oraz zasady.

            Głowna pętla gry. Wykonuje działania dopóki gra nie jest skończona "while not self.is_over()" takie jak:
                * pokazuje aktualny stan gry
                    self.show()
                * sprawdza, który gracz wykonuje ruch, jeżeli inny niż 1 to AI
                    if self.current_player == 1
                        ...
                    else:
                        ...
                * pozwala wybrać graczowi dzielnik jeżeli takowy może wybrać
                    move = None
                    while move not in possible_moves:
                        move = input("Wybierz dzielnik: ")
                            if move not in possible_moves:
                                print("Błąd: %s nie jest poprawnym dzielnikiem." % move)
                                ...
                    ...
                * informuje o wybranym ruchu przez gracza
                    print("Gracz wybiera: %s \n" % move)
                * AI wybiera następny ruch jeżeli jest jego ruch oraz o nim informuje
                    if self.current_player == 1:
                        ...
                    else:
                        move = self.get_move()
                        print("AI wybiera: %s \n" % move)
                * Każdy wykonany ruch jest zapisywany do historii ruchów, przy okazji aktualizowany.
                    history = []
                    ...
                    history.append((self.current_player, int(move), int(self.points), self.play_move(move)))
                        Description:
                            self.play_move(move)
                                Description:
                                    Zmienia kolejność graczy po tym wskazuje na metodę self.make_move(move), w której jest aktualizowana punktacja
                                Parameters:
                                    move (int) - ruch jaki zostal wykonany, wykorzystywany w self.make_move(move) do aktualizacji punktacji
                                Returns:
                                    result (self.make_move(move)): Zwraca aktualną punktacje z metody self.make_move(move)
                        Format:
                            Gracz | Ruch/Dzielnik | Wynik przed dzieleniem | Wynik po dzieleniu
                        Example:
                            [(1, 2, 1000, 500), (2, 4, 500, 125), (1, 3, 125, 41), (2, 3, 41, 13), (1, 4, 13, 3), (2, 2, 3, 1)]
                        Description of example:
                            Pierwszy rekord mówi że gracz "1" (czyli człowiek) zagrał dzielnik "2" i z wyniku "1000" uzyskał "500"
                            Drugi rekord mówi że gracz "2" (czyli AI) zagrał dzielnik "4" i z wyniku "500" uzyskał "125"
                            Ostatni rekord mówi że gracz "2" (czyli AI) zagrał dzielnik "2" i z wyniku "3" uzyskał "1", czyli wygrał.

                * Sprawdza czy gra zakończyła się i kto wygrał
                    if self.points <= 1:
                        if self.current_player == 1:
                            print("Wynik 1, AI wygralo!")
                        else:
                            print("Wynik 1, Gratulacje udalo Ci sie pokonac AI!")

                * Wyświetla w formie tabelarycznej informację o przebiegu gry
                    self.show_history_table(history)

                * Wyświetla w formie wykresu historię przebiegu gry
                    self.plot_game_history(history)
        Returns:
            history (list): Historia wykonanych ruchów przez całą grę
        """
        self.show_welcome_message()
        history = []
        while not self.is_over():
            self.show()
            if self.current_player == 1:
                possible_moves = self.possible_moves()
                print("Możliwe dzielniki: %s" % possible_moves)
                move = None

                while move not in possible_moves:
                    move = input("Wybierz dzielnik: ")
                    if move not in possible_moves:
                        print("Błąd: %s nie jest poprawnym dzielnikiem." % move)
                        print("Możliwe dzielniki: %s" % possible_moves)
                print("Gracz 1 wybiera: %s \n" % move)
            else:
                move = self.get_move()
                print("AI wybiera: %s \n" % move)

            history.append((self.current_player, int(move), int(self.points), self.play_move(move)))

            if self.points <= 1:
                if self.current_player == 1:
                    print("Wynik 1, AI wygralo!")
                else:
                    print("Wynik 1, Gratulacje udalo Ci sie pokonac AI!")

        self.show_history_table(history)
        self.plot_game_history(history)
        return history
"""
    ai = Negamax(depth)
        Description:
            Inicjalizuje AI z użyciem algorytmu Negamax. Jest to strategia oparta na algorytmie minimax. 
        Parameters:
            depth (int) - oznacza głebokość przesukiwania czyli maksymalną ilość ruchów, z którą myśli do przodu. Im więcej tym cięższy jest do pokonania, co za czym idzie więcej obliczeń potrzebuje.
        Example:
            ai = Negamax(15)
"""
ai = Negamax(15)
"""
    game = DivideByHalf(TwoPlayerGame)
        Description:
            Tworzona jest instancja DivideByHalf z wyborem graczy.
        Parameters:
            TwoPlayerGame (list): lista dwóch graczy, którzy biorą udział w grze.
        Example:
            game = DivideByHalf([Human_Player(), AI_Player(ai)])
                Parameters:
                    Human_Player() - Gracz jest człowiekiem
                    AI_Player(ai) - Gracz jest AI
"""
game = DivideByHalf([Human_Player(), AI_Player(ai)])
"""
    history = game.play()
        Description:
            Zaczynamy grę poprzez wywołanie funkcji game.play().
            Opcjonalnie możemy do niej przypisać zmienną, na przykład "history". Zwraca do niej całą naszą historię gry.
        Example:
            print(history) #output [(1, 2, 1000, 500), (2, 4, 500, 125), (1, 3, 125, 41), (2, 3, 41, 13), (1, 4, 13, 3), (2, 2, 3, 1)]
            Format:
                Gracz | Ruch/Dzielnik | Wynik przed dzieleniem | Wynik po dzieleniu
"""
history = game.play()
print(history)