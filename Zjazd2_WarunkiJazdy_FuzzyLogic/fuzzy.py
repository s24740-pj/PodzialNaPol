import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
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
            4. matplotlib.pyplot

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
            Wykonanie obliczeń w systemie rozmytym oraz wizualizacja wyników.
    """
    ryzyko_sim.compute()
    ryzyko.view(sim=ryzyko_sim)
    plt.show()

    return ryzyko_sim.output['ryzyko']

print(ocena_ryzyka_jazdy(0, 100, 100))
