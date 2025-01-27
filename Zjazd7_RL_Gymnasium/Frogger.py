import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecVideoRecorder
import time
"""
    Autorzy:
        Kamil Powierza
        Dawid Feister

    Wymagane biblioteki:
        1. gymnasium
        2. ale_py
        3. stable_baselines3
        4. time

    Description:
        Program implementuje trening agenta DQN (Deep Q-Network) w środowisku Frogger. 
        Agent jest uczony za pomocą biblioteki Stable Baselines3, a jego wydajność jest wizualizowana
        w postaci wideo, które rejestruje działanie agenta w grze. Program obejmuje przygotowanie środowiska,
        konfigurację modelu, proces treningowy, oraz testowanie modelu na środowisku renderowanym.
"""
"""
    Description:
        Inicjalizacja środowiska gry Frogger za pomocą Gymnasium. Środowisko jest opakowane
        w `DummyVecEnv` w celu umożliwienia kompatybilności z biblioteką Stable Baselines3.
        Następnie stosowane jest `VecFrameStack`, które łączy kolejne ramki w stos
        (używane w algorytmach opartych na wizji komputerowej).

    Args:
        n_stack (int): Liczba ramek do połączenia w stos (domyślnie 4).

    Returns:
        train_env (VecFrameStack): Przygotowane środowisko do treningu agenta.
"""
train_env = gym.make("ALE/Frogger-v5")
train_env = DummyVecEnv([lambda: train_env])
train_env = VecFrameStack(train_env, n_stack=4)

"""
    Description:
        Inicjalizacja modelu DQN z użyciem Stable Baselines3. Model korzysta z polityki CNN,
        co pozwala przetwarzać dane obrazowe (ramki gry). Zdefiniowane parametry wpływają na
        proces treningu, takie jak szybkość uczenia się, wielkość bufora doświadczeń,
        oraz harmonogram eksploracji.

    Args:
        policy (str): Polityka modelu (np. "CnnPolicy").
        learning_rate (float): Szybkość uczenia się.
        buffer_size (int): Rozmiar bufora doświadczeń.
        exploration_fraction (float): Procent czasu treningu na eksplorację.
        exploration_final_eps (float): Minimalna wartość epsilon w eksploracji.
        target_update_interval (int): Liczba kroków pomiędzy aktualizacjami celu.
        tensorboard_log (str): Ścieżka do logów TensorBoard.

    Returns:
        model (DQN): Zainicjalizowany model DQN.
"""
model = DQN("CnnPolicy", train_env, verbose=1, learning_rate=0.0001, buffer_size=20000,
            exploration_fraction=0.1, exploration_final_eps=0.02, target_update_interval=500,
            tensorboard_log="frogger_dqn")

"""
    Description:
        Rozpoczęcie procesu treningu agenta DQN. Model uczy się na podstawie środowiska,
        zapisując wyniki logowania w TensorBoard.

    Args:
        total_timesteps (int): Całkowita liczba kroków treningowych.
        log_interval (int): Interwał, co ile kroków zapisywać logi.

    Returns:
        None
"""
model.learn(total_timesteps=250000, log_interval=100)
model.save("frogger_dqn_model")

"""
    Description:
        Inicjalizacja środowiska testowego Frogger, które umożliwia wizualizację
        działań agenta w czasie rzeczywistym. Środowisko jest renderowane
        w trybie "rgb_array", co pozwala na zapis wideo.

    Args:
        render_mode (str): Tryb renderowania środowiska ("rgb_array").
        frameskip (int): Liczba ramek pomijanych pomiędzy krokami.

    Returns:
        test_env (VecFrameStack): Przygotowane środowisko do testowania agenta.
"""
test_env = gym.make("ALE/Frogger-v5", render_mode="rgb_array", frameskip=1)
test_env = DummyVecEnv([lambda: test_env])
test_env = VecFrameStack(test_env, n_stack=4)

"""
    Description:
        Rejestracja wideo z działań agenta w środowisku testowym. `VecVideoRecorder` jest używane
        do zapisu wideo z określonymi parametrami, takimi jak długość wideo oraz warunki rozpoczęcia nagrywania.

    Args:
        video_env (VecVideoRecorder): Środowisko testowe opakowane w rejestrator wideo.
        video_length (int): Maksymalna liczba klatek wideo.
        record_video_trigger (callable): Funkcja określająca warunki rozpoczęcia nagrywania.

    Returns:
        None
"""
video_env = VecVideoRecorder(
    test_env,
    "videos/",
    record_video_trigger=lambda x: x == 0,
    video_length=60 * 60 * 5
)

obs = video_env.reset()
done = False
start_time = time.time()

"""
    Description:
        Pętla testowa, która wykonuje kroki w środowisku na podstawie akcji przewidywanych przez model.
        Test trwa przez określony czas lub do zakończenia gry.

    Args:
        obs: Obserwacja ze środowiska.
        action: Działanie przewidywane przez model.
        reward: Nagroda otrzymana po wykonaniu działania.
        done: Flaga oznaczająca zakończenie epizodu.
        info: Dodatkowe informacje o środowisku.

    Returns:
        None
"""
while time.time() - start_time < 600:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = video_env.step(action)
    if done:
        obs = video_env.reset()

"""
    Description:
        Zamknięcie wszystkich środowisk w celu zwolnienia zasobów po zakończeniu treningu i testowania.

    Returns:
        None
"""
video_env.close()
test_env.close()
train_env.close()