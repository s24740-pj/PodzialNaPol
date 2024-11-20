import json
import numpy as np
import argparse
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import requests
"""
Problem:
--------
    System rekomendacji i antyrekomendacji filmów z wykorzystaniem klasteryzacji i metod podobieństw.

    Moduł zawiera implementację systemu rekomendacji filmowych, który grupuje użytkowników na podstawie wspólnych filmów
    i kategorii, a następnie generuje rekomendacje i antyrekomendacje dla wybranego użytkownika. System uwzględnia różne
    metody obliczania podobieństwa między użytkownikami, takie jak odległość Euklidesowa i korelacja Pearsona, oraz
    wykorzystuje algorytm k-średnich (k-means) do grupowania użytkowników.

Funkcje modułu:
---------------
    - load_data(filename): Wczytuje dane użytkowników z pliku JSON.
    - euclidean_similarity(user1, user2, ratings): Oblicza podobieństwo Euklidesowe między użytkownikami na podstawie ocen wspólnych filmów.
    - pearson_similarity(user1, user2, ratings): Oblicza współczynnik korelacji Pearsona między użytkownikami na podstawie wspólnych filmów.
    - weighted_similarity(user1, user2, ratings, method): Oblicza podobieństwo z uwzględnieniem wagi wynikającej z liczby wspólnych filmów.
    - cluster_users(ratings, n_clusters): Grupuje użytkowników w klastry na podstawie oglądanych filmów i kategorii za pomocą algorytmu k-means.
    - generate_recommendations_with_similarity(user, ratings, similarity_func, user_clusters):
        Generuje rekomendacje i antyrekomendacje dla użytkownika na podstawie analizy klastrów i podobieństwa.
        Funkcja zawiera logikę awaryjną:
            - Obniża próg ocen wymaganych dla rekomendacji (np. z 8 do 7) i anty rekomendacji (np. z 3 do 4), jeśli początkowo nie uda się znaleźć wystarczającej liczby wyników.
            - Jeśli w obrębie bieżącego klastra nadal nie ma wystarczających wyników, funkcja przechodzi do kolejnego najbliższego klastra, kontynuując poszukiwania.
    - get_movie_info(movie_name): Wyszukuje w bazie TMDB <themoviedb.org>, filmu na podstawie nazwy, pełny tytuł, data wydania i skrótowy opis w języku polskim oraz angielskim.
    - main(): Obsługuje wczytywanie danych, konfigurację i uruchamianie systemu rekomendacji z linii poleceń.

Autorzy:
--------
    - Kamil Powierza
    - Dawid Feister

Wymagane biblioteki:
--------------------
    - numpy
    - argparse
    - json
    - sklearn
    - scipy
    - requests

Instrukcja użycia:
------------------
    1. Przygotuj plik JSON z danymi użytkowników w formacie:
        {
            "user1": {
                "movie1": {"rating": 9, "categories": ["akcja", "thriller"]},
                "movie2": {"rating": 6, "categories": ["dramat"]}
            },
            "user2": {
                "movie1": {"rating": 8, "categories": ["akcja", "thriller"]},
                "movie3": {"rating": 5, "categories": ["komedia"]}
            }
        }
    2. Uruchom skrypt z linii poleceń:
        python rekomendacje.py <username> <filename> [--method <similarity_method>] [--clusters <n_clusters>]
        - `username`: Użytkownik, dla którego generujemy rekomendacje.
        - `filename`: Plik JSON z danymi.
        - `--method`: Opcjonalnie metoda obliczania podobieństwa ('euclidean' lub 'pearson').
        - `--clusters`: Opcjonalnie liczba klastrów (domyślnie 3).

Zwraca:
-------
    - 5 rekomendacji filmów (filmy ocenione wysoko przez podobnych użytkowników, ale nieoglądane przez wybranego użytkownika, wraz z jego opisem i datą wydania z TMDB).
    - 5 antyrekomendacji filmów (filmy ocenione nisko przez podobnych użytkowników, ale nieoglądane przez wybranego użytkownika, wraz z jego opisem i datą wydania z TMDB).

Przykład:
---------
    python rekomendacje.py "Dawid Feister" films_ratings.json --method euclidean --clusters 3

"""
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def euclidean_similarity(user1, user2, ratings):
    common_movies = set(ratings[user1].keys()) & set(ratings[user2].keys())
    if not common_movies:
        return 0
    user1_ratings = [ratings[user1][movie]["rating"] for movie in common_movies]
    user2_ratings = [ratings[user2][movie]["rating"] for movie in common_movies]
    return 1 / (1 + np.linalg.norm(np.array(user1_ratings) - np.array(user2_ratings)))


def pearson_similarity(user1, user2, ratings):
    common_movies = set(ratings[user1].keys()) & set(ratings[user2].keys())
    if len(common_movies) < 2:
        return 0
    user1_ratings = [ratings[user1][movie]["rating"] for movie in common_movies]
    user2_ratings = [ratings[user2][movie]["rating"] for movie in common_movies]

    if len(set(user1_ratings)) == 1 and len(set(user2_ratings)) == 1:
        return 0

    corr, _ = pearsonr(user1_ratings, user2_ratings)
    return corr if not np.isnan(corr) else 0


def weighted_similarity(user1, user2, ratings, method):
    # Wybór odpowiedniej funkcji podobieństwa
    if method == 'pearson':
        similarity = pearson_similarity(user1, user2, ratings)
    elif method == 'euclidean':
        similarity = euclidean_similarity(user1, user2, ratings)
    else:
        raise ValueError("Nieznana metoda: wybierz 'pearson' lub 'euclidean'.")

    common_movies = set(ratings[user1].keys()) & set(ratings[user2].keys())
    weight = len(common_movies) / len(ratings[user1])
    return similarity * weight


def cluster_users(ratings, n_clusters=3):
    users = list(ratings.keys())
    movies = sorted(set(m for user_ratings in ratings.values() for m in user_ratings))
    categories = sorted(set(c for user_ratings in ratings.values() for movie in user_ratings for c in user_ratings[movie]["categories"]))

    presence_matrix = []
    for user in users:
        user_vector = []

        user_vector.extend([1 if movie in ratings[user] else 0 for movie in movies])

        for category in categories:
            category_present = 0
            for movie, movie_data in ratings[user].items():
                if category in movie_data["categories"]:
                    category_present = 1
                    break
            user_vector.append(category_present)

        presence_matrix.append(user_vector)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(presence_matrix)
    user_clusters = {users[i]: clusters[i] for i in range(len(users))}

    return user_clusters


def generate_recommendations_with_similarity(
    user, ratings, similarity_func, user_clusters, n_recommendations=5, n_anti_recommendations=5):
    user_cluster = user_clusters[user]
    cluster_users = [
        other_user
        for other_user, cluster in user_clusters.items()
        if cluster == user_cluster and other_user != user
    ]

    sorted_cluster_users = sorted(
        cluster_users,
        key=lambda other_user: similarity_func(user, other_user, ratings),
        reverse=True,
    )

    def find_movies(target_users, threshold_recommend=8, threshold_anti=3):
        recommendations, anti_recommendations = [], []

        for other_user in target_users:
            for movie, rating_info in sorted(ratings[other_user].items(), key=lambda x: x[1]["rating"], reverse=True):
                rating = rating_info["rating"]
                if rating >= threshold_recommend and movie not in ratings[user]:
                    recommendations.append(movie)
                if rating <= threshold_anti and movie not in ratings[user]:
                    anti_recommendations.append(movie)

                if len(recommendations) >= n_recommendations and len(anti_recommendations) >= n_anti_recommendations:
                    return recommendations[:n_recommendations], anti_recommendations[:n_anti_recommendations]

        return recommendations, anti_recommendations

    recommendations, anti_recommendations = find_movies(sorted_cluster_users)

    if len(recommendations) < n_recommendations or len(anti_recommendations) < n_anti_recommendations:
        recommendations, anti_recommendations = find_movies(sorted_cluster_users, threshold_recommend=7, threshold_anti=4)

    if len(recommendations) < n_recommendations or len(anti_recommendations) < n_anti_recommendations:
        for cluster_id in set(user_clusters.values()):
            if cluster_id == user_cluster:
                continue

            cluster_users = [other_user for other_user, cluster in user_clusters.items() if cluster == cluster_id]
            sorted_cluster_users = sorted(cluster_users, key=lambda other_user: similarity_func(user, other_user, ratings), reverse=True)

            additional_recommendations, additional_anti_recommendations = find_movies(
                sorted_cluster_users, threshold_recommend=8, threshold_anti=3
            )

            recommendations.extend(additional_recommendations)
            anti_recommendations.extend(additional_anti_recommendations)

            if len(recommendations) >= n_recommendations and len(anti_recommendations) >= n_anti_recommendations:
                break

    return recommendations[:n_recommendations], anti_recommendations[:n_anti_recommendations]

def get_movie_info(movie_name, api_key):
    if api_key is not None:
        url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": api_key,
            "query": movie_name,
            "language": "pl"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                movie = data['results'][0]
                print(f"Tytuł: {movie['title']}")
                print(f"Data wydania: {movie.get('release_date', 'Brak danych')}")
                print(f"Opis: {movie.get('overview', 'Brak opisu')}\n")
            else:
                print("Nie znaleziono filmu.\n")
        else:
            print("Nie udało się połączyć z API.\n")


def main():
    parser = argparse.ArgumentParser(description='Rekomendacje filmowe')
    parser.add_argument('username', type=str, help='Nazwa użytkownika, dla którego generujemy rekomendacje')
    parser.add_argument('filename', type=str, help='Plik z danymi w formacie JSON')
    parser.add_argument('--api', type=str, help='Podaj api do szczegółowych informacji')
    parser.add_argument('--method', type=str, choices=['euclidean', 'pearson'], default='euclidean', help='Metoda obliczania podobieństwa (domyślnie pearson)')
    parser.add_argument('--clusters', type=int, default=3, help='Liczba klastrów (domyślnie 3)')
    args = parser.parse_args()

    ratings = load_data(args.filename)
    user_clusters = cluster_users(ratings, args.clusters)
    similarity_func = lambda u1, u2, r: weighted_similarity(u1, u2, r, method=args.method)
    recommendations, anti_recommendations = generate_recommendations_with_similarity(args.username, ratings, similarity_func, user_clusters)

    print(f"Rekomendacje dla użytkownika {args.username}:")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")
        get_movie_info(movie, args.api)

    print(f"\nAnty rekomendacje dla użytkownika {args.username}:")
    for i, movie in enumerate(anti_recommendations, 1):
        print(f"{i}. {movie}")
        get_movie_info(movie, args.api)

if __name__ == "__main__":
    main()