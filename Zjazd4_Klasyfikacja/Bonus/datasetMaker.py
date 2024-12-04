import pandas as pd

mappings = {
    "Ile masz lat? (Podaj wiek)": {
        "22 od dzisiaj": 22
    },
    "Jaka jest Twoja płeć?": {
        "Mężczyzna": 0,
        "Kobieta": 1,
        "Inna": 2
    },
    "Jaki gatunek gier preferujesz?": {
        "Strzelanki": 0,
        "Battle Royale": 1,
        "Gry RPG": 2,
        "Souls Like": 2,
        "Gry strategiczne": 3,
        "Symulatory": 4,
        "Gry logiczne": 5,
        "Gry przygodowe": 6,
        "Inne": 7
    },
    "Czy używasz modów do gier?": {
        "Tak, często": 0,
        "Okazjonalnie": 1,
        "Gry RPG": 2,
        "Nie": 2
    },
    "Na jakim sprzęcie grasz najczęściej?": {
        "PC": 0,
        "Konsola (PS/Xbox)": 1,
        "Mobilne urządzenia": 2,
        "Inne": 2
    },
    "Na jakiej platformie grasz najczęściej?": {
        "Steam": 0,
        "Origin": 1,
        "Epic Games": 2,
        "Battle.net": 3,
        "Inne": 3
    },
    "Ile czasu dziennie średnio spędzasz na grach (Podaj w godzinach)?": {
        "8-10": 8,
        "30": 5
    },
    "O jakiej porze dnia preferujesz granie?": {
        "Rano": 0,
        "Po południe": 1,
        "Wieczór": 2,
        "Późna noc": 3
    },
    "Jaki tryb gry preferujesz?": {
        "Gra lokalna (Singleplayer)": 0,
        "Gra online (Multiplayer)": 1
    },
    "Kiedy zacząłeś/aś używać czatu głosowego w grach komputerowych?": {
        "Nigdy": 0,
        "Przed 2000": 1,
        "2000–2010": 2,
        "2010–2020": 3,
        "Po 2020": 4
    },
    "Jaką gre lubisz najbardziej? (Wybierz jedną)": {
        "Counter Strike (2, Global Offensive)": 0,
        "Counter Strike 2": 0,
        "Valorant": 1,
        "Fortnite": 2,
        "Minecraft": 3,
        "League of Legends": 4,
        "The Sims 4": 5,
        "FIFA (np. FIFA 24)": 6,
        "Call of Duty": 7,
        "Roblox": 8,
        "GTA V": 9,
        "Mario": 10,
        "Zelda": 11,
        "Pokémon": 12,
        "Inna": 13,
    },
    "Która z poniższych gier wywarła na Tobie największe wrażenie wizualne? (Wybierz jedną)": {
        "Mario": 0,
        "The Legend of Zelda: Breath of the Wild": 1,
        "Final Fantasy": 2,
        "Half-Life 2": 3,
        "Vrchat": 4,
        "Fortnite": 5,
        "The Witcher 3: Wild Hunt": 6,
        "Horizon Zero Dawn": 7,
        "Cyberpunk 2077": 8,
        "GTA 5": 9,
        "Inne": 10,
    },
    "W jakiej formie najczęściej komunikujesz się z innymi podczas grania?": {
        "Discord": 0,
        "TeamSpeak": 1,
        "Inne": 2,
    },
    "Jak oceniasz swoje umiejętności w grach komputerowych?": {
        "Bardzo zaawansowane": 0,
        "Średnio zaawansowane": 1,
        "Podstawowe": 2,
        "Brak umiejętności": 3,
    },
    "Czy śledzisz esport lub zawody gamingowe?": {
        "Tak, regularnie": 0,
        "Tak, sporadycznie": 1,
        "Nie, nie interesuje mnie to": 2,
    },
    "Jaki był twój pierwszy sprzęt, na którym zagrałeś/aś swoją pierwszą grę (Jeżeli nie dokładnie to bardzo podobny).": {
        "Atari": 0,
        "NES/SNES": 1,
        "PlayStation 1/2": 2,
        "Xbox (Original/360)": 3,
        "Game Boy/DS": 4,
        "Współczesna konsola (PS4/PS5, Xbox One/Series, itp.)": 5,
        "PC (przed 2005 rokiem)": 6,
        "PC (po 2005 roku)": 7,
        "PC (po 2000 roku)": 7,
        "Smartfon/Telefon (przed 2005 rokiem)": 8,
        "Smartfon/Telefon (po 2005 roku)": 9,
    },
    "Które z tych kultowych momentów związanych z grami pamiętasz, jako najstarsze?": {
        "Premiera Pokémon Red/Blue": 0,
        "Hype wokół gier w czasach ery Y2K": 1,
        "Premiera Halo 3": 2,
        "Szał na Minecrafta w latach 2010": 3,
        "Fortnite World Cup": 4,
    },
    "Który z tych dźwięków lub utworów z gier najbardziej utkwił Ci w pamięci?": {
        "Dźwięk zbierania monety z Mario": 0,
        'Motyw przewodni z "The Legend of Zelda"': 1,
        'Intro z "Halo"': 2,
        'Motyw przewodni z "Tetrisa"': 3,
        '„Still Alive” z "Portal"': 4,
        'Dźwięk Creepera z "Minecraft"': 5,
        'Dźwięk z "Roblox" "Death Sound"': 6,
        'Motyw przewodni z "The Elder Scrolls V: Skyrim" (Dragonborn)': 7,
        '„Megalovania” z "Undertale"': 8,
        'Dźwięk piosenki Victory Royale z "Fortnite"': 9,
        'Motyw przewodni z "The Last of Us" (melancholijna gitara)': 10,
        'Muzyka z lobby CSGO': 11,
    },
}

df = pd.read_csv('Gamingowe preferencje i doświadczenia.csv')

df = df.drop(df.columns[0], axis=1)

def convert_to_id(row, mappings):
    for column in row.index:
        if column in mappings:
            if row[column] in mappings[column]:
                row[column] = mappings[column][row[column]]
    return row

df = df.apply(lambda row: convert_to_id(row, mappings), axis=1)

df.to_csv('Gaming_Dataset.csv', index=False, header=False)