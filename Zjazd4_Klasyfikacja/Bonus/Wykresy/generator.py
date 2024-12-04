import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'Gamingowe preferencje i doświadczenia.csv'

data = pd.read_csv(file_path)

data['Pełnoletni'] = data['Ile masz lat? (Podaj wiek)'] >= 18

output_folder = "."
os.makedirs(output_folder, exist_ok=True)

categorical_columns = [
    col for col in data.columns
    if data[col].dtype == 'object' and col not in ['Sygnatura czasowa']
]

for col in categorical_columns:
    grouped_data = data.groupby(['Pełnoletni', col]).size().unstack(0).fillna(0)

    grouped_data.plot(kind='bar', figsize=(12, 6))
    plt.title(f"Pełnoletni vs. Niepełnoletni dla: {col}")
    plt.ylabel("Liczba osób")
    plt.xlabel(col)
    plt.xticks(rotation=45, ha='right')
    plt.legend(["Niepełnoletni", "Pełnoletni"], title="Grupa wiekowa")
    plt.tight_layout()

    file_name = f"{col.replace(' ', '_').replace('/', '_')}_plot.png"
    plt.savefig(os.path.join(output_folder, file_name))
    plt.show()
    plt.close()


print(f"Wykresy zostały zapisane w folderze: {output_folder}")
