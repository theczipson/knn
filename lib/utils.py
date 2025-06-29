import numpy as np
import pandas as pd


def get_min_max_mfcc_values(mfcc_clip_data_dict):
    """
    Funkcja zwracajaca tablice numeryczne wartosci minimalnych i maksymalnych dla kazdej cechy wektora mfcc
    """
    all_data_combined = np.concatenate(
        [
            item["feature"].reshape(1, item["feature"].shape[0])
            for item in mfcc_clip_data_dict.values()
        ],
        axis=0,
    )
    min_values_arr = np.min(all_data_combined, axis=0)
    max_values_arr = np.max(all_data_combined, axis=0)

    return min_values_arr, max_values_arr


def map_label(label):
    """
    Funkcja mapujaca etykiete do wartosci liczbowej, ktora ja reprezentuje
    """
    label_map_dict = {
        "teens": 10,
        "twenties": 20,
        "thirties": 30,
        "fourties": 40,
        "fifties": 50,
        "nineties": 90,
    }
    return label_map_dict[label]


def get_mfcc_dataframe(mfcc_dict):
    """
    funkcja tworzaca dataframe na podstawie slownika z wektorami mfcc
    """
    # dodanie etkiety pod postacia liczby dla kazdego pliku do wektora cech mfcc
    normalized_mfcc_data = np.concatenate(
        [
            np.concat(
                [
                    item["normalized_feature"].reshape(
                        1, item["normalized_feature"].shape[0]
                    ),
                    np.array(map_label(item["label"])).reshape(1, 1),
                ],
                axis=1,
            )
            for item in mfcc_dict.values()
        ],
        axis=0,
    )

    # ustalenie przedzialow wartosci dla wektora mfcc
    end_range = 101
    conditions = [
        np.logical_and(
            normalized_mfcc_data >= (x / 10) - 0.1, normalized_mfcc_data < (x / 10)
        )
        for x in range(1, end_range, 1)
    ]
    values = [x for x in range(1, end_range, 1)]

    # mapowanie wartosci wektora mfcc do odpowiedajacych im przedzialow
    mapped_arr = np.select(conditions, values, default=normalized_mfcc_data)

    mfcc_df = pd.DataFrame(mapped_arr)

    return mfcc_df


def entropy(series):
    """
    Funkcja licząca entropię.
    """
    probabilities = series.value_counts(normalize=True)
    return -sum(probabilities * np.log2(probabilities))


def information_gain(df, target_col, split_col):
    """
    Funkcja licząca przyrost informacji
    """

    # Oblicz entropię dla całej kolumny
    total_entropy = entropy(df[target_col])

    # podzial danych wedlug wybranej kolumny
    grouped = df.groupby(split_col)

    # obliczenie entropii uwzgledniajac podzial na grupy
    group_entropy = sum(
        (len(group) / len(df)) * entropy(group[target_col]) for _, group in grouped
    )

    # zwrocenie zysku informaci
    return total_entropy - group_entropy


def calculate_information_gains(mfcc_dict):
    """
    Funkcja liczaca zysk informacji
    """
    # utworzenie dataframe'u z wartosci wektorow mfcc
    df = get_mfcc_dataframe(mfcc_dict)

    # wybranie wszsytkich kolumn oprocz kolumny opisujace klase
    last_col = df.columns[-1]

    # obliczenie zyskow dla kazdej kolumny
    gains_lst = [(col, information_gain(df, last_col, col)) for col in df.columns[:-1]]
    gains_lst.sort(key=lambda x: x[0])

    return np.array([gain for _, gain in gains_lst], np.float32)
