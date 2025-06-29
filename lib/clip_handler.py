import numpy as np
import librosa.feature
import os
import random
import pandas as pd
from lib.utils import get_min_max_mfcc_values
from threading import Lock
from concurrent.futures import ThreadPoolExecutor


class ClipsHandler:
    """
    Klasa odpowiadająca za odczytanie plikow z nagraniami, okreslenie wektora MFCC nagrania i jego pochodnej,
    obliczenie wartosci sredniej i mediany dla kazdej cechy opisanej przez wektro i pochodna

    UWAGA: KLASA KORZYSTA Z WIELOWATKOWOSCI!
    """

    def __init__(self, logger, user_input, clip_data_csv, clips_path):
        self.logger = logger
        self.locker = Lock()

        # odczytanie informacji o nagraniach i podzial na zbior testowy i uczacy
        self.learn_data_lst = []
        self.test_data_lst = []
        self.get_clips_data(
            clip_data_csv, user_input.number_of_examples, user_input.percent_to_learn
        )

        # obliczenie wartosci sredniej i mediany wektorow MFCC i ich pochodnych
        self.mfcc_learn_clip_data_dict = dict()
        self.mfcc_test_clip_data_dict = dict()
        self.get_all_files_mfcc(
            clips_path,
            user_input.number_of_features,
            user_input.frame_duration_ms / 1000,
            user_input.hop_duration,
        )

        # normalizacja wektorow
        self.normalize_mfcc()

    def get_clips_data(
        self, clips_data_path, example_in_class_cnt, learn_data_percentage
    ):
        """
        Metoda odpowiadajaca za odczytanie danych o nagraniach, wyborze probek dla kazdej klasy,
        podzial probek na uczace i testowe
        """
        # odczytanie informacji o probkach
        clips_data_df = pd.read_csv(clips_data_path, sep="\t", low_memory=False)[["path", "age"]]
        filtered_clips_data = clips_data_df[
            (clips_data_df["age"] != "seventies")
            & (clips_data_df["age"] != "sixties")
            & (clips_data_df["age"].notna())
        ]

        # wybor reprezentantow klas
        self.logger.save_log("")
        self.logger.save_log("Ilosc probek na dany wiek:")
        for age in filtered_clips_data["age"].unique():
            # wybor plikow dla danej klasy
            clip_lst = filtered_clips_data[filtered_clips_data["age"] == age][
                "path"
            ].tolist()

            # losowe sortowanie probek probek
            random.shuffle(clip_lst)

            # wybor reprezentantow
            shuffled_clip_lst = clip_lst[:example_in_class_cnt]

            # obliczenie ilosci prbboek uczacych dla danej klasy
            learn_data_cnt = int(
                round(len(shuffled_clip_lst) * learn_data_percentage / 100)
            )

            # wybor probek uczacych
            self.learn_data_lst.extend(
                [(clip, age) for clip in shuffled_clip_lst[:learn_data_cnt]]
            )

            # wybor probek testowych
            self.test_data_lst.extend(
                [(clip, age) for clip in shuffled_clip_lst[learn_data_cnt:]]
            )

            self.logger.save_log(
                f"{age}: uczace {learn_data_cnt} | testowe {len(shuffled_clip_lst) - learn_data_cnt}"
            )

    def get_recording_mfcc(
        self,
        clip_path,
        clip_name,
        label,
        mfcc_features_cnt,
        frame_duration=0.02,
        hop_length=50,
        test_data_flg=True,
    ):
        """
        Klasa odpowiedzialna za wyznaczenie wartosci srednich i mediany wektora MFCC danej probki
        """
        recording_path = os.path.join(clip_path, clip_name)

        # odczytanie probki
        recording_sr = librosa.get_samplerate(recording_path)
        recording, _ = librosa.load(recording_path, sr=recording_sr)

        # usuniecie ciszy z probki
        recoding_trimmed, _ = librosa.effects.trim(y=recording, top_db=40)

        # okreslenie dlugosci ramki
        frame_length = round(frame_duration * recording_sr)

        # odczytanie wektora MFCC
        mfcc_features = librosa.feature.mfcc(
            y=recoding_trimmed,
            sr=recording_sr,
            n_fft=frame_length,
            hop_length=int(round(frame_length * hop_length / 100)),
            n_mfcc=mfcc_features_cnt,
        )

        # odczytanie pochodnych wektora MFCC
        mfcc_derivative = librosa.feature.delta(mfcc_features)

        # zebranie cech w jeden wektor
        mean_feature_val = np.mean(mfcc_features, axis=1)
        median_feature_val = np.median(mfcc_features, axis=1)
        mean_derivative_feature_val = np.mean(mfcc_derivative, axis=1)
        median_derivative_feature_val = np.median(mfcc_derivative, axis=1)
        all_feature_val = np.concatenate(
            [
                mean_feature_val,
                median_feature_val,
                mean_derivative_feature_val,
                median_derivative_feature_val,
            ]
        )

        # zapis wartosci probki do zadanej wartosci
        if test_data_flg:
            with self.locker:
                self.mfcc_test_clip_data_dict[clip_name] = {}
                self.mfcc_test_clip_data_dict[clip_name]["label"] = label
                self.mfcc_test_clip_data_dict[clip_name]["feature"] = all_feature_val
        else:
            with self.locker:
                self.mfcc_learn_clip_data_dict[clip_name] = {}
                self.mfcc_learn_clip_data_dict[clip_name]["label"] = label
                self.mfcc_learn_clip_data_dict[clip_name]["feature"] = all_feature_val

    def normalize_mfcc(self):
        """
        Metoda odpowiedzialana za normalizacje wartosci srednich i mediany wektora MFCC
        """
        self.logger.save_log("")
        self.logger.save_log("Normalizacja MFCC...")

        # odczytanie wartosci minimalne i maksymalnej kazdej cechy (wartosci sredniej i medianej danej cechy)
        min_mfcc_arr, max_mfcc_arr = get_min_max_mfcc_values(
            {**self.mfcc_test_clip_data_dict, **self.mfcc_learn_clip_data_dict}
        )

        # normalizacja wektorow cech dla kazdej probki
        for key in self.mfcc_test_clip_data_dict.keys():
            self.mfcc_test_clip_data_dict[key]["normalized_feature"] = (
                self.mfcc_test_clip_data_dict[key]["feature"] - min_mfcc_arr
            ) / (max_mfcc_arr - min_mfcc_arr)

        for key in self.mfcc_learn_clip_data_dict.keys():
            self.mfcc_learn_clip_data_dict[key]["normalized_feature"] = (
                self.mfcc_learn_clip_data_dict[key]["feature"] - min_mfcc_arr
            ) / (max_mfcc_arr - min_mfcc_arr)

        self.logger.save_log("Koniec normalizacji MFCC!")

    def get_all_files_mfcc(
        self, src_path, mfcc_features_cnt, frame_duration=0.02, hop_length=50
    ):
        """
        Metoda odpowiedzialna za odczytanie cehc wszystkich probek
        """
        self.logger.save_log("")
        self.logger.save_log("Wczytywanie probek uczacych...")

        # uruchomienie wielowatkowsci
        with ThreadPoolExecutor(max_workers=20) as executor:
            # odczytanie wektorow cech zbioru uczacego
            for file, age in self.learn_data_lst:
                executor.submit(
                    self.get_recording_mfcc,
                    src_path,
                    file,
                    age,
                    mfcc_features_cnt,
                    frame_duration=frame_duration,
                    hop_length=hop_length,
                    test_data_flg=False,
                )
        self.logger.save_log(
            f"Wczytano wszystkie probki uczace ({len(self.mfcc_learn_clip_data_dict)})"
        )

        self.logger.save_log("")
        self.logger.save_log("Wczytywanie probek testowyh...")
        with ThreadPoolExecutor(max_workers=20) as executor:
            # odczytanie wektora cech probek uczacych
            for file, age in self.test_data_lst:
                executor.submit(
                    self.get_recording_mfcc,
                    src_path,
                    file,
                    age,
                    mfcc_features_cnt,
                    frame_duration=frame_duration,
                    hop_length=hop_length,
                )
        self.logger.save_log(
            f"Wczytano wszystkie probki testowe ({len(self.mfcc_test_clip_data_dict)})"
        )
