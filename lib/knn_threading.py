from collections import Counter
import numpy as np
from copy import deepcopy
import pandas as pd
from lib.utils import calculate_information_gains


class KNearestNeighbours:
    """
    Klasa klasyfikatora kNN
    """

    def __init__(self, learn_samples, test_samples, logger):
        # przypisanie probek uczacych i testowych
        self.learn_samples_dict = learn_samples
        self.test_samples_dict = test_samples
        self.logger = logger

        # obliczenie przyrostu informacji dla probek
        self.information_gain = calculate_information_gains(self.learn_samples_dict)

    @staticmethod
    def get_knn_label(
        test_sample, learn_samples, k_neighbours, normalized=False, information_gain=None
    ):
        """
        Metoda statyczna klasyfikacji probki testowej (test_sample) na podstawie ilosci (k_neighbours)
        nablizszych probek uczacych (learn_samples) wartosci znormalizownych (jesli normalized=True)
        oraz przyrsotu informacji (information_gain) jako wagi odleglosci
        """
        # wybranie wektora znormalizowanego lub nie probki testowej
        test_point_mfcc = (
            test_sample["feature"]
            if not normalized
            else test_sample["normalized_feature"]
        )

        # utworzenie przyrostu informacji o wartosci 1 dla kazdej cechy jesli nie zostal on podany
        if information_gain is None:
            information_gain = np.zeros(test_point_mfcc.shape) + 1

        distance_points = []
        label_points = []
        for learn_point in learn_samples.values():
            # wybranie cech znormalizowanych lub nie probki uczacej
            learn_point_mfcc = (
                learn_point["feature"]
                if not normalized
                else learn_point["normalized_feature"]
            )

            # utworzenie listy wspolrzednych dla kazdej probki oraz ich etykiet
            distance_points.append(learn_point_mfcc)
            label_points.append(learn_point["label"])

        # obliczenie odleglosci dla kazdej probki uczacej od probki testowej
        distance_points_arr = np.array(distance_points)
        distance_points_arr = np.sqrt(
            np.sum(
                information_gain * ((distance_points_arr - test_point_mfcc) ** 2),
                axis=1,
            )
        )

        # wybranie k probek o najmniejszej odleglosci
        min_dists = []
        for _ in range(k_neighbours):
            act_min_dist_idx = np.argmin(distance_points_arr)
            min_dists.append(
                (
                    distance_points_arr[act_min_dist_idx],
                    label_points[int(act_min_dist_idx)],
                )
            )
            distance_points_arr = np.delete(distance_points_arr, act_min_dist_idx)
            label_points = (
                label_points[:act_min_dist_idx] + label_points[act_min_dist_idx + 1 :]
            )

        # zliczenie etykiet probek o najmniejszej odleglosci
        counter = Counter([label for _, label in min_dists])
        most_common_lst = counter.most_common()

        # sprawdzenie najwiekszej licznosci dla danej etykiety
        max_count = max([cnt for _, cnt in most_common_lst])

        # wybranie etykiet o najwiekszej licznosci
        most_common_label_lst = list(
            set([label for label, cnt in most_common_lst if cnt == max_count])
        )

        # wybranie etykiety jesli tylko jedna klasa ma najwieksza licznosc
        if len(most_common_label_lst) == 1:
            return most_common_label_lst[0]
        # sprawdzenie sumy odleglosci dla etykiet o najwiekszej licznosci
        else:
            min_label = ""
            min_dist_sum = sum([distance for distance, _ in min_dists])
            for label in most_common_label_lst:
                dist = sum([distance for distance, lbl in min_dists if lbl == label])
                if dist < min_dist_sum:
                    min_dist_sum = dist
                    min_label = label
            return min_label

    def _get_all_points_labels(
        self,
        test_points,
        k_neighbours=3,
        normalized_mfcc=True,
        information_gain_as_weight=True,
        information_gain_threshold=0.000,
    ):
        """
        Metoda odpowiedzialna za wyznaczenie etykiet wszystkich probek
        """
        testing_steps = [
            round(x / 10 * (len(test_points)) - 1) for x in range(1, 11)
        ]
        matched_checks = 0

        # kopia probek testowych i uczacych
        test_samples_dict = deepcopy(test_points)
        learn_samples_dict = deepcopy(self.learn_samples_dict)

        # wybranie cech z przyrostem informacji wiekszym niz prog odciecia
        information_gain_best_lst = (
            [
                (i, gain)
                for i, gain in enumerate(self.information_gain)
                if gain > information_gain_threshold
            ]
            if information_gain_as_weight
            else [(i, 1) for i in range(len(self.information_gain))]
        )

        # wybranie cech dla kazdej probki spelniajacych przyrost informacji wiekszy niz prog odciecia
        if information_gain_threshold > 0.0:
            best_gain_vars_lst = [i for i, _ in information_gain_best_lst]

            for key in test_samples_dict.keys():
                test_samples_dict[key]["feature"] = test_samples_dict[key]["feature"][
                    best_gain_vars_lst
                ]
                test_samples_dict[key]["normalized_feature"] = test_samples_dict[key][
                    "normalized_feature"
                ][best_gain_vars_lst]

            for key in learn_samples_dict.keys():
                learn_samples_dict[key]["feature"] = learn_samples_dict[key]["feature"][
                    best_gain_vars_lst
                ]
                learn_samples_dict[key]["normalized_feature"] = learn_samples_dict[key][
                    "normalized_feature"
                ][best_gain_vars_lst]

        # generowanie wiadomosci sposobu testowania
        message = f"Testowanie "
        message += "znormalizowanych " if normalized_mfcc else ""
        message += "probek testowych "
        if information_gain_as_weight or information_gain_threshold > 0.00:
            message += "uwzgledniajac zysk informacji cech "
            message += (
                f"o wartosci powyzej {information_gain_threshold} "
                if information_gain_threshold > 0.00
                else ""
            )
            message += "jako waga odleglosci" if information_gain_as_weight else ""

        self.logger.save_log("")
        self.logger.save_log(message)

        # przewidywanie etykiet dla kazdej probki testowej
        label_guesses_dict = {}
        for i, test_point in enumerate(test_samples_dict.values()):
            if i in testing_steps:
                self.logger.save_log(
                    f"Przetestowano {round(i / (len(test_points) - 1) * 100)}% probek!",
                    save_to_file=False,
                )
            knn_label = self.get_knn_label(
                test_point,
                learn_samples_dict,
                k_neighbours,
                normalized=normalized_mfcc,
                information_gain=[gain for _, gain in information_gain_best_lst],
            )

            # przypisanie rozlozenia przewidywania
            if test_point["label"] not in label_guesses_dict:
                label_guesses_dict[test_point["label"]] = {}
            if knn_label + "_guess" not in label_guesses_dict[test_point["label"]]:
                label_guesses_dict[test_point["label"]][knn_label + "_guess"] = 1
            else:
                label_guesses_dict[test_point["label"]][knn_label + "_guess"] += 1

            # sprawdzenie poprawnosci przewidywania etykiet
            if knn_label == test_point["label"]:
                matched_checks += 1

        # zapisywanie podsumowan
        self.logger.save_log("")
        self.logger.save_log(
            "Tablica przewidywania etykiet\n"
            + pd.DataFrame(label_guesses_dict).fillna(0).to_string()
        )
        self.logger.save_log("")
        self.logger.save_log(
            f"Ilosc poprawnie przewidzianych etykiet: {matched_checks}/{len(test_points)}"
        )
        self.logger.save_log(
            f"Procent poprawnie przewidzianych etykiet: {round(matched_checks / len(test_points) * 100, 2)}%"
        )

        return (
            message,
            round(matched_checks / len(test_points) * 100, 2),
            f"{matched_checks}/{len(test_points)}",
        )

    def get_all_test_points_labels(
        self,
        k_neighbours=3,
        normalized_mfcc=True,
        information_gain_as_weight=True,
        information_gain_threshold=0.000,
    ):
        """
        Metoda przewidujaca etykiety punktow testowych
        """
        return self._get_all_points_labels(
            test_points=self.test_samples_dict,
            k_neighbours=k_neighbours,
            normalized_mfcc=normalized_mfcc,
            information_gain_as_weight=information_gain_as_weight,
            information_gain_threshold=information_gain_threshold,
        )

    def get_all_learn_points_labels(
        self,
        k_neighbours=3,
        normalized_mfcc=True,
        information_gain_as_weight=True,
        information_gain_threshold=0.000,
    ):
        """
        Metoda przewidujaca etykiety punktow uczacych
        """
        return self._get_all_points_labels(
            test_points=self.learn_samples_dict,
            k_neighbours=k_neighbours,
            normalized_mfcc=normalized_mfcc,
            information_gain_as_weight=information_gain_as_weight,
            information_gain_threshold=information_gain_threshold,
        )

    def get_all_points_labels(
        self,
        k_neighbours=3,
        normalized_mfcc=True,
        information_gain_as_weight=True,
        information_gain_threshold=0.000,
    ):
        """
        Metoda przewidujaca etykiety probek testowych oraz uczacych
        """
        self.logger.save_log("")
        self.logger.save_log("TESTOWANIE PROBEK UCZACYCH")
        self.get_all_learn_points_labels(
            k_neighbours=k_neighbours,
            normalized_mfcc=normalized_mfcc,
            information_gain_as_weight=information_gain_as_weight,
            information_gain_threshold=information_gain_threshold,
        )

        self.logger.save_log("")
        self.logger.save_log("TESTOWANIE PROBEK TESTOWYCH")
        self.get_all_test_points_labels(
            k_neighbours=k_neighbours,
            normalized_mfcc=normalized_mfcc,
            information_gain_as_weight=information_gain_as_weight,
            information_gain_threshold=information_gain_threshold,
        )
