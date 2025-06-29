from collections import namedtuple
from itertools import product
import os
import pandas as pd
from lib.logger import Logger
from lib.clip_handler import ClipsHandler
from lib.knn_threading import KNearestNeighbours

_test_path = os.path.join("D:", "msi_data", "cv-corpus-19.0-2024-09-13", "pl")
_test_clip_path = os.path.join(_test_path, "clips")

user_input = namedtuple(
    "user_input",
    [
        "number_of_examples",
        "percent_to_learn",
        "number_of_features",
        "frame_duration_ms",
        "hop_duration",
    ],
)

number_of_neighbours = [i for i in range(1, 6, 1)]
all_func_args_possibilities_lst = [
    {
        # "k_neighbours": neighbour_cnt,
        "normalized_mfcc": normalized_mfcc,
        # "use_information_gain": use_information_gain,
        "information_gain_as_weight": information_gain_as_weight,
        "information_gain_threshold": information_gain_threshold,
    }
    for information_gain_threshold in [0.0, 0.07]
    for information_gain_as_weight in [False, True]
    # for use_information_gain in [False, True]
    for normalized_mfcc in [False, True]
]


class KNNClassificatorFunctionTester:
    def __init__(self):
        self.logger = Logger("test_cases_knn_")

    def run_test(self):
        test_cases_summary_dict = {str(i): {} for i in range(1, 6)}
        test_cases_guesses_summary_dict = {str(i): {} for i in range(1, 6)}

        test_case = user_input(200, 80, 30, 20, 50)

        for j in range(5):
            clip_handler = ClipsHandler(
                self.logger,
                test_case,
                os.path.join(_test_path, "validated.tsv"),
                _test_clip_path,
            )
            knn_model = KNearestNeighbours(
                clip_handler.mfcc_learn_clip_data_dict,
                clip_handler.mfcc_test_clip_data_dict,
                self.logger,
            )
            for n_neighbour in range(1, 6):
                self.logger.save_log(f"")
                self.logger.save_log(f"Ilosc sasiadow: {n_neighbour} | Proba {j}")
                for func_kwargs in all_func_args_possibilities_lst:
                    test_type, percent_guessed, guesses = (
                        knn_model.get_all_test_points_labels(
                            k_neighbours=n_neighbour, **func_kwargs
                        )
                    )
                    if (
                        test_type
                        not in test_cases_summary_dict[str(n_neighbour)].keys()
                    ):
                        test_cases_summary_dict[str(n_neighbour)][test_type] = [
                            percent_guessed
                        ]
                        test_cases_guesses_summary_dict[str(n_neighbour)][test_type] = [
                            guesses
                        ]
                    else:
                        test_cases_summary_dict[str(n_neighbour)][test_type].append(
                            percent_guessed
                        )
                        test_cases_guesses_summary_dict[str(n_neighbour)][
                            test_type
                        ].append(guesses)

        for i in range(1, 6):
            summary_df = pd.DataFrame(test_cases_summary_dict[str(i)]).transpose()
            self.logger.save_log(f"")
            self.logger.save_log(
                f"Ilosc sasiadow {i} podsumowanie procentowe:\n"
                + summary_df.to_string()
            )

            summary_df = pd.DataFrame(
                test_cases_guesses_summary_dict[str(i)]
            ).transpose()
            self.logger.save_log(f"")
            self.logger.save_log(
                f"Ilosc sasiadow {i} podsumowanie \n" + summary_df.to_string()
            )


class KNNClassificatorMFCCValuesTester:
    def __init__(self):
        self.logger = Logger("test_cases_knn_")

    def run_test(self):
        test_cases_summary_dict = {}

        number_of_examples = [100, 500, 1000, 2000]
        percent_to_learn = [70, 80, 90]
        number_of_features = [12, 20, 30]
        frame_duration_ms = [i * 5 for i in range(4, 7, 1)]
        hop_duration = [25 * i for i in range(1, 4)]

        all_test_cases_lst = list(
            user_input(*test_case)
            for test_case in product(
                number_of_examples,
                percent_to_learn,
                number_of_features,
                frame_duration_ms,
                hop_duration,
            )
        )

        for test_case in all_test_cases_lst:
            clip_handler = ClipsHandler(
                self.logger,
                test_case,
                os.path.join(_test_path, "valid_not_empty_age.csv"),
                _test_clip_path,
            )
            knn_model = KNearestNeighbours(
                clip_handler.mfcc_learn_clip_data_dict,
                clip_handler.mfcc_test_clip_data_dict,
                self.logger,
            )

            test_type, percent_guessed, guesses = knn_model.get_all_test_points_labels(
                k_neighbours=2,
                normalized_mfcc=True,
                information_gain_as_weight=True,
            )

            test_case_message = f"Ilosc przykladow: {test_case.number_of_examples}, procent uczacy: {test_case.percent_to_learn}, ilosc cehc: {test_case.number_of_features}, dlugosc ramki: {test_case.frame_duration_ms}, stopien przeplotu: {test_case.hop_duration}"

            test_cases_summary_dict[test_case_message] = {
                "Poprawne przewidywania procentowe": percent_guessed,
                "Poprawne przewidywania": guesses,
            }

        self.logger.save_log(
            "Podsumowanie \n: "
            + pd.DataFrame(test_cases_summary_dict).transpose().to_string()
        )


if __name__ == "__main__":
    KNNClassificatorFunctionTester().run_test()
    # KNNClassificatorMFCCValuesTester().run_test()
