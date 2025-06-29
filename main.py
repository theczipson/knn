from lib.logger import Logger
from lib.user_input import UserInput
from lib.clip_handler import ClipsHandler
from lib.knn_threading import KNearestNeighbours
import os


class App:
    def __init__(self):
        self.logger = Logger()
        self.user_input = UserInput(self.logger)
        self.clip_handler = ClipsHandler(
            self.logger,
            self.user_input,
            os.path.join(self.user_input.samples_path, "validated.tsv"),
            os.path.join(self.user_input.samples_path, "clips"),
        )
        self.knn_model = KNearestNeighbours(
            self.clip_handler.mfcc_learn_clip_data_dict,
            self.clip_handler.mfcc_test_clip_data_dict,
            self.logger,
        )

        self.knn_model.get_all_points_labels(
            k_neighbours=self.user_input.number_of_neighbours,
            normalized_mfcc=True,
            information_gain_as_weight=True,
            information_gain_threshold=0.0,
        )


if __name__ == "__main__":
    app = App()
    input("Wcisnij enter zeby zakonczyc...")