import os
import sys
from datetime import datetime


class Logger:
    """
    Logger zapisujacy informacje do pliku tekstowego wiadomosci z dzialania programu
    """

    def __init__(self, file_prefix="knn_log_file_"):
        if getattr(sys, 'frozen', False):
            file_path = sys.executable
        else:
            file_path = os.path.abspath(__file__)

        project_path = file_path.split(os.sep)[:-1]
        logger_file_path = os.sep.join(
            project_path
            + [f"{file_prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"]
        )
        self.logger_file_path = logger_file_path
        print("Plik z logami: " + self.logger_file_path)

    def save_log(self, message: str, save_to_file=True):
        """
        Funkja zapisujaca wiadomosci do pliku z logami oraz wyswietlajca je w konsoli

        Argumenty:
            message - wiadomosc, ktora zostanie zapisana do pliku
        """
        logger_message = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: " + message
            if message
            else ""
        )
        print(logger_message)

        if save_to_file:
            with open(self.logger_file_path, "a", encoding="utf-8") as f:
                f.write(logger_message + "\n")
