import os


class UserInput:
    """
    Klasa odpowiadajaca za zbieranie i przetrzymywanie wartosci wejsciowych,
     podanych przez uzytkownika w konsoli programu
    """

    def __init__(self, logger):
        """
        Konstruktor klasy zbierający informacje od uzytkownika
        i podajacy je do zapisu do pliku monitorujacego.
        """
        logger.save_log("Pobieranie danych od uzytkownika")

        # odczytanie sciezki do danych
        while True:
            samples_path = input("Sciezka bezwgledna do struktury z danymi: ")
            if os.path.exists(os.path.join(samples_path, "clips")) and os.path.exists(
                os.path.join(samples_path, "validated.tsv")
            ):
                self.samples_path = samples_path
                logger.save_log("Sciezka do struktury z danymi: " + self.samples_path)
                break
            else:
                logger.save_log(
                    "Sciezka nie istenieje albo nie zawiera folderu z nagraniami i/lub pliku z opisem probek!"
                )

        # odczytanie ilosci przykladow
        while True:
            try:
                self.number_of_examples = int(input("Ilosc przykladow: "))
                if self.number_of_examples > 0:
                    logger.save_log("Ilosc przykladow: " + str(self.number_of_examples))
                    break
                else:
                    logger.save_log("Podaj liczbe dodatnia jako ilosc przykladow!")
            except ValueError:
                logger.save_log("Nie podales liczby! Sprobuj jeszcze raz!")

        # odczytanie podzialu probek na zbior uczacu i testowy
        while True:
            try:
                percent_to_learn = float(input("Procent przeznaczony na uczenie: "))
                if 0 < percent_to_learn < 100:
                    self.percent_to_learn = percent_to_learn
                    logger.save_log(
                        "Procent przeznaczony na uczenie: " + str(self.percent_to_learn)
                    )
                    break
                else:
                    logger.save_log(
                        "Wartosc procentowa musi byc z zakresu (0, 100)! Sprobuj jeszcze raz!"
                    )
            except ValueError:
                logger.save_log("Nie podales liczby! Sprobuj jeszcze raz!")

        # odczytanie ilosci sasiadow
        while True:
            try:
                number_of_neighbours = int(input("Ilosc sasiadow z zkresu 1-5: "))
                if 0 < number_of_neighbours < 6:
                    self.number_of_neighbours = number_of_neighbours
                    logger.save_log("Ilosc sasiadow: " + str(self.number_of_neighbours))
                    break
                else:
                    logger.save_log(
                        "Ilosc sasiadow musi byc z zakresu 1-5! Sprobuj jeszcze raz!"
                    )
            except ValueError:
                logger.save_log("Nie podales liczby! Sprobuj jeszcze raz!")

        # odczytanie ilosci cech wektora MFCC
        while True:
            try:
                number_of_features = int(input("Ilosc cech MFCC z zakresu 12-30: "))
                if 11 < number_of_features < 31:
                    self.number_of_features = number_of_features
                    logger.save_log("Ilosc cech MFCC: " + str(self.number_of_features))
                    break
                else:
                    logger.save_log(
                        "Ilosc cech MFCC musi byc z zakresu 12-30! Sprobuj jeszcze raz!"
                    )
            except ValueError:
                logger.save_log("Nie podales liczby! Sprobuj jeszcze raz!")

        # odczytanie dlugosci ramki
        while True:
            try:
                frame_duration_ms = int(input("Dlugosc ramki w milisekundach: "))
                if 0 < frame_duration_ms:
                    self.frame_duration_ms = frame_duration_ms
                    logger.save_log(
                        "Dlugosc ramki: " + str(self.frame_duration_ms) + "ms"
                    )
                    break
                else:
                    print("Dlugosc ramki musi byc wieksza od 0! Sprobuj jeszcze raz!")
            except ValueError:
                print("Nie podales liczby! Sprobuj jeszcze raz!")

        # odczytania stopnia przeplotu ramek sygnalu wejsciowego
        while True:
            try:
                hop_duration = int(input("Stopien nakladania sie ramek (1-99%): "))
                if 1 <= hop_duration <= 100:
                    self.hop_duration = hop_duration
                    logger.save_log(
                        "Stopien przeplotu ramek: " + str(self.hop_duration) + "%"
                    )
                    break
                else:
                    print(
                        "Stopien przeplotu ramek musi byc z zakresu (1-99)! Sprobuj jeszcze raz!"
                    )
            except ValueError:
                print("Nie podales liczby! Sprobuj jeszcze raz!")
