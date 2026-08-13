"""
    Real-Time Street Traffic Detection and Classification
    CS5330 Pattern Recognition & Computer Vision
    Drew Hill & Abhiram Banda

    logger.py

    Writes the information for each object that is detcted, classified, and 
    counted to a csv file.
"""
import csv
import configs

class Logger:
    """
        Class that writes the info about the object to the csv file.
    """
    def __init__(self, log_file_path):
        """
            log_file_path: Path - Path to the csv file.
        """
        # Indicates a new file if it doesnt exist yet or the file is empty.
        if not log_file_path.exists() or log_file_path.stat().st_size == 0:
            is_new = True

        else:
            is_new = False

        # Opens the file and sets up the writer.
        self.log_file = open(log_file_path, "a", newline="")
        self.writer = csv.DictWriter(self.log_file, fieldnames=configs.FIELDS)

        # Writes the headers if new file or file is empty.
        if is_new:
            self.writer.writeheader()
            self.log_file.flush()

    def write_row(self, row):
        """
            Writes a row with object info and saves it.

            row: dict - Info about the object.
        """
        self.writer.writerow(row)
        self.log_file.flush()

    def close(self):
        """
            Closes the csv file.
        """
        self.log_file.close()
