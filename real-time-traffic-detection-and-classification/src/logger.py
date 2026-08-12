"""
    Real-Time Street Traffic Detection and Classification
    CS5330 Pattern Recognition & Computer Vision
    Drew Hill & Abhiram Banda

    logger.py

"""
import csv
import configs

class Logger:
    """


    """
    def __init__(self, log_file_path):
        """


        """
        if not log_file_path.exists() or log_file_path.stat().st_size == 0:
            is_new = True

        else:
            is_new = False

        self.log_file = open(log_file_path, "a", newline="")
        self.writer = csv.DictWriter(self.log_file, fieldnames=configs.FIELDS)
        if is_new:
            self.writer.writeheader()
            self.log_file.flush()

    def write_row(self, row):
        """


        """
        self.writer.writerow(row)
        self.log_file.flush()

    def close(self):
        """

        """
        self.log_file.close()
