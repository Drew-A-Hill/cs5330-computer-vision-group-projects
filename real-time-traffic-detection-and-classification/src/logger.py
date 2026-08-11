import csv

class Logger:
    def __init__(self, log_file_path):
        if not log_file_path.exists() or log_file_path.stat().st_size == 0:
            is_new = True

        else:
            is_new = False

        self.log_file = open(log_file_path, "a", newline="")
        self.writer = csv.writer(self.log_file)
        if is_new:
            self.writer.writerow(["timestamp", "street", "cross_street", "direction", "class"])
            self.log_file.flush()

    def write_row(self, row):
        self.writer.writerow(row)
        self.log_file.flush()

    def close(self):
        self.log_file.close()
