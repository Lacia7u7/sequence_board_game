import os, csv, json, time
class Logger:
    def __init__(self, config):
        self.logdir = config.get("logging",{}).get("logdir","runs")
        os.makedirs(self.logdir, exist_ok=True)
        self.csv_path = os.path.join(self.logdir, "metrics.csv")
        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if self.csv_file.tell() == 0:
            self.csv_writer.writerow(["step","key","value","timestamp"])
    def log_metrics(self, step: int, metrics: dict):
        ts = int(time.time())
        for k,v in metrics.items():
            self.csv_writer.writerow([step, k, v, ts])
        self.csv_file.flush()
    def close(self):
        try: self.csv_file.close()
        except Exception: pass
