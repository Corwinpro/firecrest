import csv


class LoggerFactory:
    def __init__(self, run_mode, experiment_id, log_names, **kwargs):
        self.run_mode = run_mode
        self.experiment_id = experiment_id
        self.log_names = log_names
        self.kwargs = kwargs

    def create_basic_logger(self):
        return BaseLogger(
            run_mode=self.run_mode,
            experiment_id=self.experiment_id,
            log_names=self.log_names,
        )


class BaseLogger:
    def __init__(self, run_mode, experiment_id, log_names, **kwargs):
        self.run_mode = run_mode
        self.experiment_id = experiment_id
        self.log_names = log_names

        self.plot_every = self.log_names.get("plot_every", 10000)

    def log_name(self, filename):
        return filename + self.experiment_id + ".log"

    def write(self, file_name, data):
        with open(file_name, "a") as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def if_log_data(self, log_type):
        if log_type == "in_process":
            return self.run_mode == "single_run" and self.log_names.get("in_process")
        elif log_type == "optimization":
            return self.run_mode == "optimization" and self.log_names.get(
                "optimization"
            )
        return False

    def log_intermediate_step(self, data):
        if self.run_mode == "single_run" and self.log_names["in_process"]:
            file_name = self.log_name(self.log_names["in_process"])
            self.write(file_name, data)

    def log_optimization_step(self, data):
        if self.run_mode == "optimization" and self.log_names["optimization"]:
            file_name = self.log_name(self.log_names["optimization"])
            self.write(file_name, data)
