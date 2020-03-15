class InflowModelFactory:
    def create_normal_inflow_model(self, history):
        return NormalInflow(history)


class NormalInflow:
    def __init__(self, control_data):
        self.counter = 1
        self.control_data = control_data

    def eval(self):
        if self.counter < len(self.control_data):
            value = (0.0, self.control_data[self.counter])
            self.counter += 1
            return value
        return 0.0, 0.0
