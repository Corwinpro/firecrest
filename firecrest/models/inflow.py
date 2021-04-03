from dolfin import Expression


class InflowModelFactory:
    def create_normal_inflow_model(self, history):
        return NormalInflow(history)

    def create_parabolic_inflow_model(self, history, left, right):
        return ParabolicInflow(history, left=left, right=right)


class NormalInflow:
    _shape_expression_schema = ("0.0", "amplitude")

    def __init__(self, control_data):
        self.counter = 1
        self.control_data = control_data

        self._shape_expression = self.generate_shape_expression()

    @staticmethod
    def generate_shape_expression():
        return Expression(
            NormalInflow._shape_expression_schema, degree=2, amplitude=1.0
        )

    def eval(self):
        if self.counter < len(self.control_data):
            amplitude_value = self.control_data[self.counter]
            self.counter += 1
        else:
            amplitude_value = 0.0
        self._shape_expression.amplitude = amplitude_value
        return self._shape_expression


class ParabolicInflow:
    _shape_expression_schema = (
        "0.0",
        "amplitude*4.0/(A-B)/(A-B)*(x[0]-A)*(B-x[0])",
    )

    def __init__(self, control_data, left, right):
        self.counter = 1
        self.control_data = control_data

        self.left = left
        self.right = right
        self._shape_expression = self.generate_shape_expression()

    def generate_shape_expression(self):
        return Expression(
            ParabolicInflow._shape_expression_schema,
            degree=2,
            A=self.left,
            B=self.right,
            amplitude=1.0,
        )

    def eval(self):
        """ Evaluate the inflow profile at the current time step
        (tracked by the counter). This mutates the state of the
        object: the counter is incremented.
        """
        if self.counter < len(self.control_data):
            amplitude_value = self.control_data[self.counter]
            self.counter += 1
        else:
            amplitude_value = 0.0
        self._shape_expression.amplitude = amplitude_value
        return self._shape_expression
