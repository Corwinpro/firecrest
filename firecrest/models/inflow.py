import logging

from dolfin import Expression

logger = logging.getLogger(__name__)

UNIFORM_TYPE_LABEL = "uniform"
PARABOLIC_TYPE_LABEL = "parabolic"


class InflowModelFactory:
    _default_type = UNIFORM_TYPE_LABEL

    supported_types = (UNIFORM_TYPE_LABEL, PARABOLIC_TYPE_LABEL)

    def __init__(self, parameters):
        self.parameters = parameters

    def create_model(self, model_type=None):
        if model_type is None:
            model_type = self._default_type
            logger.warning(f"Using default type model: {model_type!r}")

        if model_type not in self.supported_types:
            raise ValueError(
                f"Model type {model_type!r} is not supported. Only "
                f"{self.supported_types} are supported."
            )

        if model_type == UNIFORM_TYPE_LABEL:
            return self.create_normal_inflow_model
        elif model_type == PARABOLIC_TYPE_LABEL:
            return self.create_parabolic_inflow_model

    def create_normal_inflow_model(self, history):
        return NormalInflow(history)

    def create_parabolic_inflow_model(self, history):
        if "left" not in self.parameters:
            raise ValueError(
                "The InflowModel parameters for the parabolic inflow"
                " profile must contain 'left' argument."
            )
        if "right" not in self.parameters:
            raise ValueError(
                "The InflowModel parameters for the parabolic inflow"
                " profile must contain 'right' argument."
            )
        left = self.parameters["left"]
        right = self.parameters["right"]
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
