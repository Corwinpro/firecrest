from unittest import TestCase
import numpy as np

from firecrest.models.inflow import (
    InflowModelFactory,
    NormalInflow,
    ParabolicInflow,
)


class TestInflowModelFactory(TestCase):
    def setUp(self) -> None:
        self.length = 10
        self.control = [i for i in range(self.length)]

    def test_normal_inflow(self):
        inflow = NormalInflow(control_data=self.control)
        for i in range(self.length - 1):
            np.testing.assert_equal(
                inflow.eval()(0.0, 0.0), np.array([0.0, i + 1])
            )
        for _ in range(100):
            np.testing.assert_equal(
                inflow.eval()(0.0, 0.0), np.array([0.0, 0.0])
            )

    def test_parabolic_inflow(self):
        inflow = ParabolicInflow(control_data=self.control, left=3.0, right=4.0)
        for i in range(self.length - 1):
            expression = inflow.eval()
            np.testing.assert_equal(
                expression(3.0, 0.0), np.array([0.0, 0.0])
            )
            np.testing.assert_equal(
                expression(4.0, 0.0), np.array([0.0, 0.0])
            )
            np.testing.assert_equal(
                expression(3.5, 0.0), np.array([0.0, i + 1])
            )
        for _ in range(100):
            np.testing.assert_equal(
                inflow.eval()(0.0, 0.0), np.array([0.0, 0.0])
            )

    def test_factory(self):
        factory = InflowModelFactory()
        inflow = factory.create_normal_inflow_model(self.control)
        self.assertIsInstance(inflow, NormalInflow)
        self.assertEqual(1, inflow.counter)
        self.assertListEqual(inflow.control_data, self.control)
