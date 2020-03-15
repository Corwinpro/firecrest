from unittest import TestCase

from firecrest.models.inflow import InflowModelFactory, NormalInflow


class TestInflowModelFactory(TestCase):
    def setUp(self) -> None:
        self.length = 10
        self.control = [i for i in range(self.length)]

    def test_normal_inflow(self):
        inflow = NormalInflow(control_data=self.control)
        for i in range(self.length - 1):
            self.assertTupleEqual(inflow.eval(), (0.0, i + 1))
        for _ in range(100):
            self.assertTupleEqual(inflow.eval(), (0.0, 0.0))

    def test_factory(self):
        factory = InflowModelFactory()
        inflow = factory.create_normal_inflow_model(self.control)
        self.assertIsInstance(inflow, NormalInflow)
        self.assertEqual(1, inflow.counter)
        self.assertListEqual(inflow.control_data, self.control)
