from unittest import TestCase

import dolfin as dolf

from firecrest.api import BSplineElement, SimpleDomain


class TestGeometryIntegration(TestCase):
    def test_domain_measures(self):
        # The first element is marked as "0"
        BSplineElement(
            [[0.0, 0.0], [0.1, 0.1], [0.5, 0.1]], bcond={}, degree=1, el_size=0.02
        )
        # The second element is marked as "1"
        boundary1 = BSplineElement(
            [[0.0, 0.0], [0.1, 0.1], [0.5, 0.1]], bcond={}, degree=1, el_size=0.02
        )
        # The third element is marked as "2"
        boundary2 = BSplineElement(
            [[0.5, 0.1], [0.5, 0.0], [0.25, -0.1], [0.0, 0.0]],
            bcond={},
            degree=2,
            el_size=0.01,
        )
        domain_boundaries = (boundary1, boundary2)
        domain = SimpleDomain(domain_boundaries)

        # Measure the length of the first element, which does not exist in the
        # geometry
        first = dolf.assemble(dolf.Constant(1.0) * domain.ds((1,)))
        # Measure the lengths of the other elements and the total surface of the
        # 2D geometry
        second = dolf.assemble(dolf.Constant(1.0) * domain.ds((2,)))
        third = dolf.assemble(dolf.Constant(1.0) * domain.ds((3,)))
        volume = dolf.assemble(dolf.Constant(1.0) * domain.dx())

        self.assertAlmostEqual(first, 0.0)
        self.assertAlmostEqual(second, 0.5414213562373094)
        self.assertAlmostEqual(third, 0.5952083648813487)
        self.assertAlmostEqual(volume, 0.06373511788850006)
