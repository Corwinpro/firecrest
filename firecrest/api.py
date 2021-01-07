"""
Core API for the firecrest package.

This module represents the public API of the firecrest library.
It contains everything you should need as a user of the library.
No stability guarantees are made for imports from other modules or subpackages.
"""

from firecrest.mesh.boundaryelement import BSplineElement, LineElement
from firecrest.mesh.geometry import SimpleDomain

from firecrest.solvers.eigenvalue_tv_acoustic_solver import EigenvalueTVAcousticSolver
from firecrest.solvers.spectral_tv_acoustic_solver import SpectralTVAcousticSolver
from firecrest.solvers.unsteady_tv_acoustic_solver import UnsteadyTVAcousticSolver

__all__ = [
    "BSplineElement",
    "LineElement",
    "SimpleDomain",
    "EigenvalueTVAcousticSolver",
    "SpectralTVAcousticSolver",
    "UnsteadyTVAcousticSolver",
]
