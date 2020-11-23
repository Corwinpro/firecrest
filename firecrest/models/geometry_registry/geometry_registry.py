from .symmetric_printhead_assembler import SymmetricPrintheadGeometryAssembler
from .symmetric_injector import SymmetricInjectorAssembler

geometry_registry = {
    "symmetric_printhead": SymmetricPrintheadGeometryAssembler,
    "symmetric_injector": SymmetricInjectorAssembler,
}
