import dolfin as dolf
import ufl


def is_numeric_argument(arg):
    _is_numeric = isinstance(arg, (int, float))
    return _is_numeric


def is_numeric_tuple(arg):
    if isinstance(arg, tuple):
        if all([is_numeric_argument(el) for el in arg]):
            return True
    return False


def is_dolfin_exp(arg):
    return isinstance(
        arg,
        (
            dolf.function.constant.Constant,
            dolf.function.expression.Expression,
            dolf.function.function.Function,
            ufl.tensors.ComponentTensor,
            ufl.algebra.Product,
        ),
    )
