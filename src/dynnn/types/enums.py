from enum import Enum, IntEnum
from pydantic.fields import Field
from torchdyn.numerics.odeint import odeint, odeint_symplectic
from typing import Any, Callable, Type


class ListableEnum(IntEnum):

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def _missing_(cls, value):
        try:
            if isinstance(value, str):
                return cls[value.upper()]
            if isinstance(value, int):
                return cls(value)
            if isinstance(value, float):
                return cls(int(value))
        except KeyError:
            raise ValueError(f"{value} is not a valid {cls.__name__}")


class GeneratorType(ListableEnum):
    """
    Enum for the type of EOM generator function
    """

    LAGRANGIAN = 1
    HAMILTONIAN = 2


class OdeSolverType(ListableEnum):
    """
    Enum for the type of ODE solver to use

    On choosing an ODE solver: https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
    """

    TSIT5 = 1
    DOPRI5 = 2
    ALF = 3
    EULER = 4
    MIDPOINT = 5
    RK4 = 6
    IEULER = 7
    SYMPLECTIC = 8

    def solve(self, *args, solver, **kwargs):
        """
        Solve ODE based on solver type
        """
        if self == OdeSolverType.SYMPLECTIC:
            return odeint_symplectic(*args, solver="tsit5", **kwargs)

        return odeint(*args, solver=solver, **kwargs)


class VectorField(ListableEnum):
    """
    Type of vector field to learn
    """

    SOLENOIDAL = 1
    CONSERVATIVE = 2
    PORT = 3
    # bad name; means solenoidal and conservative
    # https://en.wikipedia.org/wiki/Helmholtz_decomposition
    HELMHOLTZ = 4
    NONE = 5


def enum_validator(cls, v: Any, field: Field):
    """
    Pydantic validator for Enum fields
    """
    if isinstance(field.annotation, type) and issubclass(field.annotation, Enum):
        try:
            return field.annotation(v)
        except ValueError:
            if hasattr(field.annotation, "__missing__"):
                return field.annotation.__missing__(field.annotation, v)
            raise
    return v
