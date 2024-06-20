from enum import Enum
from torchdyn.numerics.odeint import odeint, odeint_symplectic
from typing import Callable


class GeneratorType(Enum):
    """
    Enum for the type of EOM generator function
    """

    LAGRANGIAN = 1
    HAMILTONIAN = 2

    @classmethod
    def _missing_(cls, value):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"{value} is not a valid GeneratorType")


class OdeSolverType(Enum):
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

    @classmethod
    def _missing_(cls, value):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"{value} is not a valid OdeSolverType")


class VectorField(Enum):
    """
    Type of vector field to learn
    """

    SOLENOIDAL = "solenoidal"
    CONSERVATIVE = "conservative"
    PORT = "port"
    # bad name; means solenoidal and conservative
    # https://en.wikipedia.org/wiki/Helmholtz_decomposition
    HELMHOLTZ = "helmholtz"
    NONE = "none"
