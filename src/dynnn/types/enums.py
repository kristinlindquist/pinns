from enum import Enum


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


class OdeSolver(Enum):
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

    @classmethod
    def _missing_(cls, value):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"{value} is not a valid OdeSolver")


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
