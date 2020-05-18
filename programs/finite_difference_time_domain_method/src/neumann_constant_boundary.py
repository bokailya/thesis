#!/usr/bin/env python
"""
Solve numerically forward problem (inital-boundary value problem)
for wave equation
with constant Neuman boundary condition
using Finite-Difference Time-Domain method
and save image of result
"""
import numpy as np  # type: ignore

from base import derivative_x, derivative_y, save_pressure
from constants import (
    DELTA_T,
    DELTA_X,
    DELTA_Y,
    N,
    PRESSURE_COEFFICIENT,
    T,
    VELOCITY_COEFFICIENT,
)


FILENAME: str = 'forward_neyman_constant_boundary'


def update_neyman(press: np.ndarray, v_x: np.ndarray, v_y: np.ndarray) -> None:
    """Update pressure and velocity values for next time step"""
    v_x += VELOCITY_COEFFICIENT * derivative_x(press)
    v_y += VELOCITY_COEFFICIENT * derivative_y(press)
    press[1:-1] += PRESSURE_COEFFICIENT * derivative_x(v_x)
    press[:, 1:-1] += PRESSURE_COEFFICIENT * derivative_y(v_y)

    # Neumann boundary condition
    press[0] = press[1] - 2 * DELTA_X
    press[-1] = press[-2] - 2 * DELTA_X
    press[:, 0] = press[:, 1] - 2 * DELTA_Y
    press[:, -1] = press[:, -2] - 2 * DELTA_Y


def main() -> None:
    """
    Save forward problem (inital-boundary value problem) for wave equation
    with constant Neuman boundary condition
    using Finite-Difference Time-Domain method
    """
    pressure: np.ndarray = np.zeros((N + 2, N + 2))

    velocity_x = np.zeros((pressure.shape[0] - 1, pressure.shape[1]))
    velocity_y = np.zeros((pressure.shape[0], pressure.shape[1] - 1))

    for _ in range(round(T / DELTA_T)):
        update_neyman(press=pressure, v_x=velocity_x, v_y=velocity_y)

    save_pressure(pressure[1:-1, 1:-1], FILENAME)


if __name__ == '__main__':
    main()
