#!/usr/bin/env python
"""
Solve numerically forward problem (initial-boundary value problem)
for wave equation
with zero Neyman boundary condition
using Finite-Difference Time-Domain method
and save animation
"""
import numpy as np

from base import derivative_x, derivative_y, in_circle, save_pressure
from constants import (
    DELTA_T,
    DELTA_X,
    DELTA_Y,
    N,
    PRESSURE_COEFFICIENT,
    T,
    VELOCITY_COEFFICIENT,
)


INITIAL_CONDITIONS_FILENAME: str = 'initial'
RESULT_FILENAME: str = 'forward_neyman_zero_boundary'


def update_neyman(press: np.ndarray, v_x: np.ndarray, v_y: np.ndarray) -> None:
    """Update pressure and velocity components for next timestep"""
    v_x += VELOCITY_COEFFICIENT * derivative_x(press)
    v_y += VELOCITY_COEFFICIENT * derivative_y(press)
    press[1:-1] += PRESSURE_COEFFICIENT * derivative_x(v_x)
    press[:, 1:-1] += PRESSURE_COEFFICIENT * derivative_y(v_y)

    # Neyman boundary condition
    press[0] += 2 * PRESSURE_COEFFICIENT * v_x[0]
    press[-1] += 2 * PRESSURE_COEFFICIENT * -v_x[-1]
    press[:, 0] += 2 * PRESSURE_COEFFICIENT * v_y[:, 0]
    press[:, -1] += 2 * PRESSURE_COEFFICIENT * -v_y[:, -1]


def main() -> None:
    """
    Solve numerical forward problem (initial-boundary value problem)
    for wave equation
    with zero Neuman boundary condition
    using Finite-Difference Time-Domain method
    and save animation
    """
    pressure: np.ndarray = np.fromfunction(
        lambda i, j: in_circle(
            i * DELTA_X, j * DELTA_Y, 0.75, 0.75, 0.2
        ) + in_circle(
            i * DELTA_X, j * DELTA_Y, 0.75, 0.25, 0.01
        ),
        (N, N)
    ).astype(float)

    save_pressure(pressure, INITIAL_CONDITIONS_FILENAME)

    velocity_x = np.zeros((pressure.shape[0] - 1, pressure.shape[1]))
    velocity_y = np.zeros((pressure.shape[0], pressure.shape[1] - 1))

    for _ in range(round(T / DELTA_T)):
        update_neyman(press=pressure, v_x=velocity_x, v_y=velocity_y)

    save_pressure(pressure, RESULT_FILENAME)


if __name__ == '__main__':
    main()
