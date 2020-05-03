#!/usr/bin/env python
"""Solve numerically forward problem for wave equation
with zero Neyman boundary
using Finite-Difference Time-Domain method
and save animation
"""
import numpy as np

from .base import derivative_x, derivative_y, in_circle, save_pressure
from constants import C, DELTA_T, DELTA_X, DELTA_Y, N, RHO, T


INITIAL_CONDITIONS_FILENAME: str = 'initial'
RESULT_FILENAME: str = 'forward_neyman_zero_boundary'


def update_neyman(
    pressure: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    pressure_coefficient: float,
    velocity_coefficient: float,
) -> None:
    velocity_x += velocity_coefficient * derivative_x(pressure)
    velocity_y += velocity_coefficient * derivative_y(pressure)
    pressure[1:-1] += pressure_coefficient * derivative_x(velocity_x)
    pressure[:, 1:-1] += pressure_coefficient * derivative_y(velocity_y)

    # Neyman boundary condition
    pressure[0] += 2 * pressure_coefficient * velocity_x[0]
    pressure[-1] += 2 * pressure_coefficient * -velocity_x[-1]
    pressure[:, 0] += 2 * pressure_coefficient * velocity_y[:, 0]
    pressure[:, -1] += 2 * pressure_coefficient * -velocity_y[:, -1]


def main() -> None:
    pressure_coefficient: float = -DELTA_T * RHO * C**2 / DELTA_X
    velocity_coefficient: float = -DELTA_T / (DELTA_X * RHO)

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

    for i in range(round(T / DELTA_T)):
        update_neyman(
            pressure,
            velocity_x,
            velocity_y,
            pressure_coefficient,
            velocity_coefficient,
        )

    save_pressure(pressure, RESULT_FILENAME)


if __name__ == '__main__':
    main()
