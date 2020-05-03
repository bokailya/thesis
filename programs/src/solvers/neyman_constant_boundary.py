#!/usr/bin/env python
"""Solve numerically forward problem for wave equation
with constant Neuman boundary condition
using Finite-Difference Time-Domain method
and save image of result
"""
import numpy as np

from .base import derivative_x, derivative_y, save_pressure
from constants import C, DELTA_T, DELTA_X, DELTA_Y, N, RHO, T


FILENAME: str = 'forward_neyman_constant_boundary'


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
    pressure[0] = pressure[1] - 2 * DELTA_X
    pressure[-1] = pressure[-2] - 2 * DELTA_X
    pressure[:, 0] = pressure[:, 1] - 2 * DELTA_Y
    pressure[:, -1] = pressure[:, -2] - 2 * DELTA_Y


def main() -> None:
    pressure_coefficient: float = -DELTA_T * RHO * C**2 / DELTA_X
    velocity_coefficient: float = -DELTA_T / (DELTA_X * RHO)

    pressure: np.ndarray = np.zeros((N + 2, N + 2))

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

    save_pressure(pressure[1:-1, 1:-1], FILENAME)


if __name__ == '__main__':
    main()
