#!/usr/bin/env python
"""Solve numerically forward problem for wave equation
with Perfectly Matched Layer
using Finite-Difference Time-Domain method
and save animation
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


INITIAL_CONDITIONS_FILENAME: str = 'initial'
RESULT_FILENAME: str = 'forward_neyman_zero_boundary'

X_LENGTH: int = 1
Y_LENGTH: int = 1
C: int = 1
RHO: int = 1
T: float = 0.5

HALF_MAX_ACCURACY: int = 6
N: int = 512
DELTA_X: float = X_LENGTH / (N - 1)
DELTA_Y: float = Y_LENGTH / (N - 1)
DELTA_T: float = DELTA_X / (4 * C)


def coefficients(half_accuracy: int) -> np.ndarray:
    accuracy: int = 2 * half_accuracy
    return np.delete(
        np.linalg.solve(
            [
                np.concatenate(
                    (
                        np.arange(-half_accuracy + 0.5, 0),
                        (0,),
                        np.arange(0.5, half_accuracy),
                    ),
                )
                **
                i
                for i in range(accuracy + 1)
            ],
            [0, 1] + [0] * (accuracy - 1),
        ),
        half_accuracy,
    )


derivative_approximation_coefficients = [
    coefficients(half_accuracy)
    for half_accuracy in range(1, HALF_MAX_ACCURACY + 1)
]


def derivative_x(function: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.zeros((function.shape[0] - 1, function.shape[1]))
    for i in range(function.shape[0] - 1):
        accuracy: int = min(12, 2 * i + 2, (function.shape[0] - i - 1) * 2)
        half_accuracy: int = accuracy // 2
        for j in range(accuracy):
            result[i] += derivative_approximation_coefficients[
                half_accuracy - 1
            ][j] * function[i + j + 1 - half_accuracy]
    return result


def derivative_y(function: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.zeros((function.shape[0], function.shape[1] - 1))
    for i in range(function.shape[1] - 1):
        accuracy: int = min(12, 2 * i + 2, (function.shape[1] - i - 1) * 2)
        half_accuracy: int = accuracy // 2
        for j in range(accuracy):
            result[:, i] += derivative_approximation_coefficients[
                half_accuracy - 1
            ][j] * function[:, i + j + 1 - half_accuracy]
    return result


def in_circle(x: float, y: float, x0: float, y0: float, radius: float) -> bool:
    return (x - x0)**2 + (y - y0)**2 < radius**2


def save_pressure(pressure: np.ndarray, filename: str) -> None:
    plt.imshow(pressure, vmin=-1, vmax=1, extent=[0, X_LENGTH, 0, Y_LENGTH])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(Path('..') / 'images' / f'{filename}.png')


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
