#!/usr/bin/env python
"""Solve numerically forward problem for wave equation
with constant Neuman boundary condition
using Finite-Difference Time-Domain method
and save image of result
"""
from typing import List, Tuple

import numpy as np
from scipy import integrate

from solvers.base import derivative_x, derivative_y
from constants import (
    C,
    DELTA_T,
    DELTA_X,
    DELTA_Y,
    N,
    NUMBER_OF_BASIS_FUNCTIONS,
    NUMBER_OF_BASIS_FUNCTIONS_BY_SPACE,
    NUMBER_OF_BASIS_FUNCTIONS_BY_TIME,
    NUMBER_OF_BORDERS,
    RHO,
    T,
)


def update_neyman(
    pressure: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    pressure_coefficient: float,
    velocity_coefficient: float,
    borders: np.ndarray,
) -> None:
    velocity_x += velocity_coefficient * derivative_x(pressure)
    velocity_y += velocity_coefficient * derivative_y(pressure)
    pressure[1:-1] += pressure_coefficient * derivative_x(velocity_x)
    pressure[:, 1:-1] += pressure_coefficient * derivative_y(velocity_y)

    # Neyman boundary condition
    pressure[0] = pressure[1] - 2 * DELTA_X * borders[0]
    pressure[-1] = pressure[-2] - 2 * DELTA_X * borders[1]
    pressure[:, 0] = pressure[:, 1] - 2 * DELTA_Y * borders[2]
    pressure[:, -1] = pressure[:, -2] - 2 * DELTA_Y * borders[3]


def build_system_of_linear_equations(
    pressure_coefficient: float,
    velocity_coefficient: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build system of linear equations for boundary control problem"""

    system_matrix: np.ndarray = np.zeros(
        (NUMBER_OF_BASIS_FUNCTIONS, NUMBER_OF_BASIS_FUNCTIONS),
    )

    for border_index in range(NUMBER_OF_BORDERS):
        for basis_function_space_index in range(
                NUMBER_OF_BASIS_FUNCTIONS_BY_SPACE,
        ):
            for basis_function_time_index in range(
                    NUMBER_OF_BASIS_FUNCTIONS_BY_TIME,
            ):
                basis_function_border: np.ndarray = np.sin(
                    basis_function_space_index * np.pi * np.linspace(0, 1, N),
                )
                borders: np.ndarray = np.array(
                    [
                        basis_function_border
                        if border_index == 0
                        else np.zeros((1, N)),

                        basis_function_border
                        if border_index == 1
                        else np.zeros((1, N)),

                        basis_function_border.transpose()
                        if border_index == 2
                        else np.zeros((N, 1)),

                        basis_function_border.transpose()
                        if border_index == 3
                        else np.zeros((N, 1)),
                    ],
                )

                solution_border: List[List[np.ndarray]] = (
                    solve_forward_problem(
                        borders,
                        pressure_coefficient,
                        velocity_coefficient,
                    )
                )


def main() -> None:
    pressure_coefficient: float = -DELTA_T * RHO * C**2 / DELTA_X
    velocity_coefficient: float = -DELTA_T / (DELTA_X * RHO)

    build_system_of_linear_equations(
        pressure_coefficient,
        velocity_coefficient,
    )


def solve_forward_problem(
    borders: np.ndarray,
    pressure_coefficient: float,
    velocity_coefficient: float,
) -> List[List[np.ndarray]]:
    pressure: np.ndarray = np.zeros((N + 2, N + 2))

    velocity_x: np.ndarray = np.zeros(
        (pressure.shape[0] - 1, pressure.shape[1]),
    )
    velocity_y: np.ndarray = np.zeros(
        (pressure.shape[0], pressure.shape[1] - 1),
    )

    solution_border = []

    for time_index in range(round((2 * T) / DELTA_T)):
        update_neyman(
            pressure,
            velocity_x,
            velocity_y,
            pressure_coefficient,
            velocity_coefficient,
            borders * np.sin(np.pi * time_index * DELTA_T / T),
        )
        solution_border.append(
            [pressure[1], pressure[-2], pressure[:, 1], pressure[:, -2]],
        )

    return solution_border


if __name__ == '__main__':
    main()
