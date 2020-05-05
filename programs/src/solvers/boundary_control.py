#!/usr/bin/env python
"""Solve numerically forward problem for wave equation
with constant Neuman boundary condition
using Finite-Difference Time-Domain method
and save image of result
"""
from typing import List, Tuple

import numpy as np

from solvers.base import derivative_x, derivative_y
from constants import (
    DELTA_T,
    DELTA_X,
    DELTA_Y,
    N,
    NUMBER_OF_BASIS_FUNCTIONS,
    NUMBER_OF_BASIS_FUNCTIONS_BY_SPACE,
    NUMBER_OF_BASIS_FUNCTIONS_BY_TIME,
    NUMBER_OF_BORDERS,
    PRESSURE_COEFFICIENT,
    T,
    VELOCITY_COEFFICIENT,
)


NUMBER_OF_TIME_STEPS: int = round((2 * T) / DELTA_T)


def build_border(edge_index: int, border_value: np.ndarray) -> np.ndarray:
    """Construct border
    with border_value on one of the edges and zero value on others

    Args:
        border_index: index of edge
        border_value:
        two dimensional array NUMBER_OF_TIME_STEPS x N - border value for edge

    Returns:
        three dimensional array NUMBER_OF_TIME_STEPS x NUMBER_OF_BORDERS x N
        - border value
    """
    return np.swapaxes(
        np.array(
            [
                border_value
                if edge_index == i
                else np.zeros((NUMBER_OF_TIME_STEPS, N))
                for i in range(NUMBER_OF_BORDERS)
            ],
        ),
        0,
        1,
    )


def build_system_of_linear_equations(
    pressure_coefficient: float,
    velocity_coefficient: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build system of linear equations for boundary control problem"""

    system_matrix: np.ndarray = np.zeros(
        (NUMBER_OF_BASIS_FUNCTIONS, NUMBER_OF_BASIS_FUNCTIONS),
    )

    borders, int_borders, solutions, int_solutions = solve_for_basis_borders()


def main() -> None:

    build_system_of_linear_equations()


def solve_for_basis_borders() -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
]:
    """Solve forward problem for basis functions

    Args:
        None

    Returns:
        Neuman control for basis functions

        Neuman control for basis functions integrated by time

        solution at the border for basis Neuman border functions

        solution at the border for basis Neuman border functions
        integrated by time
    """
    basis_borders: List[np.ndarray] = []
    integrated_basis_borders: List[np.ndarray] = []
    solution_for_basis_borders: List[np.ndarray] = []
    integrated_solution_for_basis_borders: List[np.ndarray] = []

    for border_index in range(NUMBER_OF_BORDERS):
        for basis_function_space_index in range(
                NUMBER_OF_BASIS_FUNCTIONS_BY_SPACE,
        ):
            for basis_function_time_index in range(
                    NUMBER_OF_BASIS_FUNCTIONS_BY_TIME,
            ):
                basis_function_space: np.ndarray = np.sin(
                    basis_function_space_index
                    * np.pi
                    * np.linspace(0, 1, N),
                )[np.newaxis]

                basis_borders.append(
                    build_border(
                        border_index,
                        np.sin(
                            np.pi
                            * basis_function_time_index
                            * np.arange(0, DELTA_T, NUMBER_OF_TIME_STEPS)
                            / T
                        )[:, np.newaxis]
                        @ basis_function_space,
                    ),
                )

                integrated_basis_borders.append(
                    build_border(
                        border_index,
                        T / (
                            np.pi
                            * basis_function_time_index
                        ) * (
                            1
                            - np.cos(
                                np.pi
                                * basis_function_time_index
                                * np.arange(0, DELTA_T, NUMBER_OF_TIME_STEPS)
                                / T
                            )
                        )[:, np.newaxis],
                    ),
                )

                solution_for_basis_borders.append(
                    solve_forward_problem(basis_borders[-1]),
                )
                integrated_solution_for_basis_borders.append(
                    solve_forward_problem(integrated_basis_borders[-1]),
                )


def solve_forward_problem(borders: np.ndarray) -> List[List[np.ndarray]]:
    pressure: np.ndarray = np.zeros((N + 2, N + 2))

    velocity_x: np.ndarray = np.zeros(
        (pressure.shape[0] - 1, pressure.shape[1]),
    )
    velocity_y: np.ndarray = np.zeros(
        (pressure.shape[0], pressure.shape[1] - 1),
    )

    solution_border = []

    for time_index in range(NUMBER_OF_TIME_STEPS):
        update_neyman(pressure, velocity_x, velocity_y, borders[time_index])
        solution_border.append(
            [pressure[1], pressure[-2], pressure[:, 1], pressure[:, -2]],
        )

    return solution_border


def update_neyman(
    pressure: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    borders: np.ndarray,
) -> None:
    velocity_x += VELOCITY_COEFFICIENT * derivative_x(pressure)
    velocity_y += VELOCITY_COEFFICIENT * derivative_y(pressure)
    pressure[1:-1] += PRESSURE_COEFFICIENT * derivative_x(velocity_x)
    pressure[:, 1:-1] += PRESSURE_COEFFICIENT * derivative_y(velocity_y)

    # Neyman boundary condition
    pressure[0] = pressure[1] - 2 * DELTA_X * borders[0]
    pressure[-1] = pressure[-2] - 2 * DELTA_X * borders[1]
    pressure[:, 0] = pressure[:, 1] - 2 * DELTA_Y * borders[2][:, np.newaxis]
    pressure[:, -1] = pressure[:, -2] - 2 * DELTA_Y * borders[3][:, np.newaxis]


if __name__ == '__main__':
    main()
