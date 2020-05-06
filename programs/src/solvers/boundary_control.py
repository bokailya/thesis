#!/usr/bin/env python
"""
Solve numerically boundary control problem for wave equation
with Neuman boundary condition and save image of result.
For forward problem used Finite-Difference Time-Domain method.
"""
from typing import Tuple

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
    X_LENGTH,
    Y_LENGTH,
)


NUMBER_OF_TIME_STEPS: int = round((2 * T) / DELTA_T)

TARGET_POINT: np.ndarray = np.array([0.5, 0.5])


def build_border(edge_index: int, border_value: np.ndarray) -> np.ndarray:
    """
    Construct border
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


def build_system_of_linear_equations() -> Tuple[np.ndarray, np.ndarray]:
    """Build system of linear equations for boundary control problem"""

    borders, int_borders, solutions, int_solutions = solve_for_basis_borders()
    even_borders: np.ndarray = (borders + np.flip(borders, 1)) / 2
    even_solutions: np.ndarray = (solutions + np.flip(solutions, 1)) / 2

    system_matrix: np.ndarray = np.zeros(
        (NUMBER_OF_BASIS_FUNCTIONS, NUMBER_OF_BASIS_FUNCTIONS),
    )

    for i in range(NUMBER_OF_BASIS_FUNCTIONS):
        for j in range(NUMBER_OF_BASIS_FUNCTIONS):
            system_matrix[i, j] = sum(
                np.trapz(
                    dx=DELTA_Y if edge_index < 2 else DELTA_X,
                    y=np.trapz(
                        axis=0,
                        dx=DELTA_T,
                        y=(
                            even_solutions[j, :, edge_index]
                            * int_borders[i, :, edge_index]
                            - even_borders[j, :, edge_index]
                            * int_solutions[i, :, edge_index]
                        ),
                    ),
                )
                for edge_index in range(NUMBER_OF_BORDERS)
            )

    system_right_hand_side_vector: np.ndarray = np.zeros(
        (NUMBER_OF_BASIS_FUNCTIONS, 1),
    )

    x_variable: np.ndarray = np.linspace(0, X_LENGTH, N)
    y_variable: np.ndarray = np.linspace(0, Y_LENGTH, N)

    target: np.ndarray = np.array(
        [
            target_function(np.array(0), y_variable),
            target_function(np.array(Y_LENGTH), y_variable),
            target_function(x_variable, np.array(0)),
            target_function(x_variable, np.array(X_LENGTH)),
        ],
    )
    target_normal_derivative: np.ndarray = np.array(
        [
            target_function(np.array(0), y_variable),
            target_function(np.array(Y_LENGTH), y_variable),
            target_function(x_variable, np.array(0)),
            target_function(x_variable, np.array(X_LENGTH)),
        ],
    )
    for i in range(NUMBER_OF_BASIS_FUNCTIONS):
        system_right_hand_side_vector[i] = sum(
            np.trapz(
                dx=DELTA_Y if edge_index < 2 else DELTA_X,
                y=(
                    np.trapz(
                        axis=0,
                        dx=DELTA_T,
                        y=int_borders[i, :, edge_index],
                    )
                    * target[edge_index]
                    - np.trapz(
                        axis=0,
                        dx=DELTA_T,
                        y=int_solutions[i, :, edge_index],
                    )
                    * target_normal_derivative
                )
            )
            for edge_index in range(NUMBER_OF_BORDERS)
        )

    return system_matrix, system_right_hand_side_vector


def main() -> None:
    """
    Solve numerically boundary control problem for wave equation
    with Neuman boundary condition and save image of result.
    For forward problem used Finite-Difference Time-Domain method.

    Args:
        None

    Returns:
        None
    """
    build_system_of_linear_equations()


def solve_for_basis_borders() -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
]:
    """
    Solve forward problem for basis functions

    Args:
        None

    Returns:
        Neuman control for basis functions

        Neuman control for basis functions integrated by time

        solution at the border for basis Neuman border functions

        solution at the border for basis Neuman border functions
        integrated by time
    """
    basis_borders: np.ndarray = np.zeros(
        (
            NUMBER_OF_BASIS_FUNCTIONS,
            NUMBER_OF_TIME_STEPS,
            NUMBER_OF_BORDERS,
            N,
        ),
    )
    integrated_basis_borders: np.ndarray = basis_borders.copy()
    solution_for_basis_borders: np.ndarray = basis_borders.copy()
    integrated_solution_for_basis_borders: np.ndarray = basis_borders.copy()

    basis_function_index: int = 0

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

                basis_borders[basis_function_index] = build_border(
                    border_index,
                    np.sin(
                        np.pi
                        * basis_function_time_index
                        * np.arange(0, DELTA_T, NUMBER_OF_TIME_STEPS)
                        / T
                    )[:, np.newaxis]
                    @ basis_function_space,
                )

                integrated_basis_borders[basis_function_index] = build_border(
                    border_index,
                    T
                    / (np.pi * basis_function_time_index)
                    * (
                        1
                        - np.cos(
                            np.pi
                            * basis_function_time_index
                            * np.arange(0, DELTA_T, NUMBER_OF_TIME_STEPS)
                            / T
                        )
                    )[:, np.newaxis]
                    @ basis_function_space,
                )

                solution_for_basis_borders[
                    basis_function_index,
                ] = solve_forward_problem(basis_borders[-1])

                integrated_solution_for_basis_borders[
                    basis_function_index,
                ] = solve_forward_problem(integrated_basis_borders[-1])


def solve_forward_problem(borders: np.ndarray) -> np.ndarray:
    pressure: np.ndarray = np.zeros((N + 2, N + 2))

    velocity_x: np.ndarray = np.zeros(
        (pressure.shape[0] - 1, pressure.shape[1]),
    )
    velocity_y: np.ndarray = np.zeros(
        (pressure.shape[0], pressure.shape[1] - 1),
    )

    solution_border: np.ndarray = np.zeros(
        (NUMBER_OF_TIME_STEPS, NUMBER_OF_BORDERS, N),
    )

    for time_index in range(NUMBER_OF_TIME_STEPS):
        update_neyman(pressure, velocity_x, velocity_y, borders[time_index])
        solution_border[time_index] = [
            pressure[1],
            pressure[-2],
            pressure[:, 1].transpose(),
            pressure[:, -2].transpose(),
        ]

    return solution_border


def target_function(x: np.ndarray, y: np.ndarray):
    """
    Target value for wave function u(., T)

    Args:
        Array of x coordinates

    Returns:
        Target function values
    """
    return np.log(np.sqrt((x - TARGET_POINT[0])**2 + (y - TARGET_POINT[1])**2))


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
