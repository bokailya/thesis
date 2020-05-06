#!/usr/bin/env python
"""
Solve numerically boundary control problem for wave equation
with Neuman boundary condition and save image of result.
For forward problem used Finite-Difference Time-Domain method.
"""
from typing import Tuple

import numpy as np

from solvers.base import derivative_x, derivative_y, save_pressure
from constants import (
    BoundaryIndex,
    DELTA_T,
    DELTA_X,
    DELTA_Y,
    N,
    NUMBER_OF_BASIS_FUNCTIONS,
    NUMBER_OF_BASIS_FUNCTIONS_BY_SPACE,
    NUMBER_OF_BASIS_FUNCTIONS_BY_TIME,
    NUMBER_OF_BOUNDARY_EDGES,
    PRESSURE_COEFFICIENT,
    T,
    VELOCITY_COEFFICIENT,
    X_LENGTH,
    Y_LENGTH,
)


NUMBER_OF_TIME_STEPS: int = round((2 * T) / DELTA_T)

RESULT_FILENAME: str = 'boundary_control'
TARGET_POINT: np.ndarray = np.array([0.5, 0.5])


def build_boundary(edge_index: int, boundary_value: np.ndarray) -> np.ndarray:
    """
    Construct boundary
    with boundary_value on one of the edges and zero value on others

    Args:
        boundary_index: index of edge
        boundary_value:
        two dimensional array NUMBER_OF_TIME_STEPS x N
        - boundary value for edge

    Returns:
        three dimensional array
        NUMBER_OF_TIME_STEPS x NUMBER_OF_BOUNDARY_EDGES x N
        - boundary value
    """
    return np.swapaxes(
        np.array(
            [
                boundary_value
                if edge_index == i
                else np.zeros((NUMBER_OF_TIME_STEPS, N))
                for i in range(NUMBER_OF_BOUNDARY_EDGES)
            ],
        ),
        0,
        1,
    )


def build_boundary_control() -> np.ndarray:
    """
    Build Neuman boundary control generating target function

    Args:
        None

    Returns:
        Neuman boundary control generating target function
    """
    basis_boundaries, basis_solutions = solve_for_basis_boundaries()
    system_matrix, system_right_hand_side_vector = (
        build_system_of_linear_equations(basis_boundaries, basis_solutions)
    )
    boundary_control_coefficients = np.linalg.lstsq(
        system_matrix,
        system_right_hand_side_vector,
    )[0]
    return sum(
        boundary_control_coefficients[i] * basis_boundaries[i]
        for i in range(NUMBER_OF_BASIS_FUNCTIONS)
    )


def build_system_of_linear_equations(
    basis_boundaries: np.ndarray,
    basis_solutions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build system of linear equations for boundary control problem

    Args:
        None

    Returns:
       matrix of linear system matrix
       right hand side vector of linear system of equations
    """

    integrated_basis_boundaries, integrated_basis_solutions = (
        solve_for_integrated_basis_boundaries()
    )
    even_basis_boundaries: np.ndarray = (
        basis_boundaries + np.flip(basis_boundaries, 1)
    ) / 2
    even_basis_solutions: np.ndarray = (
        basis_solutions + np.flip(basis_solutions, 1)
    ) / 2

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
                            even_basis_solutions[j, :, edge_index]
                            * integrated_basis_boundaries[i, :, edge_index]
                            - even_basis_boundaries[j, :, edge_index]
                            * integrated_basis_solutions[i, :, edge_index]
                        ),
                    ),
                )
                for edge_index in range(NUMBER_OF_BOUNDARY_EDGES)
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
            target_normal_derivative(np.array(0), y_variable),
            target_normal_derivative(np.array(Y_LENGTH), y_variable),
            target_normal_derivative(x_variable, np.array(0)),
            target_normal_derivative(x_variable, np.array(X_LENGTH)),
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
                        y=integrated_basis_boundaries[i, :, edge_index],
                    )
                    * target[edge_index]
                    - np.trapz(
                        axis=0,
                        dx=DELTA_T,
                        y=integrated_basis_solutions[i, :, edge_index],
                    )
                    * target_normal_derivative
                )
            )
            for edge_index in range(NUMBER_OF_BOUNDARY_EDGES)
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
    save_pressure(
        solve_forward_problem(build_boundary_control())[0],
        RESULT_FILENAME,
    )


def solve_for_basis_boundaries() -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve forward problem for basis functions

    Args:
        None

    Returns:
        Neuman control for basis functions

        solution at the boundary for basis Neuman boundary functions
    """
    basis_boundaries: np.ndarray = np.zeros(
        (
            NUMBER_OF_BASIS_FUNCTIONS,
            NUMBER_OF_TIME_STEPS,
            NUMBER_OF_BOUNDARY_EDGES,
            N,
        ),
    )
    solution_for_basis_boundaries: np.ndarray = basis_boundaries.copy()

    basis_function_index: int = 0

    for boundary_index in range(NUMBER_OF_BOUNDARY_EDGES):
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

                basis_boundaries[basis_function_index] = build_boundary(
                    boundary_index,
                    np.sin(
                        np.pi
                        * basis_function_time_index
                        * np.arange(0, DELTA_T, NUMBER_OF_TIME_STEPS)
                        / T
                    )[:, np.newaxis]
                    @ basis_function_space,
                )

                solution_for_basis_boundaries[
                    basis_function_index,
                ] = solve_forward_problem(basis_boundaries[-1])[1]

    return basis_boundaries, solution_for_basis_boundaries


def solve_for_integrated_basis_boundaries() -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve forward problem for basis functions

    Args:
        None

    Returns:
        Neuman control for basis functions integrated by time

        solution at the boundary for basis Neuman boundary functions
        integrated by time
    """
    integrated_basis_boundaries: np.ndarray = np.zeros(
        (
            NUMBER_OF_BASIS_FUNCTIONS,
            NUMBER_OF_TIME_STEPS,
            NUMBER_OF_BOUNDARY_EDGES,
            N,
        ),
    )
    integrated_solution_for_basis_boundaries: np.ndarray = (
        integrated_basis_boundaries.copy()
    )

    basis_function_index: int = 0

    for boundary_index in range(NUMBER_OF_BOUNDARY_EDGES):
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

                integrated_basis_boundaries[basis_function_index] = (
                    build_boundary(
                        boundary_index,
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
                )

                integrated_solution_for_basis_boundaries[
                    basis_function_index,
                ] = solve_forward_problem(integrated_basis_boundaries[-1])[1]

    return (
        integrated_basis_boundaries,
        integrated_solution_for_basis_boundaries,
    )


def solve_forward_problem(boundaries: np.ndarray) -> np.ndarray:
    """
    Solve forward problem (initial-boundary value problem) for wave equation
    using Finite-Difference Time-Domain method

    Args:
        Neuman boundary control
    """
    pressure: np.ndarray = np.zeros((N + 2, N + 2))

    velocity_x: np.ndarray = np.zeros(
        (pressure.shape[0] - 1, pressure.shape[1]),
    )
    velocity_y: np.ndarray = np.zeros(
        (pressure.shape[0], pressure.shape[1] - 1),
    )

    solution_boundary: np.ndarray = np.zeros(
        (NUMBER_OF_TIME_STEPS, NUMBER_OF_BOUNDARY_EDGES, N),
    )

    for time_index in range(NUMBER_OF_TIME_STEPS):
        update_neyman(pressure, velocity_x, velocity_y, boundaries[time_index])
        solution_boundary[time_index] = [
            pressure[1],
            pressure[-2],
            pressure[:, 1].transpose(),
            pressure[:, -2].transpose(),
        ]

    return pressure, solution_boundary


def target_function(x: np.ndarray, y: np.ndarray):
    """
    Target value for wave function u(., T)

    Args:
        Array of x coordinates

    Returns:
        Target function values
    """
    return np.log(np.sqrt((x - TARGET_POINT[0])**2 + (y - TARGET_POINT[1])**2))


def target_normal_derivative(x: np.ndarray, y: np.ndarray):
    """
    Target value for normal derivative of wave function du(., T) / dn

    Args:
        Array of x coordinates

    Returns:
        Target function values
    """
    if (x == 0).all():
        return -(x - TARGET_POINT[0]) / (
            (x - TARGET_POINT[0])**2
            + (y - TARGET_POINT[1])**2
        )
    elif (x == X_LENGTH).all():
        return (x - TARGET_POINT[0]) / (
            (x - TARGET_POINT[0])**2
            + (y - TARGET_POINT[1])**2
        )
    elif (y == 0).all():
        return -(y - TARGET_POINT[1]) / (
            (x - TARGET_POINT[0])**2
            + (y - TARGET_POINT[1])**2
        )
    elif (y == Y_LENGTH).all():
        return (y - TARGET_POINT[1]) / (
            (x - TARGET_POINT[0])**2
            + (y - TARGET_POINT[1])**2
        )
    else:
        raise ValueError('One of arguments should have boundary value')


def update_neyman(
    pressure: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    boundaries: np.ndarray,
) -> None:
    velocity_x += VELOCITY_COEFFICIENT * derivative_x(pressure)
    velocity_y += VELOCITY_COEFFICIENT * derivative_y(pressure)
    pressure[1:-1] += PRESSURE_COEFFICIENT * derivative_x(velocity_x)
    pressure[:, 1:-1] += PRESSURE_COEFFICIENT * derivative_y(velocity_y)

    # Neyman boundary condition
    pressure[0] = pressure[1] - 2 * DELTA_X * boundaries[
        BoundaryIndex.MINIMAL_X.value,
    ]
    pressure[-1] = pressure[-2] - 2 * DELTA_X * boundaries[
        BoundaryIndex.MAXIMAL_X.value,
    ]
    pressure[:, 0] = pressure[:, 1] - 2 * DELTA_Y * boundaries[
        BoundaryIndex.MINIMAL_Y.value,
    ][:, np.newaxis]
    pressure[:, -1] = pressure[:, -2] - 2 * DELTA_Y * boundaries[
        BoundaryIndex.MAXIMAL_Y.value,
    ][:, np.newaxis]


if __name__ == '__main__':
    main()
