"""
Common stuff for perfect matched layer with constant speed of sound solvers
"""
from typing import Tuple

import numpy as np  # type: ignore

import base
from constants import (
    DELTA_T,
    DELTA_X,
    DELTA_Y,
    N,
    PERFECT_MATCHED_LAYER_SIZE_X,
    PERFECT_MATCHED_LAYER_SIZE_Y,
    PRESSURE_COEFFICIENT,
    SIGMA_X_MAX,
    SIGMA_X_STAR_MAX,
    SIGMA_Y_MAX,
    SIGMA_Y_STAR_MAX,
    VELOCITY_COEFFICIENT,
)


def perfect_matched_layer_x(pressure_x: np.ndarray, sigma_max_x: int) -> None:
    """Apply Perfect Matched Layer method for x layer"""
    for i in range(PERFECT_MATCHED_LAYER_SIZE_X):
        coefficient = 1 - DELTA_T * sigma_max_x * (
            1 - i / PERFECT_MATCHED_LAYER_SIZE_X
        )
        pressure_x[i] *= coefficient
        pressure_x[-i - 1] *= coefficient


def perfect_matched_layer_y(pressure_y: np.ndarray, sigma_max_y: int) -> None:
    """Apply Perfect Matched Layer method for y layer"""
    for i in range(PERFECT_MATCHED_LAYER_SIZE_Y):
        coefficient = 1 - DELTA_T * sigma_max_y * (
            1 - i / PERFECT_MATCHED_LAYER_SIZE_Y
        )
        pressure_y[:, i] *= coefficient
        pressure_y[:, -i - 1] *= coefficient


def prepare_initial_conditions() -> Tuple[np.ndarray, np.ndarray]:
    """Prepare initial pressure for Perfect Matched Layer solvers"""
    pressure_x: np.ndarray = np.fromfunction(
        lambda i, j: base.in_circle(
            (i - PERFECT_MATCHED_LAYER_SIZE_X) * DELTA_X,
            (j - PERFECT_MATCHED_LAYER_SIZE_Y) * DELTA_Y,
            0.75,
            0.75,
            0.2,
        ) + base.in_circle(
            (i - PERFECT_MATCHED_LAYER_SIZE_X) * DELTA_X,
            (j - PERFECT_MATCHED_LAYER_SIZE_Y) * DELTA_Y,
            0.75,
            0.25,
            0.01,
        ),
        (
            N + 2 * PERFECT_MATCHED_LAYER_SIZE_X,
            N + 2 * PERFECT_MATCHED_LAYER_SIZE_Y,
        )
    ).astype(float) / 2
    return pressure_x, pressure_x.copy()


def save_pressure(
    # pylint: disable=bad-continuation
    pressure_x: np.ndarray,
    pressure_y: np.ndarray,
    filename: str,
) -> None:
    # pylint: enable=bad-continuation
    """
    Save pressure image for perfect matched layer solver

    Args:
        pressure from equations with Perfect Matched Layer for x coordinates
        pressure from equations with Perfect Matched Layer for y coordinates

    Returns:
        None
    """
    base.save_pressure(
        filename=filename,
        pressure=(pressure_x + pressure_y)[
            PERFECT_MATCHED_LAYER_SIZE_X:-PERFECT_MATCHED_LAYER_SIZE_X,
            PERFECT_MATCHED_LAYER_SIZE_Y:-PERFECT_MATCHED_LAYER_SIZE_Y,
        ],
    )


def update_constant_speed(
    # pylint: disable=bad-continuation
    pressure_x: np.ndarray,
    pressure_y: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
) -> None:
    # pylint: enable=bad-continuation
    """
    Update pressure and velocity components by time step
    for forward problem (initial value problem)
    for wave equation with constant speed of sound
    using Finite-Difference Time-Domain method
    and Perfect Matched Layer method

    Args:
        pressure array
        from equations with Perfect Matched Layer for x coordinate

        pressure array
        from equations with Perfect Matched Layer for y coordinate

        x component of velocity

        y component of velocity

    Returns:
        None
    """
    velocity_x += (
        VELOCITY_COEFFICIENT
        * base.derivative_x(pressure_x + pressure_y)
    )
    velocity_y += (
        VELOCITY_COEFFICIENT
        * base.derivative_y(pressure_x + pressure_y)
    )

    perfect_matched_layer_x(velocity_x, SIGMA_X_MAX)
    perfect_matched_layer_y(velocity_y, SIGMA_Y_MAX)

    pressure_x[1:-1] += PRESSURE_COEFFICIENT * base.derivative_x(velocity_x)
    pressure_y[:, 1:-1] += PRESSURE_COEFFICIENT * base.derivative_y(velocity_y)

    perfect_matched_layer_x(pressure_x, SIGMA_X_STAR_MAX)
    perfect_matched_layer_y(pressure_y, SIGMA_Y_STAR_MAX)
