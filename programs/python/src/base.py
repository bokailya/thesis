"""Common settings and functions"""
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from matplotlib.animation import FFMpegWriter  # type: ignore

from constants import (
    DELTA_X,
    DELTA_Y,
    HALF_MAX_ACCURACY,
    PERFECT_MATCHED_LAYER_SIZE_X,
    PERFECT_MATCHED_LAYER_SIZE_Y,
    X_LENGTH,
    Y_LENGTH,
)


def _coefficients(half_accuracy: int) -> np.ndarray:
    """Find coefficients for numerical differentiantion

    Args:
        half_accuracy: Half of numerical differentiantion accuracy

    Returns:
        Array of coefficients for numerical differentiantion
    """
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


_numerical_differentiation_coefficients = [
    _coefficients(half_accuracy)
    for half_accuracy in range(1, HALF_MAX_ACCURACY + 1)
]


def derivative_x(function: np.ndarray) -> np.ndarray:
    """Find derivative with respect to x numerically

    Args:
        function: Two-dimensional array of function values

    Returns:
        Two dimensional array with found derivative
    """
    result: np.ndarray = np.zeros((function.shape[0] - 1, function.shape[1]))
    for i in range(function.shape[0] - 1):
        half_accuracy: int = min(
            HALF_MAX_ACCURACY,
            i + 1,
            function.shape[0] - i - 1,
        )
        for j in range(half_accuracy * 2):
            result[i] += _numerical_differentiation_coefficients[
                half_accuracy - 1
            ][j] * function[i + j + 1 - half_accuracy]
    return result / DELTA_X


def derivative_y(function: np.ndarray) -> np.ndarray:
    """Find derivative with respect to y numerically

    Args:
        function: Two-dimensional array of function values

    Returns:
        Two dimensional array with found derivative
    """
    result: np.ndarray = np.zeros((function.shape[0], function.shape[1] - 1))
    for i in range(function.shape[1] - 1):
        accuracy: int = min(12, 2 * i + 2, (function.shape[1] - i - 1) * 2)
        half_accuracy: int = accuracy // 2
        for j in range(accuracy):
            result[:, i] += _numerical_differentiation_coefficients[
                half_accuracy - 1
            ][j] * function[:, i + j + 1 - half_accuracy]
    return result / DELTA_Y


def in_circle(
    # pylint: disable=bad-continuation
    x_coordinate: float,
    y_coordinate: float,
    circle_center_x_coordinate: float,
    circle_center_y_coordinate: float,
    circle_radius: float,
) -> bool:
    # pylint: enable=bad-continuation
    """Check if point lays inside circle"""
    return (
        (x_coordinate - circle_center_x_coordinate)**2
        + (y_coordinate - circle_center_y_coordinate)**2
        < circle_radius**2
    )


def save_pressure(pressure: np.ndarray, filename: str) -> None:
    """Save pressure as image file"""
    plt.imshow(X=pressure, extent=[0, X_LENGTH, 0, Y_LENGTH])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(Path('..') / 'images' / f'{filename}.png')


def update_frame(
    # pylint: disable=bad-continuation
    pressure_x: np.ndarray,
    pressure_y: np.ndarray,
    writer: FFMpegWriter,
) -> None:
    # pylint: enable=bad-continuation
    """Update video frame to new pressure for Perfect Matched Layer solvers"""
    plt.imshow(
        (
            pressure_x + pressure_y
        )[
            PERFECT_MATCHED_LAYER_SIZE_X:-PERFECT_MATCHED_LAYER_SIZE_X,
            PERFECT_MATCHED_LAYER_SIZE_Y:-PERFECT_MATCHED_LAYER_SIZE_Y,
        ],
        vmin=-1,
        vmax=1,
        extent=[0, X_LENGTH, 0, Y_LENGTH],
    )
    plt.xlabel('x')
    plt.ylabel('y')
    writer.grab_frame()
