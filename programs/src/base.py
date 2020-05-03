"""Common settings and functions"""
import numpy as np

from .constants import (
    DELTA_T,
    HALF_MAX_ACCURACY,
    PERFECT_MATCHED_LAYER_SIZE_X,
    PERFECT_MATCHED_LAYER_SIZE_Y,
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
        accuracy: int = min(12, 2 * i + 2, (function.shape[0] - i - 1) * 2)
        half_accuracy: int = accuracy // 2
        for j in range(accuracy):
            result[i] += _numerical_differentiation_coefficients[
                half_accuracy - 1
            ][j] * function[i + j + 1 - half_accuracy]
    return result


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
    return result


def in_circle(x: float, y: float, x0: float, y0: float, radius: float) -> bool:
    """Check if point lays inside circle

    Args:
        x: Point x coordinate
        y: Point y coordinate
        x0: Circle center x coordinate
        y0: Circle center y coordinate
        radius: Circle radius

    Returns:
        True if point lays inside circle, False otherwise
    """
    return (x - x0)**2 + (y - y0)**2 < radius**2


def perfect_matched_layer_x(data: np.ndarray, sigma_max: int):
    for i in range(PERFECT_MATCHED_LAYER_SIZE_X):
        coefficient = 1 - DELTA_T * sigma_max * (
            1 - i / PERFECT_MATCHED_LAYER_SIZE_X
        )
        data[i] *= coefficient
        data[-i - 1] *= coefficient


def perfect_matched_layer_y(data: np.ndarray, sigma_max: int):
    for i in range(PERFECT_MATCHED_LAYER_SIZE_Y):
        coefficient = 1 - DELTA_T * sigma_max * (
            1 - i / PERFECT_MATCHED_LAYER_SIZE_Y
        )
        data[:, i] *= coefficient
        data[:, -i - 1] *= coefficient
