"""Tests for src/base.py"""
import unittest

import numpy as np  # type: ignore

from base import derivative_x
from constants import DELTA_X


class TestDifferentiation(unittest.TestCase):
    """Tests for derivative_x, derivative_y"""

    @staticmethod
    def test_derivative_x_2_by_2_array():
        """Test differentiation by first coordinate"""
        array_for_differentiation: np.ndarray = np.array(
            object=[[2, 3], [5, 7]],
        )
        expected_derivative: np.ndarray = np.array(
            object=[[3 / DELTA_X, 4 / DELTA_X]],
        )

        derivative = derivative_x(array_for_differentiation)

        np.testing.assert_almost_equal(derivative, expected_derivative)

    @staticmethod
    def test_derivative_x_1_by_1_array():
        """Test differentiation by first coordinate"""
        array_for_differentiation: np.ndarray = np.array(object=[[0]])
        expected_derivative: np.ndarray = np.empty(shape=(0, 1))

        derivative = derivative_x(array_for_differentiation)

        np.testing.assert_almost_equal(derivative, expected_derivative)

    @staticmethod
    def test_derivative_x_constant_array():
        """Test differentiation by first coordinate"""
        array_for_differentiation: np.ndarray = np.array(
            object=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        )
        expected_derivative: np.ndarray = np.array(
            object=[[0, 0, 0], [0, 0, 0]],
        )

        derivative = derivative_x(array_for_differentiation)

        np.testing.assert_almost_equal(derivative, expected_derivative)
