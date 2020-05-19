#!/usr/bin/env python
"""
Solve numerically forward problem for wave equation
with Perfectly Matched Layer
using Finite-Difference Time-Domain method
"""
import numpy as np  # type: ignore

from constants import DELTA_T, T
from perfect_matched_layer_base import (
    prepare_initial_conditions,
    save_pressure,
    update_constant_speed,
)


INITIAL_CONDITIONS_FILENAME: str = 'initial_conditions'
FORWARD_SOLUTION_FILENAME: str = 'forward_perfect_matched_layer'


def main() -> None:
    """
    Solve forward problem (inital value problem) for wave equation
    using Finite-Difference Time-Domain method
    and Perfect Matched Layer method
    and save the image of result
    """
    pressure_x, pressure_y = prepare_initial_conditions()

    save_pressure(
        pressure_x=pressure_x,
        pressure_y=pressure_y,
        filename=INITIAL_CONDITIONS_FILENAME,
    )

    velocity_x = np.zeros(shape=(pressure_x.shape[0] - 1, pressure_x.shape[1]))
    velocity_y = np.zeros(shape=(pressure_y.shape[0], pressure_y.shape[1] - 1))

    for _ in range(round(T / DELTA_T)):
        update_constant_speed(
            pressure_x=pressure_x,
            pressure_y=pressure_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
        )

    save_pressure(
        pressure_x=pressure_x,
        pressure_y=pressure_y,
        filename=FORWARD_SOLUTION_FILENAME,
    )


if __name__ == '__main__':
    main()
