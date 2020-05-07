"""
Solve numerically forward problem (initial value problem)
for wave equation with variable speed of sound
using Finite-Difference Time-Domain method and Perfect Matched Layer method.
Save resulting animation.
"""
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.animation import FFMpegWriter  # type: ignore

from base import derivative_x, derivative_y, in_circle, update_frame
from constants import (
    DELTA_T,
    DELTA_X,
    DELTA_Y,
    DPI,
    FPS,
    N,
    PERFECT_MATCHED_LAYER_SIZE_X,
    PERFECT_MATCHED_LAYER_SIZE_Y,
    RHO,
    SIGMA_X_MAX,
    SIGMA_X_STAR_MAX,
    SIGMA_Y_MAX,
    SIGMA_Y_STAR_MAX,
    T,
    VELOCITY_COEFFICIENT,
)
from perfect_matched_layer_base import (
    perfect_matched_layer_x,
    perfect_matched_layer_y,
)


# for variable speed solver
PRESSURE_COEFFICIENT: float = -DELTA_T * RHO / DELTA_X

RESULT_FILENAME: str = 'perfect_matched_layer_variable_speed'


def update_perfect_matched_layer(
    # pylint: disable=bad-continuation
    pressure_x: np.ndarray,
    pressure_y: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    speed_squared: np.ndarray,
) -> None:
    # pylint: enable=bad-continuation
    """
    Update velocity and pressure by time step
    in forward problem (inital value problem) for wave equation
    with variable speed
    using Finite-Difference Time-Domain method
    and Perfect Matched Layer method
    """
    velocity_x += VELOCITY_COEFFICIENT * derivative_x(pressure_x + pressure_y)
    velocity_y += VELOCITY_COEFFICIENT * derivative_y(pressure_x + pressure_y)

    perfect_matched_layer_x(velocity_x, SIGMA_X_MAX)
    perfect_matched_layer_y(velocity_y, SIGMA_Y_MAX)

    pressure_x[1:-1] += (
        PRESSURE_COEFFICIENT
        * speed_squared[1:-1]
        * derivative_x(velocity_x)
    )
    pressure_y[:, 1:-1] += (
        PRESSURE_COEFFICIENT
        * speed_squared[:, 1:-1]
        * derivative_y(velocity_y)
    )

    perfect_matched_layer_x(pressure_x, SIGMA_X_STAR_MAX)
    perfect_matched_layer_y(pressure_y, SIGMA_Y_STAR_MAX)


def main() -> None:
    """
    Solve forward problem (initial value problem)
    for wave equation with variable speed of sound
    using Finite-Difference Time-Domain method
    and Perfect Matched Layer method
    """
    speed_squared: np.ndarray = np.fromfunction(
        function=(
            lambda i, j: (
                1
                - 0.5
                * in_circle(
                    circle_center_x_coordinate=0,
                    circle_center_y_coordinate=0,
                    circle_radius=1,
                    x_coordinate=(i - PERFECT_MATCHED_LAYER_SIZE_X) * DELTA_X,
                    y_coordinate=(j - PERFECT_MATCHED_LAYER_SIZE_Y) * DELTA_Y,
                )
            )
        ),
        shape=(
            N + 2 * PERFECT_MATCHED_LAYER_SIZE_X,
            N + 2 * PERFECT_MATCHED_LAYER_SIZE_Y,
        ),
    ).astype(float)

    pressure_x: np.ndarray = np.fromfunction(  # pylint: disable=no-member
        function=(
            lambda i, j: (
                in_circle(
                    circle_center_x_coordinate=0.75,
                    circle_center_y_coordinate=0.75,
                    circle_radius=0.2,
                    x_coordinate=(i - PERFECT_MATCHED_LAYER_SIZE_X) * DELTA_X,
                    y_coordinate=(j - PERFECT_MATCHED_LAYER_SIZE_Y) * DELTA_Y,
                )
                + in_circle(
                    circle_center_x_coordinate=0.75,
                    circle_center_y_coordinate=0.25,
                    circle_radius=0.01,
                    x_coordinate=(i - PERFECT_MATCHED_LAYER_SIZE_X) * DELTA_X,
                    y_coordinate=(j - PERFECT_MATCHED_LAYER_SIZE_Y) * DELTA_Y,
                ),
            )
        ),
        shape=(
            N + 2 * PERFECT_MATCHED_LAYER_SIZE_X,
            N + 2 * PERFECT_MATCHED_LAYER_SIZE_Y,
        ),
    ).astype(float) / 2
    pressure_y: np.ndarray = pressure_x.copy()

    fig = plt.figure()

    writer = FFMpegWriter(FPS)
    with writer.saving(
            fig=fig,
            outfile=Path('..') / 'videos' / f'{RESULT_FILENAME}.mp4',
            dpi=DPI,
    ):
        update_frame(
            pressure_x=pressure_x,
            pressure_y=pressure_y,
            writer=writer,
        )
        writer.grab_frame()

        velocity_x = np.zeros(
            shape=(pressure_x.shape[0] - 1, pressure_x.shape[1]),
        )
        velocity_y = np.zeros(
            shape=(pressure_y.shape[0], pressure_y.shape[1] - 1),
        )

        for _ in range(round(T / DELTA_T)):
            update_perfect_matched_layer(
                pressure_x=pressure_x,
                pressure_y=pressure_y,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                speed_squared=speed_squared,
            )
            update_frame(pressure_x, pressure_y, writer)


if __name__ == '__main__':
    main()
