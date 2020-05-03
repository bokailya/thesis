from pathlib import Path

import numpy as np
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

from .base import (
    derivative_x,
    derivative_y,
    in_circle,
    perfect_matched_layer_x,
    perfect_matched_layer_y,
    update_frame,
)
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
)


FILE_PATH: Path = Path(
    '..'
) / 'videos' / f'perfect_matched_layer_variable_speed.mp4'

writer = FFMpegWriter(FPS)


def update_perfect_matched_layer(
    pressure_x: np.ndarray,
    pressure_y: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    speed: np.ndarray,
    pressure_coefficient: float,
    velocity_coefficient: float,
) -> None:
    velocity_x += velocity_coefficient * derivative_x(pressure_x + pressure_y)
    velocity_y += velocity_coefficient * derivative_y(pressure_x + pressure_y)

    perfect_matched_layer_x(velocity_x, SIGMA_X_MAX)
    perfect_matched_layer_y(velocity_y, SIGMA_Y_MAX)

    pressure_x[1:-1] += pressure_coefficient * speed[1:-1] * derivative_x(
        velocity_x
    )
    pressure_y[:, 1:-1] += pressure_coefficient * speed[
        :, 1:-1
    ] * derivative_y(
        velocity_y
    )

    perfect_matched_layer_x(pressure_x, SIGMA_X_STAR_MAX)
    perfect_matched_layer_y(pressure_y, SIGMA_Y_STAR_MAX)

    # Neyman boundary condition
    pressure_x[0] += 2 * pressure_coefficient * velocity_x[0]
    pressure_x[-1] += 2 * pressure_coefficient * -velocity_x[-1]
    pressure_y[:, 0] += 2 * pressure_coefficient * velocity_y[:, 0]
    pressure_y[:, -1] += 2 * pressure_coefficient * -velocity_y[:, -1]


def main() -> None:
    pressure_coefficient: float = -DELTA_T * RHO / DELTA_X
    velocity_coefficient: float = -DELTA_T / (DELTA_X * RHO)

    speed: np.ndarray = np.fromfunction(
        lambda i, j: 1 - 0.5 * in_circle(
            (i - PERFECT_MATCHED_LAYER_SIZE_X) * DELTA_X,
            (j - PERFECT_MATCHED_LAYER_SIZE_Y) * DELTA_Y,
            0,
            0,
            1,
        ),
        (
            N + 2 * PERFECT_MATCHED_LAYER_SIZE_X,
            N + 2 * PERFECT_MATCHED_LAYER_SIZE_Y,
        )
    ).astype(float)

    pressure_x: np.ndarray = np.fromfunction(
        lambda i, j: in_circle(
            (i - PERFECT_MATCHED_LAYER_SIZE_X) * DELTA_X,
            (j - PERFECT_MATCHED_LAYER_SIZE_Y) * DELTA_Y,
            0.75,
            0.75,
            0.2,
        ) + in_circle(
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
    pressure_y: np.ndarray = pressure_x.copy()

    fig = plt.figure()
    with writer.saving(fig, FILE_PATH, DPI):
        update_frame(pressure_x, pressure_y, writer)
        writer.grab_frame()

        velocity_x = np.zeros((pressure_x.shape[0] - 1, pressure_x.shape[1]))
        velocity_y = np.zeros((pressure_y.shape[0], pressure_y.shape[1] - 1))

        for i in range(round(T / DELTA_T)):
            update_perfect_matched_layer(
                pressure_x,
                pressure_y,
                velocity_x,
                velocity_y,
                speed,
                pressure_coefficient,
                velocity_coefficient,
            )
            update_frame(pressure_x, pressure_y, writer)


if __name__ == '__main__':
    main()
