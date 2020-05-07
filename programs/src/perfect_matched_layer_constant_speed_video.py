"""
Solve numerically forward problem (initial value problem)
for wave equation with constant speed of sound.
using Finite-Difference Time-Domain method and Perfectly Matched Layer method.
Save resulting animation.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from base import update_frame
from constants import DELTA_T, DPI, FPS, T
from perfect_matched_layer_base import (
    prepare_initial_conditions,
    update_constant_speed,
)


RESULT_FILENAME: str = 'perfect_matched_layer_constant_speed'
FILE_PATH: Path = Path(
    '..'
) / 'videos' / f'{RESULT_FILENAME}.mp4'

writer = FFMpegWriter(FPS)


def main() -> None:
    """
    Solve forward problem (initial-boundary value problem)
    for wave equation with constant speed coefficient
    using Finite-Difference Time-Domain method
    and save animation
    """
    pressure_x, pressure_y = prepare_initial_conditions()

    fig = plt.figure()
    with writer.saving(fig, FILE_PATH, DPI):
        update_frame(
            pressure_x=pressure_x,
            pressure_y=pressure_y,
            writer=writer,
        )

        velocity_x = np.zeros(
            shape=(pressure_x.shape[0] - 1, pressure_x.shape[1]),
        )
        velocity_y = np.zeros(
            shape=(pressure_y.shape[0], pressure_y.shape[1] - 1),
        )

        for _ in range(round(T / DELTA_T)):
            update_constant_speed(
                pressure_x=pressure_x,
                pressure_y=pressure_y,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
            )
            update_frame(
                pressure_x=pressure_x,
                pressure_y=pressure_y,
                writer=writer,
            )


if __name__ == '__main__':
    main()
