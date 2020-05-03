from pathlib import Path

import numpy as np
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

FILE_PATH: Path = Path(
    '..'
) / 'videos' / f'perfect_matched_layer_variable_speed.mp4'
FPS: int = 30
DPI: int = 100

X_LENGTH: int = 1
Y_LENGTH: int = 1
RHO: int = 1
T: float = 0.5
PERFECT_MATCHED_LAYER_WIDTH: float = 0.2

HALF_MAX_ACCURACY: int = 6
N: int = 512
MAX_SPEED: int = 2
DELTA_X: float = X_LENGTH / (N - 1)
DELTA_Y: float = Y_LENGTH / (N - 1)
DELTA_T: float = DELTA_X / (4 * MAX_SPEED)

PERFECT_MATCHED_LAYER_SIZE_X: int = round(
    PERFECT_MATCHED_LAYER_WIDTH / DELTA_X
)
PERFECT_MATCHED_LAYER_SIZE_Y: int = round(
    PERFECT_MATCHED_LAYER_WIDTH / DELTA_Y
)
SIGMA_X_MAX: int = 1000
SIGMA_X_STAR_MAX: int = 1000
SIGMA_Y_MAX: int = 1000
SIGMA_Y_STAR_MAX: int = 1000

writer = FFMpegWriter(FPS)


def coefficients(half_accuracy: int) -> np.ndarray:
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


derivative_approximation_coefficients = [
    coefficients(half_accuracy)
    for half_accuracy in range(1, HALF_MAX_ACCURACY + 1)
]


def derivative_x(function: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.zeros((function.shape[0] - 1, function.shape[1]))
    for i in range(function.shape[0] - 1):
        accuracy: int = min(12, 2 * i + 2, (function.shape[0] - i - 1) * 2)
        half_accuracy: int = accuracy // 2
        for j in range(accuracy):
            result[i] += derivative_approximation_coefficients[
                half_accuracy - 1
            ][j] * function[i + j + 1 - half_accuracy]
    return result


def derivative_y(function: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.zeros((function.shape[0], function.shape[1] - 1))
    for i in range(function.shape[1] - 1):
        accuracy: int = min(12, 2 * i + 2, (function.shape[1] - i - 1) * 2)
        half_accuracy: int = accuracy // 2
        for j in range(accuracy):
            result[:, i] += derivative_approximation_coefficients[
                half_accuracy - 1
            ][j] * function[:, i + j + 1 - half_accuracy]
    return result


def in_circle(x: float, y: float, x0: float, y0: float, radius: float) -> bool:
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


def save_pressure(pressure_x: np.ndarray, pressure_y: np.ndarray) -> None:
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
        lambda i, j: 2 - in_circle(
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
        save_pressure(pressure_x, pressure_y)
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
            save_pressure(pressure_x, pressure_y)
            if i % 4 == 3:
                writer.grab_frame()


if __name__ == '__main__':
    main()
