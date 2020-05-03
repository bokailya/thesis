"""Constant definitions"""


# Speed of sound, for solvers with constant speed of sound,
# maximal speed of sound for solvers with variable speed of sound
C: int = 1


N: int = 512

X_LENGTH: int = 1
DELTA_X: float = X_LENGTH / (N - 1)

Y_LENGTH: int = 1
DELTA_Y: float = Y_LENGTH / (N - 1)


DELTA_T: float = DELTA_X / (4 * C)

DPI: int = 100
FPS: int = 30

HALF_MAX_ACCURACY: int = 6


# Perfect Matched Layer size
PERFECT_MATCHED_LAYER_WIDTH: float = 0.2
PERFECT_MATCHED_LAYER_SIZE_X: int = round(
    PERFECT_MATCHED_LAYER_WIDTH / DELTA_X
)
PERFECT_MATCHED_LAYER_SIZE_Y: int = round(
    PERFECT_MATCHED_LAYER_WIDTH / DELTA_Y
)

# Other Perfect Matched Layer parameters
SIGMA_X_MAX: int = 1000
SIGMA_X_STAR_MAX: int = 1000
SIGMA_Y_MAX: int = 1000
SIGMA_Y_STAR_MAX: int = 1000

RHO: int = 1
T: float = 0.5
