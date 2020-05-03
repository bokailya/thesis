"""Constant definitions"""


N: int = 512

X_LENGTH: int = 1
DELTA_X: float = X_LENGTH / (N - 1)

Y_LENGTH: int = 1
DELTA_Y: float = Y_LENGTH / (N - 1)

C: int = 1
DELTA_T: float = DELTA_X / (4 * C)

HALF_MAX_ACCURACY: int = 6

PERFECT_MATCHED_LAYER_WIDTH: float = 0.2
PERFECT_MATCHED_LAYER_SIZE_X: int = round(
    PERFECT_MATCHED_LAYER_WIDTH / DELTA_X
)
PERFECT_MATCHED_LAYER_SIZE_Y: int = round(
    PERFECT_MATCHED_LAYER_WIDTH / DELTA_Y
)
