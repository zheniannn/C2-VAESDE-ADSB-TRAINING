"""Shared physical and architectural constants for the VAESDE pipeline."""

SEQ_LEN  = 30
N_FEAT   = 4
FLAT_DIM = SEQ_LEN * N_FEAT   # 120
LATENT_DIM = 16

DT = 10.0   # seconds between ADS-B pings

FEATURES = ["E_m", "N_m", "vE_mps", "vN_mps"]
IDX_E, IDX_N, IDX_VE, IDX_VN = 0, 1, 2, 3

SPEED_VALID_THRESH = 10.0   # m/s — min speed for heading/turn-rate validity
