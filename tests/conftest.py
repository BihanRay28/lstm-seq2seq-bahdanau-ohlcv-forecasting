from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    rows = 160
    index = np.arange(rows, dtype=np.float64)
    open_price = 100.0 + index * 0.1
    close_price = open_price * (1.0 + 0.001 * np.sin(index / 3.0))
    high = np.maximum(open_price, close_price) + 0.5
    low = np.minimum(open_price, close_price) - 0.5
    volume = 10.0 + index % 17
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=rows, freq="30min"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close_price,
            "volume": volume,
        }
    )
