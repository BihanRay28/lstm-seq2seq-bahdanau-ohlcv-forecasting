# Dataset

`BTC_USDT_30m_Binance_20260913_184821.csv` is the exact input used by Experiment 1.

- Market: BTC/USDT
- Venue indicated by the source filename: Binance
- Interval: 30 minutes
- Rows: 52,560
- Range: 2023-09-14 13:30:00 to 2026-09-13 13:00:00
- Columns: `datetime, open, high, low, close, volume`
- SHA-256: `6c20e9718cc91ee95732226af876c21ca0d7767f612f5056cdce633fc83cde62`

The loader requires strict chronological order, no duplicate timestamps, an uninterrupted 30-minute cadence, finite values, positive OHLC prices, nonnegative volume, and valid high/low bounds.

The original retrieval script and exchange response metadata were not present with the experiment, so those details are intentionally not inferred here.
