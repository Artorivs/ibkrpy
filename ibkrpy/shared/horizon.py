# ibkrpy/shared/horizon.py
# Forecast horizon: how many bars ahead each term predicts

import math

# Cost per trade is fixed while the expected move grows with the horizon, so a
# multi-bar forecast is what lets a weak edge clear trading costs.
DEFAULT_HORIZON_BARS = {"short_term": 12, "mid_term": 7, "long_term": 5}


def forecast_horizon(config, term: str) -> int:
    """Bars ahead that models for this term are trained and tuned to predict (≥ 1)."""
    value = DEFAULT_HORIZON_BARS.get(term, 1)
    if config is not None:
        try:
            table = config.get("forecast_settings.horizon_bars") or {}
            if isinstance(table, dict) and table.get(term) is not None:
                value = table[term]
        except Exception:
            pass
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 1


def horizon_sigma(sigma_per_bar: float, horizon_bars: int) -> float:
    """σ of the move over the horizon, assuming independent bars (square-root of time)."""
    return float(sigma_per_bar) * math.sqrt(max(int(horizon_bars), 1))
