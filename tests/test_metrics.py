import numpy as np
import pytest
from src.evaluation import metrics as m


def test_mae_rmse_known_values():
    y, p = [10, 20, 30], [12, 18, 33]
    assert m.mae(y, p) == pytest.approx((2 + 2 + 3) / 3)
    assert m.rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))


def test_perfect_forecast_is_zero():
    y = [5.0, 6.0, 7.0]
    assert m.mae(y, y) == 0 and m.rmse(y, y) == 0
    assert m.mape(y, y) == 0 and m.smape(y, y) == 0


def test_mape_and_smape_known_values():
    assert m.mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert m.smape([100], [150]) == pytest.approx(50 / 125 * 100)


def test_mape_rejects_zero():
    with pytest.raises(ValueError):
        m.mape([0, 1], [1, 1])


def test_peak_hour_mae_only_uses_peak_hours():
    hours = [3, 17, 18, 22]
    y = [100, 100, 100, 100]
    p = [999, 110, 90, 999]
    assert m.peak_hour_mae(y, p, hours) == pytest.approx(10.0)


def test_pinball_loss_median_equals_half_mae():
    y, p = [10, 20, 30], [12, 18, 33]
    assert m.pinball_loss(y, p, 0.5) == pytest.approx(0.5 * m.mae(y, p))


def test_pinball_asymmetry():
    # under-predicting is penalised more at q=0.9
    assert m.pinball_loss([10], [8], 0.9) > m.pinball_loss([10], [12], 0.9)


def test_interval_coverage():
    assert m.interval_coverage([1, 2, 3, 4], [0, 0, 0, 5], [2, 2, 2, 6]) == 0.5
