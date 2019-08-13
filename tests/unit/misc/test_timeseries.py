import pytest
import decimal
from firecrest.misc.time_storage import TimeSeries, TimeGridError


def dicts():
    return [
        {},
        {1: "a"},
        {1: 1, 2: 2},
        {decimal.Decimal(0): "abc", decimal.Decimal(1): "cde", decimal.Decimal(2): -2},
    ]


def numeric_series():
    return [
        TimeSeries.from_dict({i: i for i in range(10)}),
        TimeSeries.from_dict({i + 0.5: i + 0.5 for i in range(9)}),
    ]


def test_create_timeseries():
    series_empty = TimeSeries()
    assert len(series_empty) == 0

    series_one = TimeSeries(1, 0)
    assert series_one
    assert series_one[decimal.Decimal(0)] == 1
    assert len(series_one) == 1


def test_create_from_dict():
    for dictionary in dicts():
        series = TimeSeries.from_dict(dictionary)
        assert len(series) == len(dictionary)
        for el in dictionary:
            assert series[decimal.Decimal(el)] == dictionary[el]
            assert series[el] == dictionary[el]

        reversed_series = TimeSeries.from_dict(dictionary, reversed=True)

        if len(reversed_series) != 0:
            assert reversed_series.recent == series.popitem(last=False)[1]


def test_apply_func():
    series = numeric_series()[0]

    def func(x):
        return x ** 2

    new_series = series.apply(func)
    assert new_series
    for el in series:
        assert func(series[el]) == new_series[el]


def test_interpolate_to_keys():
    grid, mid_point = numeric_series()
    interpolated = TimeSeries.interpolate_to_keys(grid, mid_point)
    assert interpolated == mid_point
    assert interpolated._recent == mid_point._recent


def test_multiply():
    series, mid_point = numeric_series()
    multiplied = series * series
    assert len(multiplied) == len(series)
    assert multiplied == TimeSeries.from_dict({i: i ** 2 for i in series})
    with pytest.raises(TimeGridError):
        _ = series * mid_point

    assert mid_point * series == mid_point * mid_point


def test_integrate():
    series, mid_point = numeric_series()
    assert series.integrate() == 45
    assert mid_point.integrate() == 40.5
