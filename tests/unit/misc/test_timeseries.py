import pytest
import decimal
from firecrest.misc.time_storage import TimeSeries, TimeGridError


def dicts():
    return [
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

    assert series_empty._first is None
    assert series_empty.first is None

    assert series_empty._last is None
    assert series_empty.last is None

    series_one = TimeSeries(1, 0)
    assert series_one
    assert series_one[decimal.Decimal(0)] == 1
    assert len(series_one) == 1

    assert series_one.first == series_one.last == 1
    assert series_one._first == series_one._last == 0


def test_create_from_dict():
    for dictionary in dicts():
        series = TimeSeries.from_dict(dictionary)
        assert len(series) == len(dictionary)
        for el in dictionary:
            assert series[decimal.Decimal(el)] == dictionary[el]
            assert series[el] == dictionary[el]

        assert series.first == dictionary[min(dictionary)]
        assert series.last == dictionary[max(dictionary)]

        assert series._first == min(dictionary)
        assert series._last == max(dictionary)


def test_create_from_list():
    for series in numeric_series() + dicts():
        values = series.values()
        new_series = TimeSeries.from_list(values, series)
        assert new_series == series


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

    assert interpolated._first == mid_point._first
    assert interpolated._last == mid_point._last

    assert interpolated.first == mid_point.first
    assert interpolated.last == mid_point.last


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
