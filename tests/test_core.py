from timesynk.core import TimeBase, TimeBaseMap
import numpy as np
import pytest


def define_timebase():
    source_events_array = np.array([1, 2, 3, 4, 5])

    timebase1 = TimeBase(source_events_array)

    assert np.array_equal(timebase1.event_times, source_events_array)


def test_timebase_map():
    true_coef = 2
    true_offset = 5

    source_events_array = np.array([1, 2, 3, 4, 5])
    target_events_array = source_events_array * 2 + 5

    timebase1 = TimeBase(source_events_array)
    timebase2 = TimeBase(target_events_array)

    timebase_map = TimeBaseMap.from_timebases(timebase1, timebase2)

    # make sure estimated are very close to true values:
    assert np.isclose(timebase_map.coef, true_coef)
    assert np.isclose(timebase_map.offset, true_offset)
    assert all(
        np.isclose(timebase_map.transform(source_events_array), timebase2.event_times)
    )


def test_inverse_map():
    events_array = np.array([1, 2, 3, 4, 5])
    true_coef = 2
    true_offset = 5
    timebase1 = TimeBase(events_array)
    timebase2 = TimeBase(events_array * true_coef + true_offset)

    timebase_map = TimeBaseMap.from_timebases(timebase1, timebase2)

    inverse_map = timebase_map.inverse
    assert np.isclose(inverse_map.coef, 1 / true_coef)
    assert np.isclose(inverse_map.offset, -true_offset / true_coef)
    assert all(
        np.isclose(inverse_map.transform(timebase2.event_times), timebase1.event_times)
    )


# Parameterized fixture for TimeBase instances
@pytest.fixture(
    params=[
        (np.array([1, 2, 3, 4, 5]), None, np.array([0, 1, 2, 3, 4])),
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([10, 20, 30, 40, 50]),
            np.array([10, 20, 30, 40, 50]),
        ),
    ]
)
def timebase_fixture(request):
    event_times, event_ids, expected_ids = request.param
    return TimeBase(event_times, event_ids), event_times, expected_ids


# Parameterized fixture for TimeBase instances with span
@pytest.fixture(
    params=[
        ((0, 10, 2), (0, 10, 2)),
        ((0, 10), (0, 10, 1)),
        (5, (0, 5, 1)),
        (None, None),
    ]
)
def timebase_span_fixture(request):
    span, expected_span = request.param
    events_array = np.array([1, 2, 3, 4, 5])
    return TimeBase(events_array, span=span), expected_span


# Parameterized fixture for TimeBaseMap instances
@pytest.fixture(
    params=[(2, 5, (0, 10, 1), (0, 20, 2)), (1.5, 3, (0, 5, 0.5), (0, 10, 1))]
)
def timebasemap_fixture(request):
    coef, offset, source_domain, target_domain = request.param
    return (
        TimeBaseMap(coef, offset, source_domain, target_domain),
        coef,
        offset,
        source_domain,
        target_domain,
    )


class TestTimeBase:
    def test_timebase_init(self, timebase_fixture):
        timebase, event_times, expected_ids = timebase_fixture
        assert np.array_equal(timebase.event_times, event_times)
        assert np.array_equal(timebase.event_ids, expected_ids)

    def test_timebase_with_invalid_event_ids_length(self):
        events_array = np.array([1, 2, 3, 4, 5])
        event_ids = np.array([10, 20])
        with pytest.raises(ValueError):
            TimeBase(events_array, event_ids)

    def test_timebase_with_span(self, timebase_span_fixture):
        timebase, expected_span = timebase_span_fixture
        assert np.array_equal(timebase.span, expected_span)

    def test_timebase_invalid_span(self):
        events_array = np.array([1, 2, 3, 4, 5])
        
        invalid_span = (10, 0, "a")
        with pytest.raises(TypeError):
            TimeBase(events_array, span=invalid_span)

        invalid_span = (10, 0, 1, 2)
        with pytest.raises(ValueError):
            TimeBase(events_array, span=invalid_span)

        invalid_span = lambda x: x  # a random non-int, non-iterable
        with pytest.raises(TypeError):
            TimeBase(events_array, span=invalid_span)


class TestTimeBaseMap:
    def test_timebasemap_init(self, timebasemap_fixture):
        timebasemap, coef, offset, source_span, target_span = timebasemap_fixture
        assert timebasemap.coef == coef
        assert timebasemap.offset == offset
        assert timebasemap.source_span == source_span
        assert timebasemap.target_span == target_span

    @pytest.fixture
    def timebase1(self):
        events_array = np.array([1, 2, 3, 4, 5])
        return TimeBase(events_array)

    @pytest.fixture
    def timebase2(self):
        events_array = np.array([1, 2, 3, 4, 5])
        true_coef = 2
        true_offset = 5
        return TimeBase(events_array * true_coef + true_offset)

    def test_timebasemap_from_timebases(self, timebase1, timebase2):
        timebasemap = TimeBaseMap.from_timebases(timebase1, timebase2)
        assert np.isclose(timebasemap.coef, 2)
        assert np.isclose(timebasemap.offset, 5)

    @pytest.mark.parametrize(
        "input_times, expected_times",
        [
            (np.array([1, 2, 3, 4, 5]), np.array([7, 9, 11, 13, 15])),
            (np.array([0, 1, 2]), np.array([5, 7, 9])),
        ],
    )
    def test_timebasemap_transform(
        self, timebase1, timebase2, input_times, expected_times
    ):
        timebasemap = TimeBaseMap.from_timebases(timebase1, timebase2)
        transformed_times = timebasemap.transform(input_times)
        assert np.allclose(transformed_times, expected_times)

    def test_timebasemap_inverse(self, timebase1, timebase2):
        timebasemap = TimeBaseMap.from_timebases(timebase1, timebase2)
        inverse_map = timebasemap.inverse
        assert np.isclose(inverse_map.coef, 0.5)
        assert np.isclose(inverse_map.offset, -2.5)

    def test_timebasemap_resample(self, timebase1, timebase2):
        timebasemap = TimeBaseMap.from_timebases(timebase1, timebase2)
        data = np.array([10, 20, 30, 40, 50])
        resampled_data = timebasemap.resample(data)
        # Example check assuming linear data transformation
        assert np.allclose(resampled_data, data * 2 + 5)


if __name__ == "__main__":
    pytest.main()
