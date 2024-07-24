import numpy as np
import pytest

from timesynk.core import TimeBase, TimeBaseMap


def define_timebase():
    source_events_array = np.array([1, 2, 3, 4, 5])

    timebase_base = TimeBase(source_events_array)

    assert np.array_equal(timebase_base.event_times, source_events_array)


def test_timebase_map():
    true_coef = 2
    true_offset = 5

    source_events_array = np.array([1, 2, 3, 4, 5])
    target_events_array = source_events_array * 2 + 5

    timebase_base = TimeBase(source_events_array)
    timebase_scaled = TimeBase(target_events_array)

    timebase_map = TimeBaseMap.from_timebases(timebase_base, timebase_scaled)

    # make sure estimated are very close to true values:
    assert np.isclose(timebase_map._coef, true_coef)
    assert np.isclose(timebase_map._offset, true_offset)
    assert all(
        np.isclose(
            timebase_map.transform(source_events_array), timebase_scaled.event_times
        )
    )


def test_inverse_map():
    events_array = np.array([1, 2, 3, 4, 5])
    true_coef = 2
    true_offset = 5
    timebase_base = TimeBase(events_array)
    timebase_scaled = TimeBase(events_array * true_coef + true_offset)

    timebase_map = TimeBaseMap.from_timebases(timebase_base, timebase_scaled)

    inverse_map = timebase_map.inverse
    assert np.isclose(inverse_map._coef, 1 / true_coef)
    assert np.isclose(inverse_map._offset, -true_offset / true_coef)
    assert all(
        np.isclose(
            inverse_map.transform(timebase_scaled.event_times),
            timebase_base.event_times,
        )
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
        ([5], (0, 5, 1)),
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
    coef, offset, source_span, target_span = request.param
    return (
        TimeBaseMap(coef, offset, source_span, target_span),
        coef,
        offset,
        source_span,
        target_span,
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

        invalid_span = object()  # a random non-int, non-iterable
        with pytest.raises(TypeError):
            TimeBase(events_array, span=invalid_span)

    def test_timebase_map_to(self, timebase_base, timebase_scaled):
        timebase_map = timebase_base._map_to(timebase_scaled)
        assert isinstance(timebase_map, TimeBaseMap)
        assert np.isclose(timebase_map._coef, 2)
        assert np.isclose(timebase_map._offset, 5)

    def test_timebase_map_times_to(self, timebase_base, timebase_scaled):
        data = np.array([10, 20, 30, 40, 50])
        mapped_times = timebase_base.map_times_to(timebase_scaled, data)
        expected_times = timebase_base._map_to(timebase_scaled).transform(data)
        assert np.allclose(mapped_times, expected_times)

    def test_timebase_interpolate_to(
        self, timebase_with_span_base, timebase_with_span_scaled
    ):
        data = np.array([10, 20, 30, 40, 50])
        interpolated_data = timebase_with_span_base.interpolate_to(
            timebase_with_span_scaled, data
        )
        # print(timebase_with_span_base.event_ids.shape, data.shape)
        _map = timebase_with_span_base._map_to(timebase_with_span_scaled)
        # print(_map.source_timestamps.shape, _map.target_timestamps.shape)
        # assert False
        expected_data = np.interp(_map.target_timestamps, _map.source_timestamps, data)
        assert np.allclose(interpolated_data, expected_data)

        expected_source = timebase_with_span_scaled.interpolate_to(
            timebase_with_span_base, interpolated_data
        )
        assert np.allclose(expected_source, data)

    @pytest.fixture
    def timebase_base(self):
        events_array = np.array([1, 2, 3, 4, 5])
        return TimeBase(events_array)

    @pytest.fixture
    def timebase_with_span_base(self):
        events_array = np.array([1, 2, 3, 4, 5])
        span = (0, 5, 1)
        return TimeBase(events_array, span=span)

    @pytest.fixture
    def timebase_scaled(self):
        events_array = np.array([1, 2, 3, 4, 5])
        true_coef = 2
        true_offset = 5
        return TimeBase(events_array * true_coef + true_offset)

    @pytest.fixture
    def timebase_with_span_scaled(self):
        events_array = np.array([1, 2, 3, 4, 5])
        return TimeBase(events_array, span=(0, 10, 1))

    @pytest.fixture
    def timebase_fixture(self):
        event_times = np.array([1, 2, 3, 4, 5])
        event_ids = np.array([0, 1, 2, 3, 4])
        timebase = TimeBase(event_times)
        return timebase, event_times, event_ids

    @pytest.fixture
    def timebase_span_fixture(self):
        event_times = np.array([1, 2, 3, 4, 5])
        span = (0, 5, 1)
        timebase = TimeBase(event_times, span=span)
        return timebase, span


base_events_array = np.array([1, 2, 3, 4, 5])


class TestTimeBaseMap:
    def test_timebasemap_init(self, timebasemap_fixture):
        timebasemap, coef, offset, source_span, target_span = timebasemap_fixture
        assert timebasemap._coef == coef
        assert timebasemap._offset == offset
        assert timebasemap._source_span == source_span
        assert timebasemap._target_span == target_span

    @pytest.fixture
    def timebase_base(self):
        events_array = np.array(base_events_array)
        return TimeBase(events_array)

    @pytest.fixture
    def timebase_scaled(self):
        events_array = np.array(base_events_array)
        true_coef = 2
        true_offset = 5
        return TimeBase(events_array * true_coef + true_offset)

    @pytest.fixture
    def timebase_partial_overlap(self):
        events_array = np.array(base_events_array[1:])
        events_ids = np.array([1, 2, 3, 4])
        true_coef = 2
        true_offset = 5
        return TimeBase(events_array * true_coef + true_offset, event_ids=events_ids)

    @pytest.fixture
    def timebase_no_overlap(self):
        events_array = np.array([10, 20, 30, 40, 50])
        events_ids = np.array([10, 20, 30, 40, 50])
        return TimeBase(events_array, event_ids=events_ids)

    @pytest.fixture
    def timebase_with_span_scaled(self):
        events_array = np.array(base_events_array)
        return TimeBase(events_array, span=(0, 10, 1))

    def test_timebasemap_from_timebases(self, timebase_base, timebase_scaled):
        timebasemap = TimeBaseMap.from_timebases(timebase_base, timebase_scaled)
        assert np.isclose(timebasemap._coef, 2)
        assert np.isclose(timebasemap._offset, 5)

    def test_timebasemap_from_timebases_partial_overlap(
        self, timebase_base, timebase_partial_overlap
    ):
        timebasemap = TimeBaseMap.from_timebases(
            timebase_base, timebase_partial_overlap
        )
        assert np.isclose(timebasemap._coef, 2)
        assert np.isclose(timebasemap._offset, 5)

    def test_timebasemap_from_timebases_no_overlap(
        self, timebase_base, timebase_no_overlap
    ):
        with pytest.raises(ValueError) as e:
            TimeBaseMap.from_timebases(timebase_base, timebase_no_overlap)

        assert "No shared barcodes found" in str(e.value)

    @pytest.mark.parametrize(
        "input_times, expected_times",
        [
            (np.array([1, 2, 3, 4, 5]), np.array([7, 9, 11, 13, 15])),
            (np.array([0, 1, 2]), np.array([5, 7, 9])),
        ],
    )
    def test_timebasemap_transform(
        self, timebase_base, timebase_scaled, input_times, expected_times
    ):
        timebasemap = TimeBaseMap.from_timebases(timebase_base, timebase_scaled)
        transformed_times = timebasemap.transform(input_times)
        assert np.allclose(transformed_times, expected_times)

    def test_timebasemap_inverse(self, timebase_base, timebase_scaled):
        timebasemap = TimeBaseMap.from_timebases(timebase_base, timebase_scaled)
        inverse_map = timebasemap.inverse
        assert np.isclose(inverse_map._coef, 0.5)
        assert np.isclose(inverse_map._offset, -2.5)
        assert np.allclose(
            inverse_map.transform(timebase_scaled.event_times),
            timebase_base.event_times,
        )

    def test_timebasemap_from_average(
        self, timebase_base, timebase_scaled, timebase_partial_overlap
    ):
        timebasemap1 = TimeBaseMap.from_timebases(timebase_base, timebase_scaled)
        timebasemap2 = TimeBaseMap.from_timebases(
            timebase_base, timebase_partial_overlap
        )
        average_timebasemap = TimeBaseMap.average_timebase_maps(
            [timebasemap1, timebasemap2]
        )

        # Expect the averaged coef and offset to be the average of the two timebase maps
        expected_coef = np.mean([timebasemap1._coef, timebasemap2._coef])
        expected_offset = np.mean([timebasemap1._offset, timebasemap2._offset])

        assert np.isclose(average_timebasemap._coef, expected_coef)
        assert np.isclose(average_timebasemap._offset, expected_offset)


class TestTimeBaseMapInterpolation:
    def setup_method(self):
        self.events_array = np.array([1, 2, 3, 4, 5])
        self.true_coef = 2
        self.true_offset = 5

        # Adding span for both timebases
        self.timebase1 = TimeBase(self.events_array, span=(0, 5, 1))
        self.timebase2 = TimeBase(
            self.events_array * self.true_coef + self.true_offset, span=(5, 15, 2)
        )

        self.timebase_map = TimeBaseMap.from_timebases(self.timebase1, self.timebase2)

    def test_resample_basic(self):
        data = np.array([10, 20, 30, 40, 50])
        resampled_data = self.timebase_map.resample(data)
        expected_resampled_data = np.interp(
            self.timebase_map.target_timestamps,
            self.timebase_map.source_timestamps,
            data,
        )
        assert np.allclose(resampled_data, expected_resampled_data)

    @pytest.mark.parametrize(
        "data",
        [
            np.array([1]),
            [],
            np.array([10, 20, 30, 40, 10, 20, 30, 40]),
        ],
    )
    def test_resample_edge_case_different_lengths(self, data):
        with pytest.raises(ValueError):
            self.timebase_map.resample(data)

    def test_resample_edge_case_non_numeric_data(self):
        data = np.array(["a", "b", "c", "d", "e"])
        with pytest.raises(TypeError):
            self.timebase_map.resample(data)

    def test_resample_edge_case_invalid_target_span(self):
        self.timebase_map._target_span = None
        data = np.array([10, 20, 30, 40, 50])
        with pytest.raises(
            ValueError, match="Target span must be provided to interpolate values!"
        ):
            self.timebase_map.resample(data)

    def test_resample_with_inverse(self):
        data = np.array([10, 20, 30, 40, 50])
        inverse_timebase_map = self.timebase_map.inverse
        resampled_data = inverse_timebase_map.resample(data)
        expected_resampled_data = np.interp(
            inverse_timebase_map.target_timestamps,
            inverse_timebase_map.source_timestamps,
            data,
        )
        assert np.allclose(resampled_data, expected_resampled_data)


if __name__ == "__main__":
    pytest.main()
