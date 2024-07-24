from unicodedata import numeric
import numpy as np
from typing import Union, Tuple, Optional, Iterable
import numbers

SpanType = Union[numbers.Real, Iterable[numbers.Real]]


def _validate_and_set_span(span: Optional[SpanType]) -> np.ndarray:
    if span is None:
        return None

    if isinstance(span, (int, float)):
        return (0, span, 1)  # Single numeric value

    if isinstance(span, Iterable):
        for elem in span:
            if not isinstance(elem, numbers.Number):
                raise TypeError("Span iterable must contain only numeric values")

        if len(span) == 1:
            return (0, span[0], 1)
        elif len(span) == 2:
            return (span[0], span[1], 1)
        elif len(span) == 3:
            return tuple(span)
        else:
            raise ValueError("Span iterable must have between 1 and 3 elements")

    raise TypeError(
        "Span must be a numeric value, a tuple, or an iterable of numeric values"
    )


class TimeBase:
    def __init__(
        self,
        event_times: Iterable[numbers.Real],
        event_ids: Optional[Iterable[numbers.Real]] = None,
        span: Optional[SpanType] = None,
    ) -> None:
        
        self.span = _validate_and_set_span(span)
        
        self.event_times = np.array(event_times)
        # span: either number of samples, if working with idxs, or a tuple of (start, end, step)
        if event_ids is None:
            self.event_ids = np.arange(len(self.event_times))
        else:
            self.event_ids = np.array(event_ids)

    def map_to(self, other_timebase: "TimeBase") -> "TimeBaseMap":
        return TimeBaseMap.from_timebases(self, other_timebase)

    def map_times_to(
        self, other_timebase: "TimeBase", data: Iterable[numbers.Real]
    ) -> Iterable[numbers.Real]:
        """Core index conversion function, without type conversion."""
        own_to_other_map = self.map_to(other_timebase)
        return own_to_other_map.transform(data)

    def interpolate_to(
        self, other_timebase: "TimeBase", data: Iterable[numbers.Real]
    ) -> Iterable[numbers.Real]:
        """Core index conversion function, with type conversion."""
        pass


class TimeBaseMap:
    def __init__(
        self,
        coef: numbers.Real,
        offset: numbers.Real,
        source_span: Optional[SpanType] = None,
        target_span: Optional[SpanType] = None,
    ) -> None:
        self.coef = coef
        self.offset = offset

        self.source_span = _validate_and_set_span(source_span)
        self.target_span = _validate_and_set_span(target_span)

        self._source_timestamps = None
        self._target_timestamps = None

    @classmethod
    def from_timebases(
        cls, source_timebase: TimeBase, target_timebase: TimeBase
    ) -> "TimeBaseMap":
        # Pull the index values from events shared by both sets of data:
        shared, source_index, target_index = np.intersect1d(
            source_timebase.event_ids, target_timebase.event_ids, return_indices=True
        )
        if len(shared) <= 1:
            raise ValueError(
                "No shared barcodes found. Check the barcode_numbers for the two original barcodes."
            )

        source_shared_event_times = source_timebase.event_times[source_index]
        target_shared_event_times = target_timebase.event_times[target_index]

        coef, offset = np.polyfit(
            source_shared_event_times, target_shared_event_times, 1
        )

        return cls(
            coef=coef,
            offset=offset,
            source_span=source_timebase.span,
            target_span=target_timebase.span,
        )

    @classmethod
    def average_timebase_maps(
        cls, timebase_map_list: list["TimeBaseMap"]
    ) -> "TimeBaseMap":
        coef = np.mean([timebase_map.coef for timebase_map in timebase_map_list])
        offset = np.mean([timebase_map.offset for timebase_map in timebase_map_list])
        return cls(coef, offset)

    @property
    def source_timestamps(self) -> np.ndarray:
        if self.source_span is None:
            raise ValueError("Source span must be provided to interpolate values!")
        if self._source_timestamps is None:
            self._source_timestamps = np.arange(
                self.source_span[0], self.source_span[1], self.source_span[2]
            )
        return self._source_timestamps

    @property
    def target_timestamps(self) -> np.ndarray:
        if self.target_span is None:
            raise ValueError("Target span must be provided to interpolate values!")
        if self._target_timestamps is None:
            self._target_timestamps = np.arange(
                self.target_span[0], self.target_span[1], self.target_span[2]
            )
        return self._target_timestamps

    @property
    def inverse(self) -> "TimeBaseMap":
        return TimeBaseMap(
            coef=1 / self.coef,
            offset=-self.offset / self.coef,
            source_span=self.target_span,
            target_span=self.source_span,
        )

    def transform(self, times: int | float | np.ndarray) -> int | np.ndarray:
        return times * self.coef + self.offset

    def resample(self, data: np.ndarray) -> np.ndarray:
        return np.interp(self.source_timestamps, self.target_timestamps, data)


if __name__ == "__main__":
    # Example usage
    events_array = np.array([1, 2, 3, 4, 5])
    true_coef = 2
    true_offset = 5
    timebase1 = TimeBase(events_array)
    timebase2 = TimeBase(events_array * true_coef + true_offset)

    timebase_map = TimeBaseMap.from_timebases(timebase1, timebase2)

    # make sure estimated are very close to true values:
    assert np.isclose(timebase_map.coef, true_coef)
    assert np.isclose(timebase_map.offset, true_offset)
    assert all(np.isclose(timebase_map.transform(events_array), timebase2.event_times))

    print(timebase_map.transform(1))  # 1.0
    print(timebase1.event_times)
