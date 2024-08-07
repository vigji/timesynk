import numbers
from functools import cached_property
from typing import Iterable, Optional, Union

import numpy as np

SpanType = Union[numbers.Real, Iterable[numbers.Real]]


def _round_to_some_decimal(value: numbers.Real, decimal: int) -> numbers.Real:
    """
    Round a value to some decimal places.

    Parameters
    ----------
    value : numbers.Real
        Value to be rounded.
    decimal : int
        Number of decimal places to round to.

    Returns
    -------
    numbers.Real
        Rounded value.
    """
    return round(value, decimal)


def _validate_and_set_span(span: Optional[SpanType]) -> np.ndarray:
    """
    Validate and convert the span parameter to a numpy array.

    Parameters
    ----------
    span : Optional[SpanType]
        Span value which can be None, a single numeric value, or an iterable of numeric values.

    Returns
    -------
    np.ndarray
        Converted span in the form of a numpy array.

    Raises
    ------
    TypeError
        If the span is not a numeric value or an iterable of numeric values.
    ValueError
        If the iterable span has more than 3 elements.
    """
    if span is None:
        return None

    if isinstance(span, (int, float)):
        return (0, span, 1)  # Single numeric value

    if isinstance(span, Iterable):
        for elem in span:
            if not isinstance(elem, numbers.Number) and not elem is None:
                raise TypeError("Span iterable must contain only numeric values")

        if len(span) >= 1 and len(span) <= 3:
            # leverage known behavior of slice() to handle 1, 2, or 3 elements:
            _slice = slice(*span)
            return (_slice.start, _slice.stop, _slice.step)
        else:
            raise ValueError("Span iterable must have between 1 and 3 elements")

    raise TypeError(
        "Span must be a numeric value, a tuple, or an iterable of numeric values"
    )


class TimeBase:
    """
    A class representing a base of event times and corresponding event IDs, with optional span.

    Attributes
    ----------
    event_times : np.ndarray
        Array of event times.
    event_ids : np.ndarray
        Array of event IDs.
    span : Optional[np.ndarray]
        Span of the time base.

    Methods
    -------
    map_times_to(target_timebase, data)
        Maps the times from this TimeBase to another TimeBase.
    resample_to(target_timebase, data)
        Resample the values from this TimeBase to another TimeBase.
    """

    def __init__(
        self,
        event_times: Iterable[numbers.Real],
        event_ids: Optional[Iterable] = None,
        span: Optional[SpanType] = None,
    ) -> None:
        """
        Initialize a TimeBase instance.

        Parameters
        ----------
        event_times : Iterable[numbers.Real]
            Iterable of event times.
        event_ids : Optional[Iterable[numbers.Real]]
            Optional iterable of event IDs.
        span : Optional[SpanType]
            Optional span value.

        Raises
        ------
        ValueError
            If the lengths of event_times and event_ids do not match.
        """
        self.span = _validate_and_set_span(span)
        self.event_times = np.array(event_times)

        if event_ids is None:
            self.event_ids = np.arange(len(self.event_times))
        else:
            self.event_ids = np.array(event_ids)

        if len(self.event_times) != len(self.event_ids):
            raise ValueError("Event times and event ids must have the same length.")

    def __repr__(self) -> str:
        return (
            f"TimeBase(event_times={self.event_times}, event_ids={self.event_ids}, "
            f"span={self.span})"
        )

    def __str__(self) -> str:
        return f"TimeBase with {len(self.event_times)} events, " f"span={self.span}"

    def _map_to(self, other_timebase: "TimeBase") -> "TimeBaseMap":
        """
        Create a mapping from this TimeBase to another TimeBase.

        Parameters
        ----------
        other_timebase : TimeBase
            The target TimeBase.

        Returns
        -------
        TimeBaseMap
            A mapping object from this TimeBase to the target TimeBase.
        """
        return TimeBaseMap.from_timebases(self, other_timebase)

    def transform_to(
        self, target_timebase: "TimeBase", data: Iterable[numbers.Real]
    ) -> Iterable[numbers.Real]:
        """
        Maps the times from this TimeBase to another TimeBase.

        Parameters
        ----------
        target_timebase : TimeBase
            The target TimeBase.
        data : Iterable[numbers.Real]
            The data to be mapped.

        Returns
        -------
        Iterable[numbers.Real]
            The mapped data.
        """
        own_to_target_map = self._map_to(target_timebase)
        return own_to_target_map.transform(data)

    def resample_to(
        self, target_timebase: "TimeBase", data: Iterable[numbers.Real]
    ) -> Iterable[numbers.Real]:
        """
        Resample the times from this TimeBase to another TimeBase.

        Parameters
        ----------
        target_timebase : TimeBase
            The target TimeBase.
        data : Iterable[numbers.Real]
            The data to be resampled.

        Returns
        -------
        Iterable[numbers.Real]
            The resampled data.
        """
        own_to_target_map = self._map_to(target_timebase)
        return own_to_target_map.resample(data)


class TimeBaseMap:
    """
    A class representing a mapping between two TimeBase instances.

    Attributes
    ----------
    _coef : numbers.Real
        Coefficient of the linear transformation.
    _offset : numbers.Real
        Offset of the linear transformation.
    _source_span : Optional[np.ndarray]
        Span of the source TimeBase.
    _target_span : Optional[np.ndarray]
        Span of the target TimeBase.

    Methods
    -------
    from_timebases(source_timebase, target_timebase)
        Create a TimeBaseMap from two TimeBase instances.
    average_timebase_maps(timebase_map_list)
        Create a TimeBaseMap from the average of a list of TimeBaseMap instances.
    transform(times)
        Transform times using the TimeBaseMap.
    resample(data)
        Resample data using the TimeBaseMap.
    inverse()
        Get the inverse of the TimeBaseMap.
    """

    def __init__(
        self,
        coef: numbers.Real,
        offset: numbers.Real,
        source_span: Optional[SpanType] = None,
        target_span: Optional[SpanType] = None,
    ) -> None:
        """
        Initialize a TimeBaseMap instance.

        Parameters
        ----------
        coef : numbers.Real
            Coefficient of the linear transformation.
        offset : numbers.Real
            Offset of the linear transformation.
        source_span : Optional[SpanType]
            Span of the source TimeBase.
        target_span : Optional[SpanType]
            Span of the target TimeBase.
        """
        self._coef = coef
        self._offset = offset

        self._source_span = _validate_and_set_span(source_span)
        self._target_span = _validate_and_set_span(target_span)

    def __repr__(self) -> str:
        return (
            f"TimeBaseMap(coef={self._coef}, offset={self._offset}, "
            f"source_span={self._source_span}, target_span={self._target_span})"
        )

    def __str__(self) -> str:
        return (
            f"TimeBaseMap with coef={self._coef}, offset={self._offset}, "
            f"source_span={self._source_span}, target_span={self._target_span}"
        )

    @classmethod
    def from_timebases(
        cls, source_timebase: TimeBase, target_timebase: TimeBase
    ) -> "TimeBaseMap":
        """
        Create a TimeBaseMap from two TimeBase instances.

        Parameters
        ----------
        source_timebase : TimeBase
            The source TimeBase.
        target_timebase : TimeBase
            The target TimeBase.

        Returns
        -------
        TimeBaseMap
            A mapping object from the source TimeBase to the target TimeBase.

        Raises
        ------
        ValueError
            If there are no shared event IDs between the source and target TimeBase.
        """

        shared, source_index, target_index = np.intersect1d(
            source_timebase.event_ids, target_timebase.event_ids, return_indices=True
        )
        if len(shared) <= 2:
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
        """
        Create a TimeBaseMap from the average of a list of TimeBaseMap instances.

        Parameters
        ----------
        timebase_map_list : list of TimeBaseMap
            List of TimeBaseMap instances to average.

        Returns
        -------
        TimeBaseMap
            A mapping object representing the average of the input TimeBaseMap instances.
        """
        coef = np.mean([timebase_map._coef for timebase_map in timebase_map_list])
        offset = np.mean([timebase_map._offset for timebase_map in timebase_map_list])
        return cls(coef=coef, offset=offset)

    @cached_property
    def source_timestamps(self) -> np.ndarray:
        """
        Get the source timestamps.

        Returns
        -------
        np.ndarray
            Source timestamps.

        Raises
        ------
        ValueError
            If the source span is not provided.
        """
        if self._source_span is None:
            raise ValueError("Source span must be provided to resample values!")
        return np.arange(
            self._source_span[0], self._source_span[1], self._source_span[2]
        )

    @cached_property
    def target_timestamps(self) -> np.ndarray:
        """
        Get the target timestamps.

        Returns
        -------
        np.ndarray
            Target timestamps.

        Raises
        ------
        ValueError
            If the target span is not provided.
        """
        if self._target_span is None:
            raise ValueError("Target span must be provided to resample values!")
        
        # ROUND_DECIMALS=10
        #target_dt = round(self._target_span[2] / self._coef, ROUND_DECIMALS)
        # self.transform(
        #target_endpoint = round(self._target_span[1] / self._coef, ROUND_DECIMALS)  # + self._offset
        return np.arange(
             self._target_span[0], self._target_span[1]+1e-10, self._target_span[2])
        
        # return self.transform(np.arange(
        #      self._target_span[0], target_endpoint, target_dt)
        # )
        # ROUND_DECIMALS = 10

        # if self._target_span is None:
        #     return None
        # target_dt = round(self._target_span[2] * self._coef, ROUND_DECIMALS)
        # target_endpoint = round(self._target_span[1] * self._coef, ROUND_DECIMALS)  # + self._offset
        
        # print(self._target_span[0], target_endpoint, target_dt,
        #       len(np.arange(self._target_span[0], target_endpoint, target_dt)))
        # return np.arange(
        #     self._target_span[0], target_endpoint, target_dt
        # )

    @property
    def inverse(self) -> "TimeBaseMap":
        """
        Get the inverse of the TimeBaseMap.

        Returns
        -------
        TimeBaseMap
            The inverse mapping object.
        """
        return TimeBaseMap(
            coef=1 / self._coef,
            offset=-self._offset / self._coef,
            source_span=self._target_span,
            target_span=self._source_span,
        )

    def transform(self, times: Union[int, float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Transform times using the TimeBaseMap.

        Parameters
        ----------
        times : Union[int, float, np.ndarray]
            Times to be transformed.

        Returns
        -------
        Union[int, np.ndarray]
            Transformed times.
        """
        return times * self._coef + self._offset

    def resample(self, data: np.ndarray) -> np.ndarray:
        """
        Resample data using the TimeBaseMap.

        Parameters
        ----------
        data : np.ndarray
            Data to be resampled.

        Returns
        -------
        np.ndarray
            Resampled data.

        Raises
        ------
        ValueError
            If the length of the data does not match the length of the source timestamps.
        """

        #print(len(self.target_timestamps), len(self.source_timestamps), len(data))

        if abs(len(data) - len(self.source_timestamps)) > 1:
            raise ValueError(
                f"Data to resample (len {len(data)}) must have the same length as the source timestamps {len(self.source_timestamps)}"
            )
        
        # clip or pad to avoid off-by-one errors of data and source_timestamps:
        if len(data) < len(self.source_timestamps):
            data = np.pad(data, (0, len(self.source_timestamps) - len(data)))
        elif len(data) > len(self.source_timestamps):
            data = data[:len(self.source_timestamps)]

        # We will resample the data to target timestamps mapped onto source timebase:
        target_to_source_timestamps = self.inverse.transform(self.target_timestamps)

        return np.interp(target_to_source_timestamps, 
                         self.source_timestamps, 
                         data,
                         left=np.nan,
                         right=np.nan)


if __name__ == "__main__":
    # %%

    data_path_list = [r"F:\Luigi\M19_D558\20240419\133356\NpxData"]
    run_barcodeSync = False
    run_preprocessing = True # run preprocessing and spikesorting
    callKSfromSI = False

    # %%
    # %matplotlib widget
    from matplotlib import pyplot as plt
    import spikeinterface.extractors as se
    import spikeinterface.widgets as sw
    import spikeinterface.preprocessing as st

    from spikeinterface import get_noise_levels, aggregate_channels
    from pathlib import Path
    import os
    import numpy as np

    # from preprocessing_utils import *
    from nwb_conv.oephys import OEPhysDataFolder


    # %%
    data_path = data_path_list[0]
    # data_path = "/Users/vigji/Desktop/test_mpa_dir/P02_MPAOPTO_LP/e05_doubleservoarm-ephys-pagfiber/v01/M20_D545/20240424/154810"
    oephys_data = OEPhysDataFolder(data_path)

    all_stream_names, ap_stream_names = oephys_data.stream_names, oephys_data.ap_stream_names

    # %%
    npx_barcode = oephys_data.reference_npx_barcode

    oephys_data.nidaq_channel_map = {0: "frames-log", 
                                    1: "laser-log", 
                                    2: "-", 
                                    3: "motor-log", 
                                    4: "barcode", 
                                    5: "-", 
                                    6: "-", 
                                    7: "-"}
    nidaq_data = oephys_data.nidaq_recording

    nidaq_barcode = nidaq_data.barcode
    laser_data = nidaq_data.continuous_signals["laser-log"]
    nidaq_barcode.barcode_numbers