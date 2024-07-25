import numpy as np
import pandas as pd


class DigitalSignal:
    """Class to handle digital signals. Features onsets and offsets
    detection, and conversion to timebase."""

    RANGE_CALCULATION_PERCENT = 0.1

    def __init__(
        self,
        array: list | np.ndarray | pd.DataFrame,
        fs: float | int = None,
        times: float | int | None = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        array : list | np.ndarray | pd.DataFrame
            _description_
        times : float | int, optional
            _description_, by default None
        fs : float | int, optional
            _description_, by default None
        """
        self.array = np.array(array)

        if self.array.dtype != bool:
            if self.array.max() == 1:
                self.array = self.array.astype(bool)
            else:
                # Issue: empty signal with no barcodes but baseline noise could result
                # in funky detections.
                range_min = np.percentile(self.array, self.RANGE_CALCULATION_PERCENT)
                range_max = np.percentile(
                    self.array, 100 - self.RANGE_CALCULATION_PERCENT
                )
                self.array = self.array > (range_min + range_max) / 2

        if times is not None:
            self.times = times
            fs = 1 / np.mean(np.diff(self.times))
        else:
            if fs is None:
                raise ValueError(
                    "Sampling frequency must be provided if times are not."
                )
            self.times = np.arange(len(self.array)) / fs

        self.fs = int(fs)
        self._onsets = None
        self._offsets = None
        self._all_events = None

        self._onset_times = None
        self._offset_times = None
        self._all_event_times = None

    @property
    def n_pts(self) -> int:
        return len(self.array)

    @property
    def duration(self) -> float:
        return self.n_pts / self.fs

    @property
    def onsets(self) -> np.ndarray:
        if self._onsets is None:
            onsets_arr = np.insert(self.array[1:] & ~self.array[:-1], 0, False)
            self._onsets = np.nonzero(onsets_arr)[0]

        return self._onsets

    @property
    def onset_times(self) -> np.ndarray:
        if self._onset_times is None:
            self._onset_times = self.onsets / self.fs

        return self._onset_times

    @property
    def offsets(self) -> np.ndarray:
        if self._offsets is None:
            offsets_arr = np.insert(~self.array[1:] & self.array[:-1], 0, False)
            self._offsets = np.nonzero(offsets_arr)[0]

        return self._offsets

    @property
    def offset_times(self) -> np.ndarray:
        if self._offset_times is None:
            self._offset_times = self.offsets / self.fs

        return self._offset_times

    @property
    def all_events(self) -> np.ndarray:
        if self._all_events is None:
            self._all_events = np.concatenate([self.onsets, self.offsets])
            self._all_events.sort()

        return self._all_events

    @property
    def all_event_times(self) -> np.ndarray:
        if self._all_event_times is None:
            self._all_event_times = self.all_events / self.fs

        return self._all_event_times
