"""Adapted from:
    Optogenetics and Neural Engineering Core ONE Core, University of Colorado, School of Medicine
    18.Nov.2021 (See bit.ly/onecore for more information, including a more detailed write up.)
"""

from functools import cached_property

import numpy as np

from ..core import TimeBase, TimeBaseMap
from .digital_signal import DigitalSignal

EPSILON = 1e-10


class BarcodeSignal(DigitalSignal):
    """Class for extracting and analyzing barcodes from a digital signal.

    Attributes:
    -----------
    barcodes : np.ndarray
        Array of barcodes and their index.

    Parameters (class attributes - based on your Arduino barcode generator settings):
        NBITS = (int) the number of bits (bars) that are in each barcode (not
              including wrappers).
        INTER_BC_INTERVAL_MS = (int) The duration of time (in milliseconds)
                               between each barcode's start.
        WRAP_DURATION_MS = (int) The duration of time (in milliseconds) of the
                          ON wrapper portion (default = 10 ms) in the barcodes.
        BAR_DURATION_MS = (int) The duration of time (in milliseconds) of each
                         bar (bit) in the barcode.

        GLOBAL_TOLERANCE = (float) The fraction (in %/100) of tolerance allowed
                         for duration measurements (ex: BAR_DURATION_MS).

    """

    # General variables; make sure these align with the timing format of
    # your Arduino-generated barcodes.
    NBITS = 32  # Number of bits in each barcode
    INTER_BC_INTERVAL_MS = 5000  # Distance of barcodes, in milliseconds
    WRAP_DURATION_MS = 10  # Duration of the wrapper pulses, in milliseconds
    BAR_DURATION_MS = 30  # Duration of the barcode, in milliseconds
    GLOBAL_TOLERANCE = 0.20  # In %/100

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the BarcodeSignal class."""
        super().__init__(*args, **kwargs)
        if self.fs is None:
            raise ValueError("Sampling rate must be specified.")

    @property
    def barcode_numbers(self) -> np.ndarray:
        """Array of barcodes."""
        return self._barcodes_props["barcode_numbers"]

    @property
    def barcode_idxs(self) -> np.ndarray:
        """Array of barcodes' index."""
        return self._barcodes_props["barcode_idxs"]

    @property
    def barcode_times(self) -> np.ndarray:
        """Array of barcodes' times."""
        return self.times[self.barcode_idxs]

    @cached_property
    def timebase_idx(self):
        return TimeBase(
            event_times=self.barcode_idxs,
            event_ids=self.barcode_numbers,
            span=len(self.array),
        )

    @cached_property
    def timebase_times(self):

        return TimeBase(
            event_times=self.barcode_times,
            event_ids=self.barcode_numbers,
            span=(self.times[0], self.times[-1], 1 / self.fs),
        )

    @cached_property
    def _barcodes_props(self) -> None:
        """Analyzes the digital signal to extract the barcodes. Lengthy function inherited from the barcode
        system developers, could be optimized in the future. A bit slow - lots of loops involved.
        """
        wrap_duration = 3 * self.WRAP_DURATION_MS  # Off-On-Off
        total_barcode_duration = self.NBITS * self.BAR_DURATION_MS + 2 * wrap_duration

        # Tolerance conversions
        min_wrap_duration = (
            self.WRAP_DURATION_MS - self.WRAP_DURATION_MS * self.GLOBAL_TOLERANCE
        )
        max_wrap_duration = (
            self.WRAP_DURATION_MS + self.WRAP_DURATION_MS * self.GLOBAL_TOLERANCE
        )
        sample_conversion = 1000 / self.fs  # Convert sampling rate to msec

        # Signal extraction and barcode analysis
        indexed_times = self.all_events

        # Find time difference between index values (ms), and extract barcode wrappers.
        events_time_diff = np.diff(indexed_times) * sample_conversion  # convert to ms

        wrapper_array = indexed_times[
            np.where(
                np.logical_and(
                    min_wrap_duration < events_time_diff,
                    events_time_diff < max_wrap_duration,
                )
            )[0]
        ]

        # Isolate the wrapper_array to wrappers with ON values, to avoid any
        # "OFF wrappers" created by first binary value.
        false_wrapper_check = (
            np.diff(wrapper_array) * sample_conversion
        )  # Convert to ms
        # Locate indices where two wrappers are next to each other.
        false_wrappers = np.where(false_wrapper_check < max_wrap_duration)[0]
        # Delete the "second" wrapper (it's an OFF wrapper going into an ON bar)
        wrapper_array = np.delete(wrapper_array, false_wrappers + 1)

        # Find the barcode "start" wrappers, set these to wrapper_start_times, then
        # save the "real" barcode start times to signals_barcode_start_idxs, which
        # will be combined with barcode values for the output .npy file.
        wrapper_time_diff = np.diff(wrapper_array) * sample_conversion  # convert to ms
        barcode_index = np.where(wrapper_time_diff < total_barcode_duration)[0]
        wrapper_start_times = wrapper_array[barcode_index]
        signals_barcode_start_idxs = (
            wrapper_start_times - self.WRAP_DURATION_MS / sample_conversion
        )
        # Actual barcode start is 10 ms before first 10 ms ON value.

        # Convert wrapper_start_times, on_times, and off_times to ms
        wrapper_start_times = wrapper_start_times * sample_conversion
        on_times = self.onsets * sample_conversion
        off_times = self.offsets * sample_conversion

        signals_barcodes = []

        for start_time in wrapper_start_times:
            oncode = on_times[
                np.where(
                    np.logical_and(
                        on_times > start_time,
                        on_times < start_time + total_barcode_duration,
                    )
                )[0]
            ]
            offcode = off_times[
                np.where(
                    np.logical_and(
                        off_times > start_time,
                        off_times < start_time + total_barcode_duration,
                    )
                )[0]
            ]
            curr_time = (
                offcode[0] + self.WRAP_DURATION_MS
            )  # Jumps ahead to start of barcode
            bits = np.zeros((self.NBITS,))
            interbit_on = False  # Changes to "True" during multiple ON bars

            for bit in range(0, self.NBITS):
                next_on = np.where(
                    oncode >= (curr_time - self.BAR_DURATION_MS * self.GLOBAL_TOLERANCE)
                )[0]
                next_off = np.where(
                    offcode
                    >= (curr_time - self.BAR_DURATION_MS * self.GLOBAL_TOLERANCE)
                )[0]

                if next_on.size > 1:  # Don't include the ending wrapper
                    next_on = oncode[next_on[0]]
                else:
                    next_on = start_time + self.INTER_BC_INTERVAL_MS

                if next_off.size > 1:  # Don't include the ending wrapper
                    next_off = offcode[next_off[0]]
                else:
                    next_off = start_time + self.INTER_BC_INTERVAL_MS

                # Recalculate min/max bar duration around curr_time
                min_bar_duration = (
                    curr_time - self.BAR_DURATION_MS * self.GLOBAL_TOLERANCE
                )
                max_bar_duration = (
                    curr_time + self.BAR_DURATION_MS * self.GLOBAL_TOLERANCE
                )

                if min_bar_duration <= next_on <= max_bar_duration:
                    bits[bit] = 1
                    interbit_on = True
                elif min_bar_duration <= next_off <= max_bar_duration:
                    interbit_on = False
                elif interbit_on == True:
                    bits[bit] = 1

                curr_time += self.BAR_DURATION_MS

            barcode = 0

            for bit in range(0, self.NBITS):  # least sig left
                barcode += bits[bit] * pow(2, bit)

            signals_barcodes.append(barcode)

        # Create merged array with timestamps stacked above their barcode values
        barcode_numbers = np.array(signals_barcodes, dtype=np.uint64)
        barcode_idxs = np.array(signals_barcode_start_idxs, dtype=np.uint64)

        return dict(barcode_numbers=barcode_numbers, barcode_idxs=barcode_idxs)

    def _map_to(self, target_barcode_signal: "BarcodeSignal", to_map) -> "TimeBaseMap":
        """Map the indexes of the BarcodeSignal to another BarcodeSignal.

        Parameters
        ----------
        other_barcode_signal : BarcodeSignal object
            The BarcodeSignal object to which the indexes will be mapped.
        indexes : int | np.ndarray
            The indexes to be mapped.

        Returns
        -------
        int | np.ndarray :
            The mapped indexes.

        """
        if to_map == "idxs":
            return self.timebase_idx._map_to(target_barcode_signal.timebase_idx)
        elif to_map == "times":
            return self.timebase_times._map_to(target_barcode_signal.timebase_times)
        else:
            raise ValueError("to_map must be 'idxs' or 'times'.")

    def _transform_to(
        self, target_barcode_signal: "BarcodeSignal", data, to_map
    ) -> int | np.ndarray:
        """Transform the indexes of the BarcodeSignal to another BarcodeSignal."""
        timebasemap = self._map_to(
            target_barcode_signal=target_barcode_signal, to_map=to_map
        )
        return timebasemap.transform(data)

    def transform_idxs_to(
        self, target_barcode_signal: "BarcodeSignal", indexes: int | np.ndarray
    ) -> int | np.ndarray:
        """Map the indexes of the BarcodeSignal to another BarcodeSignal.

        Parameters
        ----------
        target_barcode_signal : BarcodeSignal object
            The BarcodeSignal object to which the indexes will be mapped.
        indexes : int | np.ndarray
            The indexes to be mapped.

        Returns
        -------
        int | np.ndarray :
            The mapped indexes.

        """
        return self._transform_to(target_barcode_signal, indexes, "idxs")

    def transform_times_to(
        self, other_barcode_signal: "BarcodeSignal", times: int | np.ndarray
    ) -> int | np.ndarray:
        return self._transform_to(other_barcode_signal, times, "times")

    def _resample_to(
        self, target_barcode_signal: "BarcodeSignal", own_timebase_data, to_map
    ) -> np.ndarray:
        """Resample the data in self timebase to another BarcodeSignal.

        Parameters
        ----------
        target_barcode_signal : BarcodeSignal object
            The BarcodeSignal object describing the timebase to which the data will be resampled.
        own_timebase_data : np.ndarray
            The data to be resampled.

        Returns
        -------
        np.ndarray :
            The resampled data.

        """
        timebasemap = self._map_to(target_barcode_signal, to_map)
        return timebasemap.resample(own_timebase_data)

    def resample_to(
        self, target_barcode_signal: "BarcodeSignal", own_timebase_data: np.ndarray
    ) -> np.ndarray:
        return self._resample_to(target_barcode_signal, own_timebase_data, "times")

    # I believe this can be confusing, leaving out for now in the future:

    # def resample_on_times_to(
    #     self, target_barcode_signal: "BarcodeSignal", times: int | np.ndarray
    # ) -> np.ndarray:
    #     return self._resample_to(target_barcode_signal, times, "times")

    @classmethod
    def from_digital_signal(cls, digital_signal: DigitalSignal) -> "BarcodeSignal":
        """Create a BarcodeSignal object from a DigitalSignal object.

        Parameters
        ----------
        digital_signal : DigitalSignal
            The DigitalSignal object to be converted to a BarcodeSignal.

        Returns
        -------
        BarcodeSignal
            The BarcodeSignal object created from the DigitalSignal.

        """
        return cls(digital_signal.array, fs=digital_signal.fs)
