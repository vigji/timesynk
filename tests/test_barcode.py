# %%
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib import widgets

from timesynk.signals import BarcodeSignal

test_barcode_filename = Path(__file__).parent / "assets" / "barcode.npy"
fs_test = 1000
actual_barcode_idxs = [
    3772,
    8773,
    13773,
    18774,
    23775,
    28776,
    33776,
    38777,
    43778,
    48778,
]
actual_barcode_times = np.array(actual_barcode_idxs) / fs_test
actual_barcode_numbers = [
    26687,
    26688,
    26689,
    26690,
    26691,
    26692,
    26693,
    26694,
    26695,
    26696,
]


@pytest.fixture
def barcode_raw_data():
    return np.load(test_barcode_filename)


def test_barcode_instantiation(barcode_raw_data):
    barc = BarcodeSignal(barcode_raw_data, fs=fs_test)
    assert barc.fs == 1000
    assert np.allclose(barc.barcode_times, actual_barcode_times)
    assert np.allclose(barc.barcode_numbers, actual_barcode_numbers)
    assert np.allclose(barc.barcode_idxs, actual_barcode_idxs)


def test_barcode_instantiation_no_fs(barcode_raw_data):
    with pytest.raises(ValueError):
        BarcodeSignal(barcode_raw_data, fs=None)


@pytest.mark.parametrize("fs_scale_factor", [1, 2])
@pytest.mark.parametrize("offset", [0, 100, 1000, 5000, -3000])
def test_barcode_params(barcode_raw_data, offset, fs_scale_factor):
    base_offset = 10000
    n_pts = len(barcode_raw_data)
    range1 = np.arange(base_offset, n_pts - base_offset, dtype=int)

    barcode1 = BarcodeSignal(barcode_raw_data[range1], fs=fs_test)
    barcode2 = BarcodeSignal(
        barcode_raw_data[range1 + offset][::fs_scale_factor],
        fs=fs_test / fs_scale_factor,
    )

    # print(barcode1.map_indexes_to(barcode2, 0))
    assert np.isclose(
        barcode1.transform_idxs_to(barcode2, 0), -offset / fs_scale_factor, atol=0.5
    )
    # assert np.isclose(barcode2.map_indexes_to(barcode1, 0), offset, atol=0.5)


def test_identity_resampling(barcode_raw_data):
    barcode1 = BarcodeSignal(barcode_raw_data, fs=fs_test)
    barcode2 = BarcodeSignal(barcode_raw_data, fs=fs_test)

    barcode1data_to_barcode2data = barcode1.resample_to(barcode2, barcode1.array)
    barcode2data_to_barcode1data = barcode2.resample_to(
        barcode1, barcode1data_to_barcode2data
    )

    # assert np.allclose(barcode1.array, barcode2data_to_barcode1data)
    print(sum(barcode1.array - barcode2data_to_barcode1data))
    assert (
        sum(barcode1.array - barcode2data_to_barcode1data) < len(barcode1.array) * 0.01
    )

    assert np.isclose(barcode1.transform_idxs_to(barcode2, 0), 0, atol=0.5)
    assert np.isclose(barcode2.transform_times_to(barcode1, 0), 0, atol=0.5)


def test_transform_times(barcode_raw_data):
    barcode_raw_data = barcode_raw_data[:47000]  # to avoid issue with partial reading
    barcode1 = BarcodeSignal(barcode_raw_data, fs=fs_test)
    barcode2 = BarcodeSignal(barcode_raw_data[::2], fs=fs_test / 2)

    assert np.allclose(
        barcode1.barcode_times,
        barcode2.transform_times_to(barcode1, barcode2.barcode_times),
        rtol=0.1,
    )
    assert np.allclose(
        barcode2.barcode_times,
        barcode1.transform_times_to(barcode2, barcode1.barcode_times),
        rtol=0.1,
    )


def test_transform_idxs(barcode_raw_data):
    barcode_raw_data = barcode_raw_data[
        :47000
    ]  # crop to avoid issues with partial reading
    barcode1 = BarcodeSignal(barcode_raw_data, fs=fs_test)
    barcode2 = BarcodeSignal(barcode_raw_data[::2], fs=fs_test / 2)

    assert np.allclose(
        barcode1.barcode_idxs,
        barcode2.transform_idxs_to(barcode1, barcode2.barcode_idxs),
        rtol=0.5,
    )
    assert np.allclose(
        barcode2.barcode_idxs,
        barcode1.transform_idxs_to(barcode2, barcode1.barcode_idxs),
        rtol=0.5,
    )


# Resample to signal with different sampling frequency:
def test_freqchange_resampling(barcode_raw_data):
    freq_change = 2
    barcode1 = BarcodeSignal(barcode_raw_data, fs=fs_test)
    barcode2 = BarcodeSignal(barcode_raw_data[::freq_change], fs=fs_test / freq_change)

    barcode1data_to_barcode2data = barcode1.resample_to(barcode2, barcode1.array)

    barcode2data_to_barcode1data = barcode2.resample_to(
        barcode1, barcode1data_to_barcode2data
    )

    # test difference to allow for +/-1 shifts in resampling
    assert (
        sum(barcode1.array - barcode2data_to_barcode1data) < len(barcode1.array) * 0.01
    )

# resample to signal with broader scope
def test_freqchange_broader_resampling(barcode_raw_data):
    n_pts = len(barcode_raw_data)
    signal = np.random.randn(n_pts)
    b1_slice = slice(n_pts//4, 3*n_pts//4)
    freq_change = 2
    barcode1 = BarcodeSignal(barcode_raw_data[b1_slice], fs=fs_test)
    barcode2 = BarcodeSignal(barcode_raw_data[::freq_change], fs=fs_test / freq_change)

    barcode1data_to_barcode2data = barcode1.resample_to(barcode2, signal[b1_slice])

    barcode2data_to_barcode1data = barcode2.resample_to(
        barcode1, barcode1data_to_barcode2data
    )

    # test difference to allow for +/-1 shifts in resampling
    assert (
        sum(barcode1.array - barcode2data_to_barcode1data) < len(barcode1.array) * 0.01
    )


if __name__ == "__main__":
    # %%
    # %matplotlib widget

    barc = BarcodeSignal(np.load(test_barcode_filename), fs=fs_test)
    # print(", ".join([str(i) for i in barc.barcode_idxs]))
    barcode_raw_data = np.load(test_barcode_filename)[:47000]
    from matplotlib import pyplot as plt

    freq_change = 2
    barcode1 = BarcodeSignal(barcode_raw_data, fs=fs_test)
    barcode2 = BarcodeSignal(barcode_raw_data[::freq_change], fs=fs_test / freq_change)

    barcode1data_to_barcode2data = barcode1.resample_to(barcode2, barcode1.array)
    # print(len(barcode1.array), len(barcode1data_to_barcode2data))
    barcode2data_to_barcode1data = barcode2.resample_to(
        barcode1, barcode1data_to_barcode2data
    )

    _map = barcode1._map_to(barcode2, "idxs")
    print(_map)
    # print(len(_map.source_timestamps))
    # print(len(_map.target_timestamps))
    # _map.resample()
    # assert np.allclose(barcode1.array, barcode2data_to_barcode1data)

    # plt.figure()
    # plt.plot(barcode1.array)
    # plt.plot(barcode2.array + 1)
    # plt.plot(barcode1data_to_barcode2data + 2)
    # #plt.show()

    n_pts = len(barcode_raw_data)
    data = np.random.randn(n_pts)
    b1_slice = slice(n_pts//3, 2*n_pts//3)
    freq_change = 2
    barcode1 = BarcodeSignal(barcode_raw_data[b1_slice], fs=fs_test)
    barcode2 = BarcodeSignal(barcode_raw_data[::freq_change], fs=fs_test / freq_change)

    barcode1_to_barcode2data = barcode1.resample_to(barcode2, data[b1_slice])

    barcode2data_to_barcode1data = barcode2.resample_to(
        barcode1, barcode1_to_barcode2data
    )

    # plt.figure()
    # plt.plot(data[b1_slice])
    # plt.plot(barcode1.array)
    # plt.plot(barcode2data_to_barcode1data)
    # plt.show()
    print(np.corrcoef())
    # # plt.plot(barcode1_to_barcode2data+6)
    # # plt.plot(barcode2.array + 6)
    # plt.show()

