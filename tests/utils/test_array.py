from typing import Tuple
import numpy as np
import pytest

from ertk.utils.array import (
    batch_arrays_by_length,
    check_3d,
    clip_arrays,
    flat_to_inst,
    frame_array,
    frame_arrays,
    inst_to_flat,
    make_array_array,
    pad_array,
    pad_arrays,
    shuffle_multiple,
    transpose_time,
)


def test_make_array_array_vlen():
    arr = make_array_array([np.arange(i) for i in [3, 5, 7, 11]])
    assert arr.dtype == object
    assert arr.shape == (4,)
    assert [len(x) for x in arr] == [3, 5, 7, 11]


def test_make_array_array_3d():
    arr = make_array_array(list(np.arange(1000).reshape(10, 10, 10)))
    assert arr.shape == (10,)
    assert all(x.shape == (10, 10) for x in arr)


def test_check_3d():
    check_3d(np.arange(1000).reshape(10, 10, 10))
    check_3d([np.arange(3 * i).reshape(3, i) for i in [3, 5, 7, 11]])
    with pytest.raises(ValueError):
        check_3d(np.arange(100).reshape(10, 10))


@pytest.mark.parametrize(
    ["shape", "frame_size", "frame_shift", "pad", "axis", "expected_shape"],
    [
        ((100,), 30, 10, False, 0, (8, 30)),
        ((107,), 30, 10, True, 0, (9, 30)),
        ((100, 10), 30, 10, False, 0, (8, 30, 10)),
        ((100, 10), 30, 10, True, 0, (8, 30, 10)),
        ((107, 10), 30, 10, False, 0, (8, 30, 10)),
        ((107, 10), 30, 10, True, 0, (9, 30, 10)),
        ((110, 10), 30, 10, False, 0, (9, 30, 10)),
        ((47, 10), 13, 5, False, 0, (7, 13, 10)),
        ((47, 10), 13, 5, True, 0, (8, 13, 10)),
        ((20, 100, 10), 30, 10, False, 1, (20, 8, 30, 10)),
        ((20, 100, 10), 30, 10, True, 1, (20, 8, 30, 10)),
        ((1, 37, 10), 50, 50, True, 1, (1, 1, 50, 10)),
        ((37, 10), 50, 50, True, 0, (1, 50, 10)),
        ((37, 10), 50, 10, True, 0, (1, 50, 10)),
    ],
)
def test_frame_array_shape(
    shape: Tuple[int],
    frame_size: int,
    frame_shift: int,
    pad: bool,
    axis: int,
    expected_shape: Tuple[int],
):
    rng = np.random.default_rng()
    x = rng.random(size=shape)
    framed = frame_array(x, frame_size, frame_shift, pad=pad, axis=axis, copy=True)
    assert framed.shape == expected_shape
    if not pad:
        framed = frame_array(x, frame_size, frame_shift, pad=pad, axis=axis, copy=False)
        assert framed.shape == expected_shape


@pytest.mark.parametrize(
    ["shape", "frame_size", "frame_shift", "axis"],
    [
        ((1, 37, 10), 50, 50, 1),
        ((37, 10), 50, 50, 0),
        ((37, 10), 50, 10, 0),
    ],
)
def test_frame_array_small_nopad(
    shape: Tuple[int], frame_size: int, frame_shift: int, axis: int
):
    rng = np.random.default_rng()
    x = rng.random(size=shape)
    with pytest.raises(ValueError, match="sequence is shorter than frame_size"):
        frame_array(x, frame_size, frame_shift, pad=False, axis=axis)


@pytest.mark.parametrize(
    ["shape", "frame_size", "frame_shift", "axis"],
    [
        ((107,), 30, 10, 0),
        ((47, 10), 13, 5, 0),
        ((47, 10), 13, 5, 0),
    ],
)
def test_frame_array_pad_nocopy(
    shape: Tuple[int], frame_size: int, frame_shift: int, axis: int
):
    rng = np.random.default_rng()
    x = rng.random(size=shape)
    with pytest.raises(ValueError, match="pad must be False"):
        frame_array(x, frame_size, frame_shift, pad=True, axis=axis, copy=False)


def test_frame_array_content_no_pad():
    rng = np.random.default_rng()
    x = rng.random(size=(105, 10))
    framed = frame_array(x, 30, 10, pad=False, axis=0)
    assert np.array_equal(framed[0], x[0:30])
    assert np.array_equal(framed[1], x[10:40])
    assert np.array_equal(framed[7], x[70:100])


def test_frame_array_content_pad():
    rng = np.random.default_rng()
    x = rng.random(size=(107, 10))
    framed = frame_array(x, 30, 10, pad=True, axis=0)
    assert np.array_equal(framed[0], x[0:30])
    assert np.array_equal(framed[1], x[10:40])
    assert np.array_equal(framed[7], x[70:100])
    assert np.array_equal(framed[8, 0:27], x[80:107])
    assert np.all(framed[8, 27:30] == 0)


def test_frame_arrays_cont():
    arrays = [np.arange(i).reshape(-1, 1) for i in [100, 207, 150]]
    framed = frame_arrays(arrays, 30, 17)
    assert framed.shape == (3, 11, 30, 1)
    assert np.array_equal(framed[0, 0, :, 0], np.arange(30))
    assert np.array_equal(framed[1, 10, :, 0], np.arange(170, 200))


def test_frame_arrays_vlen():
    arrays = [np.arange(i).reshape(-1, 1) for i in [100, 207, 150]]
    framed = frame_arrays(arrays, 30, 17, vlen=True)
    assert framed.shape == (3,)
    assert np.array_equal(framed[0][0, :, 0], np.arange(30))
    assert np.array_equal(framed[1][10, :, 0], np.arange(170, 200))


def test_frame_arrays_2d():
    arrays = [np.arange(i) for i in [100, 207, 150]]
    with pytest.raises(ValueError):
        frame_arrays(arrays, 30, 10)


@pytest.mark.parametrize(
    ["shape", "to_multiple", "to_size", "axis", "expected_shape"],
    [
        ((107, 10), 10, None, 0, (110, 10)),
        ((107, 10), 5, None, 0, (110, 10)),
        ((107, 10), 20, None, 0, (120, 10)),
        ((55, 10), None, 60, 0, (60, 10)),
        ((30, 10), None, 60, 0, (60, 10)),
        ((5, 30, 10), None, 60, 1, (5, 60, 10)),
        ((5, 30, 10), None, 20, 2, (5, 30, 20)),
        ((5, 30, 10), 50, None, 1, (5, 50, 10)),
    ],
)
def test_pad_array(
    shape: Tuple[int],
    to_multiple: int,
    to_size: int,
    axis: int,
    expected_shape: Tuple[int],
):
    rng = np.random.default_rng()
    x = rng.random(size=shape)
    padded = pad_array(x, to_multiple, to_size, axis=axis)
    assert padded.shape == expected_shape
    idx_same = tuple(slice(0, x) for x in shape)
    assert np.array_equal(padded[idx_same], x)
    idx_zeros = [slice(None, None) for _ in shape]
    idx_zeros[axis] = slice(shape[axis], None)
    assert np.all(padded[tuple(idx_zeros)] == 0)


def test_pad_arrays_type():
    arrays = [np.arange(i) for i in [100, 207, 150]]
    padded = pad_arrays(arrays, 32)
    assert isinstance(padded, list)
    arrays = make_array_array([np.arange(i) for i in [100, 207, 150]])
    padded = pad_arrays(arrays, 32)
    assert isinstance(padded, np.ndarray)


def test_pad_arrays_vlen():
    arrays = [np.arange(i) for i in [100, 207, 150]]
    padded = pad_arrays(arrays, 32)
    assert len(padded) == 3
    assert padded[0].shape == (128,)


def test_pad_arrays_3d():
    x = np.arange(1000).reshape(10, 10, 10)
    padded = pad_arrays(x, 32)
    assert padded.shape == (10, 32, 10)


def test_pad_arrays_cont():
    arrays = make_array_array([np.arange(i) for i in [10, 11, 12]])
    padded = pad_arrays(arrays, 32)
    assert padded.shape == (3, 32)


def test_clip_arrays_type():
    arrays = [np.arange(i) for i in [100, 207, 150]]
    clipped = clip_arrays(arrays, 130)
    assert isinstance(clipped, list)
    arrays = make_array_array([np.arange(i) for i in [100, 207, 150]])
    clipped = clip_arrays(arrays, 130)
    assert isinstance(clipped, np.ndarray)


def test_clip_arrays_vlen():
    arrays = [np.arange(i) for i in [100, 207, 150]]
    clipped = clip_arrays(arrays, 130)
    assert clipped[0].shape == (100,)
    assert clipped[1].shape == (130,)
    assert clipped[2].shape == (130,)


def test_clip_arrays_3d():
    arrays = np.arange(1000).reshape(10, 10, 10)
    clipped = clip_arrays(arrays, 7)
    assert clipped.shape == (10, 7, 10)


def test_clip_arrays_cont():
    arrays = make_array_array([np.arange(i) for i in [100, 207, 150]])
    clipped = clip_arrays(arrays, 80)
    assert clipped.shape == (3, 80)


def test_transpose_time_exception():
    with pytest.raises(ValueError):
        transpose_time([np.arange(i) for i in [100, 207, 150]])


def test_transpose_time_3d():
    rng = np.random.default_rng()
    x = rng.random(size=(10, 100, 13))
    transposed = transpose_time(x.copy())
    assert np.array_equal(transposed, x.transpose(0, 2, 1))


def test_transpose_time_vlen():
    arrays = [np.arange(i * 3).reshape(i, 3) for i in [100, 207, 150]]
    transposed = transpose_time(arrays)
    assert transposed[0].shape == (3, 100)
    assert transposed[1].shape == (3, 207)
    assert np.array_equal(transposed[2], np.arange(150 * 3).reshape(150, 3).T)


def test_shuffle_multiple():
    array1 = [0, 1, 6, 7]
    array2 = [12, 17, 14, 13]
    sh1, sh2 = shuffle_multiple(array1, array2, numpy_indexing=False)
    assert set(sh1) == set(array1)
    assert set(sh2) == set(array2)


def test_shuffle_multiple_exception():
    array1 = [0, 1, 6, 7]
    array2 = [12, 17, 14]
    with pytest.raises(ValueError):
        shuffle_multiple(array1, array2, numpy_indexing=False)


def test_batch_arrays_by_length():
    arrays = [np.arange(i) for i in [20, 5, 5, 20]]
    batch_x, batch_y = batch_arrays_by_length(
        arrays, np.arange(4), batch_size=2, shuffle=False
    )
    assert batch_x.shape == (2,)
    assert batch_x[0].shape == (2, 5)
    assert batch_x[1].shape == (2, 20)
    assert batch_y.shape == (2,)
    assert list(batch_y[0]) == [1, 2]
    assert list(batch_y[1]) == [0, 3]

    batch_x, batch_y = batch_arrays_by_length(
        arrays, np.arange(4), batch_size=2, shuffle=False, uniform_batch_size=True
    )
    assert batch_y.shape == (2, 2)


def test_batch_arrays_by_length_nonuniform():
    arrays = [np.arange(i) for i in [20, 5, 5, 22]]
    batch_x, batch_y = batch_arrays_by_length(
        arrays, np.arange(4), batch_size=2, shuffle=False
    )
    assert batch_x.shape == (3,)
    assert batch_x[0].shape == (2, 5)
    assert batch_x[1].shape == (1, 20)
    assert batch_x[2].shape == (1, 22)
    assert batch_y.shape == (3,)
    assert list(batch_y[0]) == [1, 2]
    assert list(batch_y[1]) == [0]
    assert list(batch_y[2]) == [3]


def test_flat_to_inst_2d():
    flat = np.arange(1000).reshape(100, 10)
    inst = flat_to_inst(flat, [1] * 100)
    assert inst is flat


def test_flat_to_inst_3d():
    flat = np.arange(1000).reshape(100, 10)
    inst = flat_to_inst(flat, [20] * 5)
    assert np.array_equal(inst, flat.reshape(5, 20, 10))


def test_flat_to_inst_vlen():
    flat = np.arange(1000).reshape(100, 10)
    inst = flat_to_inst(flat, [30, 10, 10, 40, 10])
    assert inst.shape == (5,)
    assert np.array_equal(inst[0], flat[0:30])
    assert np.array_equal(inst[1], flat[30:40])
    assert np.array_equal(inst[4], flat[90:100])


def test_inst_to_flat_2d():
    inst = np.arange(1000).reshape(100, 10)
    flat, slices = inst_to_flat(inst)
    assert flat is inst
    assert list(slices) == [1] * 100


def test_inst_to_flat_3d():
    inst = np.arange(1000).reshape(5, 20, 10)
    flat, slices = inst_to_flat(inst)
    assert np.array_equal(flat, inst.reshape(100, 10))
    assert list(slices) == [20] * 5


def test_inst_to_flat_vlen():
    inst = make_array_array([np.arange(i).reshape(-1, 1) for i in [20, 5, 5, 22]])
    flat, slices = inst_to_flat(inst)
    assert flat.shape == (52, 1)
    assert list(slices) == [20, 5, 5, 22]
    assert np.array_equal(flat, np.concatenate(inst))
