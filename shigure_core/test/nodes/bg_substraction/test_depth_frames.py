import numpy as np
from shigure.nodes.bg_subtraction.depth_frames import DepthFrames


def test_add_frame():
    frame = np.array([[1, 2],
                      [3, 4]])

    depth_frames = DepthFrames(100)
    depth_frames.add_frame(frame)

    assert not depth_frames.is_full()
    assert np.all(depth_frames.frames[0] == frame)
    assert np.all(depth_frames.sum_each_pixel == frame)


def test_add_frame_with_max_size():
    frame1 = np.array([[1, 1],
                       [1, 1]])
    frame2 = np.array([[1, 2],
                       [3, 4]])

    depth_frames = DepthFrames(1)
    depth_frames.add_frame(frame1)
    depth_frames.add_frame(frame2)

    assert depth_frames.is_full()
    assert np.all(depth_frames.frames[0] == frame2)
    assert np.all(depth_frames.sum_each_pixel == frame2)


def test_is_full():
    frame = np.array([[0, 0],
                      [0, 0]])

    depth_frames = DepthFrames(3)

    depth_frames.add_frame(frame)
    assert not depth_frames.is_full()

    depth_frames.add_frame(frame)
    assert not depth_frames.is_full()

    depth_frames.add_frame(frame)
    assert depth_frames.is_full()


def test_get_average():
    frame0 = np.array([[100, 100],
                       [100, 100]])
    frame1 = np.array([[0, 0],
                       [0, 50]])
    frame2 = np.array([[0, 0],
                       [3, 30]])
    frame3 = np.array([[0, 3],
                       [3, 10]])

    depth_frames = DepthFrames(3, 2)

    depth_frames.add_frame(frame0)
    depth_frames.add_frame(frame1)
    depth_frames.add_frame(frame2)
    depth_frames.add_frame(frame3)

    expect_frame = np.array([[0, 3],
                             [3, 30]], np.float32)

    assert np.all(depth_frames.get_average() == expect_frame)


def test_get_average_of_square():
    frame1 = np.array([[0, 0],
                       [0, 2]])
    frame2 = np.array([[0, 0],
                       [3, 2]])
    frame3 = np.array([[0, 3],
                       [3, 2]])

    depth_frames = DepthFrames(3, 2)

    depth_frames.add_frame(frame1)
    depth_frames.add_frame(frame2)
    depth_frames.add_frame(frame3)

    expect_frame = np.array([[0, 9],
                             [9, 4]])

    assert np.all(depth_frames.get_average_of_square() == expect_frame)


def test_get_var():
    frame1 = np.array([[0, 0],
                       [-2, 2]])
    frame2 = np.array([[0, 2],
                       [1, 2]])
    frame3 = np.array([[4, 4],
                       [1, 2]])

    depth_frames = DepthFrames(3, 2)

    depth_frames.add_frame(frame1)
    depth_frames.add_frame(frame2)
    depth_frames.add_frame(frame3)

    expect_frame = np.array([[0, 1],
                             [3, 0]])

    assert np.all(depth_frames.get_var() == expect_frame)


def test_get_valid_pixel():
    frame1 = np.array([[0, 0],
                       [0, 0]])
    frame2 = np.array([[0, 0],
                       [0, 2]])
    frame3 = np.array([[0, 0],
                       [3, 4]])
    frame4 = np.array([[0, 3],
                       [3, 2]])

    depth_frames = DepthFrames(4, 2)

    depth_frames.add_frame(frame1)
    depth_frames.add_frame(frame2)
    depth_frames.add_frame(frame3)
    depth_frames.add_frame(frame4)

    expect = np.array([[False, False],
                       [True, True]])

    assert np.all(depth_frames.get_valid_pixel() == expect)
