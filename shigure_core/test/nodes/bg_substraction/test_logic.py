import numpy as np
from shigure.nodes.bg_subtraction.depth_frames import DepthFrames
from shigure.nodes.bg_subtraction.logic import BgSubtractionLogic


def test_execute():
    logic = BgSubtractionLogic()

    frames = DepthFrames(3, 3)

    frames.add_frame(np.array([[0, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]]))
    frames.add_frame(np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]]))
    frames.add_frame(np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]]))

    current_frame = np.array([[200, 1, 1],
                              [1, 200, 1],
                              [1, 1, 1]])

    expect_data = np.array([[0, 0, 0],
                            [0, 255, 0],
                            [0, 0, 0]])

    result, data = logic.execute(frames, current_frame)

    assert result
    assert np.all(data == expect_data)
