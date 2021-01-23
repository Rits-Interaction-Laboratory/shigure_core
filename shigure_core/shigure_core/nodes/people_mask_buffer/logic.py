from typing import Tuple

import numpy as np

from shigure_core.nodes.people_mask_buffer.people_mask_buffer_frames import PeopleMaskBufferFrames


class PeopleMaskBufferLogic:
    """人物領域マスクバッファロジック."""

    @staticmethod
    def execute(buffer_frames: PeopleMaskBufferFrames) -> Tuple[bool, np.ndarray]:
        if not buffer_frames.is_full():
            return False, np.zeros((1, 1))

        return True, (buffer_frames.get_people_mask() * 255).astype(np.uint8)
