import logging
from typing import Tuple


class PipelineConfig:
    """Hardware-tuned configuration for an NVIDIA RTX 5060 (8GB VRAM)."""
    DEFAULT_MAX_FRAMES: int = 65
    HARD_CAP_MAX_FRAMES: int = 100

    DEFAULT_POISSON_DEPTH: int = 10
    HARD_CAP_POISSON_DEPTH: int = 11

    @classmethod
    def sanitize_parameters(cls, max_frames: int, poisson_depth: int) -> Tuple[int, int]:
        sanitized_frames = max(10, min(max_frames, cls.HARD_CAP_MAX_FRAMES))
        sanitized_depth = max(7, min(poisson_depth, cls.HARD_CAP_POISSON_DEPTH))
        if sanitized_frames != max_frames:
            logging.warning(f"[CONFIG] Adjusted max_frames from {max_frames} to {sanitized_frames}.")
        if sanitized_depth != poisson_depth:
            logging.warning(f"[CONFIG] Adjusted poisson_depth from {poisson_depth} to {sanitized_depth}.")
        return sanitized_frames, sanitized_depth
