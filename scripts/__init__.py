import time
from enum import Enum
from typing import Tuple, Union

import torch
import numpy as np

PROJECT_NAME = 'activeINR-S'

OPENCV_TO_OPENGL = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]
)

# NOTE: Functions for TIMING
def start_timing(use_cuda:bool=True) -> Tuple[Union[torch.cuda.Event, float], Union[torch.cuda.Event, None]]:
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.perf_counter()
        end = None
    return start, end

def end_timing(start:Union[torch.cuda.Event, float], end:Union[torch.cuda.Event, None], use_cuda:bool=True) -> float:
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        end = time.perf_counter()
        elapsed_time = end - start
        # Convert to milliseconds to have the same units
        # as torch.cuda.Event.elapsed_time
        elapsed_time = elapsed_time * 1000
    return elapsed_time

class GlobalState(Enum):
    REPLAY = 'REPLAY'
    AUTO_PLANNING = 'AUTO_PLANNING'
    MANUAL_PLANNING = 'MANUAL_PLANNING'
    MANUAL_CONTROL = 'MANUAL_CONTROL'
    POST_PROCESSING = 'POST_PROCESSING'
    POINTNAV = 'POINTNAV'
    PAUSE = 'PAUSE'
    QUIT = 'QUIT'
