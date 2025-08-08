import re
import math
from functools import partial
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import cv2

def custom_collate(batch):
    # Filter out None values and don't stack tensors
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return batch