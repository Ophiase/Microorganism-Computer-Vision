import os

###################################################################################


DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"
ANALYSIS_GRAPHICS_PATH = os.path.join(OUTPUT_FOLDER, "analysis")

DEFAULT_VIDEO = "342843.avi"
DEFAULT_VIDEO_INTERVAL = (0, 40)
DEFAULT_VIDEO_RESOLUTION = (685, 512)

#########################################


PREPROCESSED_FOLDER = os.path.join(DATA_FOLDER, "preprocessed")
BOUNDING_BOX_FOLDER = os.path.join(DATA_FOLDER, "bounding_box")
TRACKING_FOLDER = os.path.join(DATA_FOLDER, "tracking")

#########################################


DEFAULT_FONT = "Arial.ttf"
