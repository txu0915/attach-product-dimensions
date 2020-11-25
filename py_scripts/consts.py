try:
    from model_mart_file_utils import ParameterizedFileLoc
except ImportError as e:
    print("In consts.py: Skipping model_mart_file_utils import: %s" % str(e))

    class ParameterizedFileLoc:
        ENV = None
        DATE = None

        def __init__(self, l): pass

IDM_ATTRIBUTES_LOC = ParameterizedFileLoc(["gs://hd-personalization-prod-data/MergedFullFeed/",
                                           ParameterizedFileLoc.DATE,
                                           "/ItemAttributes/*"])

MERGED_IMAGES_OUTPUT_SUFFIX = 'merged_images'
DIMENSION_OUTPUT_SUFFIX = "dimensions"
IMAGES_OUTPUT_SUFFIX = "output_images"
DEBUG_IMGS_SUFFIX = "debug_images"
