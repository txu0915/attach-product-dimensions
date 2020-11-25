import sys
sys.path.extend(['/Users/tianlongxu/Documents/My_Projects/Adding_Dimensions/py_scripts'])

#from model_mart_logging import *
from py_scripts.model_mart_file_utils import *
from py_scripts.model_mart_img_utils import *
from py_scripts.create_imgs import *
from py_scripts.consts import *
from py_scripts.utils import *
from pyspark.sql import SparkSession
from pyspark import StorageLevel
import pyspark.sql.functions as F
from pyspark.sql.types import *
from google.cloud import bigquery
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from py_scripts.utils_canopy import *

drawing_config_local_filepath = '/Users/tianlongxu/Documents/My_Projects/Adding_Dimensions/configs/image_drawing_config.json'
font_local_filepath = '/Users/tianlongxu/Documents/My_Projects/Adding_Dimensions/configs/HelveticaNeue-Medium.otf'
output_dir = './Results/test-nov-12/'
with open(drawing_config_local_filepath, 'r') as fp:
    config = json.load(fp)

myData = pd.read_csv('./input-data/indoor-chandelier-images-dims.csv')
for i, row in myData.iterrows():
    img_guid,omsid = row.image_url.split("/")[4], row.omsid
    print(i, omsid,row.image_url)
    image = get_image(img_guid)
    if whole_product_in_image(image):
        dimension_map = {"height": float(str(row.height).split()[0]),"width": float(str(row.width).split()[0]),
                         "depth": float(str(row.depth).split()[0]),"canopy":float(str(row.canopy).split()[0])}
        image_name = str(omsid) +","+ row.style
        image = add_canopy_dims_bar(image, dimension_map, config, font_local_filepath)
        save_image_with_three_major_dims_bars(image,image_name,output_dir,config, dimension_map, font_local_filepath)
    # if i==50:
    #     break
print('Job done.')


################## test
image = get_image('2736a9cb-a815-433e-b2a0-fc583bbfd16e')
dimension_map = {"height": 17,"width": 14,"depth":13 ,"canopy":5}
image_name = "fail-2"
image = add_canopy_dims_bar(image, dimension_map, config, font_local_filepath)
save_image_with_three_major_dims_bars(image,image_name,output_dir,config, dimension_map, font_local_filepath)

#whole_product_in_image(img)
