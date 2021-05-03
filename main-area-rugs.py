import sys
sys.path.extend(['/Users/tianlongxu/Documents/My_Projects/Adding_Dimensions/py_scripts'])
import json
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
from py_scripts.model_mart_img_utils import *

working_dir = '/Users/tianlongxu/Documents/My_Projects/Adding_Dimensions/'
drawing_config_local_filepath = working_dir + 'configs/image_drawing_config.json'
font_local_filepath = working_dir + 'configs/HelveticaNeue-Medium.otf'
output_dir = working_dir + 'Results/apr-30-2021/'
input_dir = working_dir + 'input-data/'

with open(drawing_config_local_filepath, 'r') as fp:
    config = json.load(fp)

query = """
with dims as ( 
select area_rugs.OMSID, ia.attributevalue width, 
-- ia1.attributevalue depth, 
-- ia2.attributevalue height  
from `analytics-online-data-sci-thd.tianlong.area_rugs_collections_apr_2021` area_rugs 
join `hd-personalization-prod.personalization.common_ItemAttributes`  ia on safe_cast(area_rugs.OMSID as string) = ia.omsid 
-- join `hd-personalization-prod.personalization.common_ItemAttributes` ia1 on safe_cast(area_rugs.OMSID as string) = ia1.omsid 
-- join `hd-personalization-prod.personalization.common_ItemAttributes` ia2 on safe_cast(area_rugs.OMSID as string) = ia2.omsid 

where ia.attributeid = 'fac03a00-67f9-402b-a121-fb9448fc552d' --and 
-- ia1.attributeid = '5ad8d718-20da-4ce0-9872-fe7c84f24bb2' and 
-- ia2.attributeid = '542b2138-e766-4426-aa1a-250eadc74b50')
)
select dims.*, itc.image_url, split(itc.image_url, "/")[offset(4)] image_guid  from dims join `analytics-online-data-sci-thd.mart.itc_weekly_results` itc on dims.OMSID = itc.oms_id 
where itc.prediction_labels like '%front%'
"""
client=bigquery.Client(project='analytics-online-data-sci-thd')

myData = client.query(query).to_dataframe()
test = myData.loc[0]
img_guid = test.image_guid

def display_img(im):
    plt.imshow(im)
    plt.show()

img = get_image(img_guid)
dimension_map = {
        "height": 92,
        "width": 67,
        "depth": 2.0}
dimension_image, _ = create_dimension_image(img, config, dimension_map, font_local_filepath,
                                                False, 'HD-Home', False, True)
#display_img(dimension_image)

create_dir_if_needed(output_dir)
plt.imsave(output_dir + 'test' + '.jpg', dimension_image)
plt.close()