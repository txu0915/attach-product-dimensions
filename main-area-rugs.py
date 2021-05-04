import sys
#sys.path.clear()
# sys.path.extend(
# ['/Applications/PyCharm.app/Contents/plugins/python/helpers/pydev',
#  '/Applications/PyCharm.app/Contents/plugins/python/helpers/pycharm_display',
#  '/Applications/PyCharm.app/Contents/plugins/python/helpers/third_party/thriftpy',
#  '/Applications/PyCharm.app/Contents/plugins/python/helpers/pydev',
#  '/Users/tianlongxu/anaconda3/lib/python38.zip',
#  '/Users/tianlongxu/anaconda3/lib/python3.8',
#  '/Users/tianlongxu/anaconda3/lib/python3.8/lib-dynload',
#  '/Users/tianlongxu/anaconda3/lib/python3.8/site-packages',
#  '/Users/tianlongxu/anaconda3/lib/python3.8/site-packages/aeosa',
#  '/Applications/PyCharm.app/Contents/plugins/python/helpers/pycharm_matplotlib_backend',
#  '/Users/tianlongxu/anaconda3/lib/python3.8/site-packages/IPython/extensions',
#  '/Users/tianlongxu/Documents/My_Projects/Adding_Dimensions',
#  '/Users/tianlongxu/Documents/My_Projects/Adding_Dimensions/py_scripts',
#  '/Users/tianlongxu/Documents/My_Projects/Adding_Dimensions/'])

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
import re

working_dir = '/Users/tianlongxu/Documents/My_Projects/Adding_Dimensions/'
drawing_config_local_filepath = working_dir + 'configs/image_drawing_config.json'
font_local_filepath = working_dir + 'configs/HelveticaNeue-Medium.otf'
output_dir = working_dir + 'Results/apr-30-2021/'
input_dir = working_dir + 'input-data/'

with open(drawing_config_local_filepath, 'r') as fp:
    config = json.load(fp)

query = """
with
thickness as(
select omsid, ia.attributeid thickness_att_id, ia.attributevalue thickness_inches_Version_1 from 
`hd-personalization-prod.personalization.common_ItemAttributes`     ia  
where  ia.attributeid = 'abc38576-51f5-4140-80fc-dd9a2e5cbdcd'),

thickness2 as(
select omsid, ia.attributeid thickness_att_id, ia.attributevalue thickness_inches_Version_2 from 
`hd-personalization-prod.personalization.common_ItemAttributes`     ia  

where  ia.attributeid = 'de66c7b9-1989-410a-a78e-a8a69be4086b'),

df_with_thickness_info as (
select area_rugs.*, thickness.thickness_inches_Version_1, thickness2.thickness_inches_Version_2   from `analytics-online-data-sci-thd.tianlong.area_rugs_collections_apr_2021` area_rugs join thickness 
on safe_cast(area_rugs.OMSID as string) = thickness.omsid 
join thickness2 on safe_cast(area_rugs.OMSID as string) = thickness2.omsid )


select df_with_thickness_info.* , ia.attributevalue  image_guid from df_with_thickness_info join   `hd-personalization-prod.personalization.common_ItemAttributes` ia on safe_cast(df_with_thickness_info.OMSID as string) = ia.omsid 

where lower(ia.attributeid ) = '645296e8-a910-43c3-803c-b51d3f1d4a89'
"""
client=bigquery.Client(project='analytics-online-data-sci-thd')

myData = client.query(query).to_dataframe()
test = myData.loc[5]
print(test.OMSID)
img_guid = test.image_guid

def grab_width_height(string):
    s1, s2 = string.split('x')
    ft1 = re.findall("[0-9]+ ft", s1)
    ft2 = re.findall("[0-9]+ ft", s2)
    in1 = re.findall("[0-9]+ in", s1)
    in2 = re.findall("[0-9]+ in", s2)
    inches = float(in1[0][:-3]) if in1 else 0
    ft = float(ft1[0][:-3])*12 if ft1 else 0
    s1 = ft + inches
    inches = float(in2[0][:-3]) if in2 else 0
    ft = float(ft2[0][:-3]) * 12 if ft2 else 0
    s2 = ft + inches

    nums = sorted([s1,s2])
    return nums

def display_img(im):
    plt.imshow(im)
    plt.show()

for i, row in myData.iterrows():
    omsid = row.OMSID
    img = get_image(row.image_guid)
    width, height = grab_width_height(row.Product_Name__120_)
    thickness = float(row.thickness_inches_Version_2)
    dimension_map = {
            "height": height,
            "width": width,
            "depth": thickness}
    dimension_image, _ = create_dimension_image(img, config, dimension_map, font_local_filepath,
                                                    False, 'HD-Home', False, True)
    create_dir_if_needed(output_dir)
    plt.imsave(output_dir + str(omsid) + '.jpg', dimension_image)
    plt.close()

    if i== 10:
        break