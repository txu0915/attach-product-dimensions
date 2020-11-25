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

with open("./configs/image_drawing_config.json", 'r') as fp:
    config = json.load(fp)

client=bigquery.Client(project='analytics-online-data-sci-thd')
query = """
with omsids_in_lighting_collection_oct_15 as 
(select 
  distinct(omsid) 
 from 
  (select ia.omsid omsid, t.Taxonomy taxonomy from `hd-datascience-np.common.ItemAttributes` ia inner join `hd-datascience-np.common.CATGPENREL_taxonomy` t on
ia.omsid = t.omsid 
where attributevalue in (
'687fdc23-5718-414a-a9cc-7afe5afa1137',
' 70950349-2d3a-41b8-9090-b84f359127d8',
' 9ade88ed-0cd3-4e84-99ed-5d1eaf132c78',
' d35b692e-05cf-4ee3-a89d-33c7c86df972',
'08aade79-e76a-4283-aafc-a619c1a0502f',
' 0d98f1b6-7592-4a94-bf56-78d663e11949',
' cd2e3c89-0dd2-4651-af6d-4b104fa0bf27',
' d95d9f91-c6cf-4097-a55b-c0290c3ac0a7',
'3f42bc63-311e-4090-b101-6137e882b39d',
' 525a1463-296d-4cdd-92c6-140e7243ab50',
' 1ff451f4-f346-488d-b2d7-2b2c40586b06',
' 4f197972-f0cf-4bee-892d-a6d0eea3a94b')
)),
 omsids_dims as (
select 
  a.omsid, a.attributevalue skuid, dims.*except(omsid) 
from 
  `hd-datascience-np.common.ItemAttributes` a
  inner join `hd-datascience-np.common.Attributes` b on a.attributeid =b.attribute_id
  inner join 
    (select omsids_in_lighting_collection_oct_15.omsid,ia.attributevalue as width,ia1.attributevalue as depth,ia2.attributevalue as height, ia3.attributevalue as length
      from omsids_in_lighting_collection_oct_15
        inner join `hd-datascience-np.common.ItemAttributes` ia on omsids_in_lighting_collection_oct_15.omsid = ia.omsid
        inner join `hd-datascience-np.common.ItemAttributes` ia1 on omsids_in_lighting_collection_oct_15.omsid = ia1.omsid
        inner join `hd-datascience-np.common.ItemAttributes` ia2 on omsids_in_lighting_collection_oct_15.omsid = ia2.omsid
        inner join `hd-datascience-np.common.ItemAttributes` ia3 on omsids_in_lighting_collection_oct_15.omsid = ia3.omsid
      where 
        ia.attributeid = 'fac03a00-67f9-402b-a121-fb9448fc552d' and
        ia1.attributeid = '5ad8d718-20da-4ce0-9872-fe7c84f24bb2' and
        ia2.attributeid = '542b2138-e766-4426-aa1a-250eadc74b50' and
        ia3.attributeid = '0c8793c6-8fc8-4dd8-86db-04ca3f0204a0') dims
    on a.omsid = dims.omsid 
where 
  lower(b.displayname) ='product image'
  and a.omsid in (select * from omsids_in_lighting_collection_oct_15 ) )
  
 select * from omsids_dims 
"""


myData = client.query(query).to_dataframe()
myData.head(100)


for i, row in hd_prod_imgs_w_dims.iterrows():
    img = get_image(row.attributevalue)
    dimension_map = {
        "height": float(str(row.height).split()[0]),
        "width": float(str(row.width).split()[0]),
        "depth": float(str(row.depth).split()[0])}
    dimension_image, _ = create_dimension_image(img, config, dimension_map, font_local_filepath,
                                                False, 'HD-Home', False)

    # plt.imshow(dimension_image)
    # fig, ax = plt.subplots(figsize=(15,15))
    plt.imsave('./' + str(row.attributevalue) + '.jpg', dimension_image)
    plt.close()
    #     if not (i+1)%10:
    #         print(str(i+1)+" done.")
    if i + 1 == 2:
        break


font_local_filepath = "./configs/HelveticaNeue-Medium.otf"
url = 'https://images.homedepot-static.com/productImages/194a2f4c-26d4-43ac-ab9b-68c563504971/svn/brushed-nickel-minka-lavery-flush-mount-lights-4107-84-64_1000.jpg'
img_guid = '194a2f4c-26d4-43ac-ab9b-68c563504971'
img = get_image(img_guid)
dimension_map = {
    "height": 50,
    "width": 60,
    "depth": 60}
dimension_image, _ = create_dimension_image(img, config, dimension_map, font_local_filepath,False, 'HD-Home', False)



image = cv2.imread('./Results/194a2f4c-26d4-43ac-ab9b-68c563504971.jpg')
image = img
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY_INV)[1]

#im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find contours
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
c = max(cnts, key=cv2.contourArea)

# Obtain outer coordinates
left = tuple(c[c[:, :, 0].argmin()][0])
right = tuple(c[c[:, :, 0].argmax()][0])
top = tuple(c[c[:, :, 1].argmin()][0])
bottom = tuple(c[c[:, :, 1].argmax()][0])
canopy_center = [top[0],int(2/3*top[1])]

hull = cv2.convexHull(c,False)
hull = cv2.convexHull()
cv2.drawContours(image,[c],0,color=(0,0,255))
cv2.drawContours(image,[hull],0,color=(0,0,255))

for i in hull[:,]:
    if i[0,1] <= top[1]:
        print(i[0,0],i[0,1])

# hull = []
# for i in range(len(cnts)):
#     hull.append(cv2.convexHull(cnts[i], False))
#
#
# for i in range(len(cnts)):
#     color_contours = (0, 255, 0) # green - color for contours
#     color = (255, 0, 0) # blue - color for convex hull
#     #cv2.drawContours(image, c, i, color_contours, 1, 8, hierarchy)
#     cv2.drawContours(image, hull, i, color, 1, 8)
from scipy.spatial import distance
a = (1, 2, 3)
b = (4, 5, 6)
dst = distance.euclidean(a, b)


canopy_dims_text = "20''"
img_guid = '194a2f4c-26d4-43ac-ab9b-68c563504971'
img_guid = '1b3cad43-67b5-4cc0-83e0-4e7bcc8304ec'
image = get_image(img_guid)
plt.imshow(image)
plt.show()


# Find contours
image = get_image('ccf15fab-8c48-4cdb-97f1-3b57933eaf39')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY_INV)[1]
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
c = max(cnts, key=cv2.contourArea)

# Obtain outer coordinates
left = tuple(c[c[:, :, 0].argmin()][0])
right = tuple(c[c[:, :, 0].argmax()][0])
top = tuple(c[c[:, :, 1].argmin()][0])
bottom = tuple(c[c[:, :, 1].argmax()][0])
canopy_center = [top[0],int(2/3*top[1])]

hull = cv2.convexHull(c,False)
#cv2.drawContours(image,[hull],0,color=(0,0,255))
## drawing a bigger contour
for h in hull:
    cv2.circle(image,(h[0,0],h[0,1]),8,color=(0,84,54),lineType=1)
plt.imshow(image)
plt.show()
plt.imsave('./Results/chandelier.jpg',image)


dist_thresh, points_to_examine = 100,25
prev = np.array(top)
for h in hull[:points_to_examine]:
    next = h.squeeze()
    print(distance.euclidean(prev,next),"|",prev,"|",next)
    if distance.euclidean(prev,next) >= dist_thresh:
        canopy_right = prev
        break
    cv2.circle(image,(h[0,0],h[0,1]),8,(255, 50, 0),-1)
    plt.imshow(image)
    plt.show()
    prev = next
prev = np.array(top)
for h in reversed(hull[-points_to_examine:]):
    next = h.squeeze()
    #print(distance.euclidean(prev, next))
    if distance.euclidean(prev,next) >= dist_thresh:
        canopy_left = prev
        break
    cv2.circle(image, (h[0, 0], h[0, 1]), 8, (255, 50, 0), -1)
    plt.imshow(image)
    plt.show()
    prev = next
print(canopy_left,canopy_right)
cv2.circle(image, (canopy_left[0], canopy_left[1]), 12, (255, 50, 0), -1)
cv2.circle(image, (canopy_right[0], canopy_right[1]), 12, (255, 50, 0), -1)
plt.imshow(image)
plt.show()



plt.imshow(image)
plt.show()

dimension_image, _ = create_dimension_image(image, config, dimension_map, font_local_filepath,False, 'HD-Home', False)
plt.imshow(dimension_image)
plt.show()




plt.imsave('./Results/' + img_guid + '.jpg', dimension_image)
plt.close()

cv2.drawContours(image, [hull], 0, (255,0,255), 1, 8)