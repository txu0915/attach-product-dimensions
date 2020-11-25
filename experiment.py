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
from scipy.spatial import distance

with open("./configs/image_drawing_config.json", 'r') as fp:
    config = json.load(fp)

def display_image(image):
    plt.imshow(image)
    plt.show()
    plt.close()
    return




#image = cv2.imread('./Results/194a2f4c-26d4-43ac-ab9b-68c563504971.jpg')
image = get_image('426d8554-b916-4f3c-b811-ee5b5bee0502')
display_image(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY_INV)[1]


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
cv2.drawContours(image,[c],0,color=(0,0,255))
cv2.drawContours(image,[hull],0,color=(0,0,255))

for i in hull[:,]:
    if i[0,1] <= top[1]:
        print(i[0,0],i[0,1])
display_image(image)
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
a = (1, 2, 3)
b = (4, 5, 6)
dst = distance.euclidean(a, b)


canopy_dims_text = "20''"
img_guid = '194a2f4c-26d4-43ac-ab9b-68c563504971'
img_guid = '1b3cad43-67b5-4cc0-83e0-4e7bcc8304ec'

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
display_image(image)


dist_thresh, points_to_examine = 200,20
prev = np.array(top)
for h in hull[:points_to_examine]:
    next = h.squeeze()
    print(distance.euclidean(prev,next),"|",prev,"|",next)
    if distance.euclidean(prev,next) >= dist_thresh:
        canopy_right = prev
        break
    cv2.circle(image,(h[0,0],h[0,1]),8,(255, 50, 0),-1)
    display_image(image)
    prev = next
prev = np.array(top)
for h in reversed(hull[-points_to_examine:]):
    next = h.squeeze()
    #print(distance.euclidean(prev, next))
    if distance.euclidean(prev,next) >= dist_thresh:
        canopy_left = prev
        break
    cv2.circle(image, (h[0, 0], h[0, 1]), 8, (255, 50, 0), -1)
    display_image(image)
    prev = next
print(canopy_left,canopy_right)

cv2.circle(image,top,12,(0, 0, 0), -1)
cv2.circle(image, (canopy_left[0], canopy_left[1]), 12, (255, 50, 0), -1)
cv2.circle(image, (canopy_right[0], canopy_right[1]), 12, (255, 50, 0), -1)
display_image(image)
canopy_width = abs(canopy_right[0] - canopy_left[0])
canopy_up_corner = (int(top[0] - canopy_width),top[1])  ##shift the top critical point to the left by about the distance of the canopy width
canopy_bottom_corner = (int(canopy_up_corner[0]+0.5*canopy_width/np.sqrt(2.0)), int(canopy_up_corner[1]+0.5*canopy_width/np.sqrt(2.0)))
cv2.circle(image, (canopy_up_corner[0], canopy_up_corner[1]), 12, (0, 0, 255), -1)
cv2.circle(image, (canopy_bottom_corner[0], canopy_bottom_corner[1]), 12, (0, 0, 255), -1)
display_image(image)
## draw the canopy dimension...

from py_scripts.utils_canopy import *
drawing_config_local_filepath = '/Users/tianlongxu/Documents/My_Projects/Adding_Dimensions/configs/image_drawing_config.json'
font_local_filepath = '/Users/tianlongxu/Documents/My_Projects/Adding_Dimensions/configs/HelveticaNeue-Medium.otf'

img_guid = '426d8554-b916-4f3c-b811-ee5b5bee0502' ## Nov 18
img_guid = '109bb97f-3c62-468a-8f67-38d64be1db13'
image = get_image(img_guid)
#display_image(image)


dimension_map = {"height": 50,"width": 60,"depth": 60,"canopy-width":2,"canopy-depth":1}
image = add_canopy_dims_bar(image, dimension_map, config, font_local_filepath)
display_image(image)



plt.imsave('./Results/' + img_guid + '.jpg', dimension_image)
plt.close()

cv2.drawContours(image, [hull], 0, (255,0,255), 1, 8)