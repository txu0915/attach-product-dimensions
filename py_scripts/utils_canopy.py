import cv2
import numpy as np
from scipy.spatial import distance
import json
from py_scripts.utils import *
from py_scripts.model_mart_img_utils import *
from py_scripts.create_imgs import *
import matplotlib.pyplot as plt

def whole_product_in_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY_INV)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    # Obtain outer coordinates
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    if top[1] <= 5:
        return False
    return True

def compute_product_convex_hull(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY_INV)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    # Obtain outer coordinates
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    hull = cv2.convexHull(c, False)
    top = hull[0].squeeze()
    return hull,top, left, right

dist_thresh, points_to_examine = 100,25

def compute_canopy_extreme_pts(hull, top, left, right, dist_thresh=200, points_to_examine = 30):
    hull_horiz_size = right[0] - left[0]
    canopy_est_horiz_size = hull_horiz_size*0.3
    canopy_left, canopy_right = np.array([int(left[0]*5.0/8.0+right[0]*3.0/8.0),int(0.67*top[1])]),np.array([int(left[0]*3.0/8.0+right[0]*5.0/8.0),int(0.67*top[1])])
    #print(canopy_est_horiz_size, canopy_left, canopy_right)
    prev = np.array(top)
    for h in hull[:points_to_examine]:
        next = h.squeeze()
        #print(distance.euclidean(prev,next),"|",prev,"|",next)
        if distance.euclidean(prev,next) >= dist_thresh:
            canopy_right = prev
            break
        # cv2.circle(image,(h[0,0],h[0,1]),8,(255, 50, 0),-1)
        # plt.imshow(image)
        # plt.show()
        prev = next
    if prev[0] == top[0] and prev[1]==top[1]:
        canopy_left = np.array([int(left[0]*5.0/8.0+right[0]*3.0/8.0),int(0.67*top[1])])
        canopy_right = np.array([int(left[0]*3.0/8.0+right[0]*5.0/8.0), int(0.67*top[1])])
        print("Using estimated canopy bound...")
    prev = np.array(top)
    for h in reversed(hull[-points_to_examine:]):
        next = h.squeeze()
        #print(distance.euclidean(prev,next),"|",prev,"|",next)
        if distance.euclidean(prev,next) >= dist_thresh:
            canopy_left = prev
            break
        # cv2.circle(image, (h[0, 0], h[0, 1]), 8, (255, 50, 0), -1)
        # plt.imshow(image)
        # plt.show()
        prev = next
    if prev[0] == top[0] and prev[1]==top[1]:
        canopy_left = np.array([int(left[0] * 5.0/8.0+right[0]*3.0/8.0), int(0.67*top[1])])
        canopy_right = np.array([int(left[0]*3.0/8.0+right[0]*5.0/8.0),int(0.67*top[1])])
        print("Using estimated canopy bound...")
    return canopy_left,canopy_right

def shift_canopy_dims_loc(canopy_left, canopy_right, top, offset=1.0):
    canopy_dims_left = Point(canopy_left[0], int(canopy_left[1] * offset))
    canopy_dims_right = Point(canopy_right[0], int(canopy_right[1] * offset))
    if (canopy_left[1] * offset+canopy_right[1] * offset)*0.5 >= float(top[1]):
        canopy_dims_left = Point(canopy_left[0], int(top[1] * offset))
        canopy_dims_right = Point(canopy_right[0], int(top[1] * offset))
    return canopy_dims_left,canopy_dims_right

def create_dir_if_needed(filepath):
    dir, _ = os.path.split(filepath)
    if not os.path.isdir(dir):
        os.makedirs(dir)


#image = get_image(img_guid)
def add_canopy_dims_bar(image, dimension_map, config,font_local_filepath):
    line_margin, line_sz = get_line_specs(image, config)
    font = get_font(font_local_filepath, image, config)
    hull, top, left, right = compute_product_convex_hull(image)
    canopy_left, canopy_right = compute_canopy_extreme_pts(hull, top, left, right)
    canopy_dims_left, canopy_dims_right = shift_canopy_dims_loc(canopy_left, canopy_right,top)
    canopy_width_text = dim_to_text(dimension_map['canopy'],config["num_dim_decimal_pts"])
    image, border_top, border_left = add_label(image, canopy_dims_left, canopy_dims_right, Point(500, 500),
                                               canopy_width_text,font, line_margin, int(line_sz * 1.5), 0, 0)
    return image

def save_image_with_three_major_dims_bars(image,output_img_name,output_dir,config, dimension_map, font_local_filepath):
    dimension_image, _ = create_dimension_image(image, config, dimension_map, font_local_filepath,
                                                False, 'HD-Home',False)
    create_dir_if_needed(output_dir)
    plt.imsave(output_dir + output_img_name + '.jpg', dimension_image)
    plt.close()

# image = get_image('ccf15fab-8c48-4cdb-97f1-3b57933eaf39')
# plt.imshow(image)
# plt.show()