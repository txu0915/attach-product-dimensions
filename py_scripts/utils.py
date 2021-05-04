try:
    from model_mart_file_utils import *
except ImportError as e:
    print("In utils.py: Skipping model_mart_file_utils import: %s" % str(e))

from consts import *

from pyspark.sql.types import *
import pyspark.sql.functions as F

import os
import math
import re

from sympy.geometry import *

import skimage.io as skio
import cv2
from skimage.color import rgb2gray
from skimage import filters
from scipy import ndimage as ndi
import numpy as np
from skimage import morphology
import time
from PIL import Image
from PIL import ImageDraw

def create_dir_if_needed(filepath):
    dir, _ = os.path.split(filepath)
    if not os.path.isdir(dir):
        os.makedirs(dir)

def get_merged_img_df(output_dir, day, spark):
    """
    Load the input data, giving the product type, the oms_id, the image guid to superimpose dimensions on,
    and whether the image is a front or side facing image.

    :param output_dir: The directory in GCS where the output is stored
    :param day: The day of data that is being processed
    :param spark: SparkSession to load data
    :return: df with columns (product_type, oms_id, image_guid, image_type)
    """
    schema = [
        StructField('product_type', StringType()),
        StructField('oms_id', StringType()),
        StructField('image_guid', StringType()),
        StructField('image_type', StringType())
    ]

    return spark.read.csv("%s/%s/%s" % (output_dir, format_day(day, '-'), MERGED_IMAGES_OUTPUT_SUFFIX),
                          schema=StructType(schema))


def extract_dimension_from_text(t, num_decimal_pts):
    """
    Given a string representing the raw textual form of the dimension value,
    parse out the numerical value and round to the specified number of decimal points.

    :param t: the string form of the dimension value, could contain units like "inches"
    :param num_decimal_pts: How many decimal points to include in the output.
    :return: The rounded numerical form of the dimension value
    """
    t = str(t)
    m = re.search(r"\d+\.?(\d+)?|\.\d+", t)

    if m is None:
        return None

    val = float(m.group())

    for _ in range(num_decimal_pts):
        val *= 10

    val = math.ceil(val)

    for _ in range(num_decimal_pts):
        val /= 10

    return val


def get_dimension_df(output_dir, day, spark, num_decimal_pts):
    """
    Load the dimension values for the SKUs.  This data is created using the get_dimensions.py script.

    :param output_dir: The directory in GCS where the output was stored.
    :param day: The day of data that is being processed
    :param spark: SparkSession to load data
    :param num_decimal_pts: How many decimal points each dimension value should be rounded to.
    :return: df with columns (oms_id, width, depth, height)
    """
    schema = [
        StructField('oms_id', StringType()),
        StructField('width', StringType()),
        StructField('height', StringType()),
        StructField('depth', StringType())
    ]

    dim_parse_udf = F.udf(lambda t: extract_dimension_from_text(t, num_decimal_pts), FloatType())

    return spark.read.csv("%s/%s/%s" % (output_dir, format_day(day, '-'), DIMENSION_OUTPUT_SUFFIX),
                          schema=StructType(schema)) \
        .select('oms_id', dim_parse_udf('width').alias('width'), dim_parse_udf('height').alias('height'),
                dim_parse_udf('depth').alias('depth'))


def get_one_image_type_with_dims(merged_img_df, dimension_df, img_type):
    """
    Combine the image data and the dimension data into a single row.

    :param merged_img_df: the image data giving which image to use for a particular oms_id
    :param dimension_df: the dimension data for each oms_id
    :param img_type: A filter for the image types to generate.  If none is specified, will generate both front and side.
    :return: df with columns (product_type, oms_id, image_guid, image_type, width, height, depth)
    """
    if img_type is not None:
        merged_img_df = merged_img_df.where(F.col('image_type') == img_type)

    return merged_img_df.join(dimension_df, how='inner', on='oms_id')


def add_border(img, border_ratio, border_color=(255, 255, 255)):
    """
    Return an image formed by adding a border of the specified size and color to the provided image.

    :param img: The original raw image
    :param border_ratio: the border will be this proportion of the input image's width/height
    :param border_color: Which RGB color to fill the border with, defaults to pure white
    :return: The image with border, as a numpy array
    """
    border_sz = int(max(img.shape[:2]) * border_ratio)
    return cv2.copyMakeBorder(img, border_sz, border_sz, border_sz, border_sz, cv2.BORDER_CONSTANT, value=border_color)


def dim_to_text(dim, num_decimal_pts):
    """
    Convert from a numerical dimension value to a textual representation
    :param dim: The numerical dimension value
    :param num_decimal_pts: The maximum number of decimal points to include.
    :return: The string representation of the dimension value, with " to indicate inches.
    """
    s = ("%%.%df" % num_decimal_pts) % dim
    while "." in s and s.endswith((".", "0")):
        s = s[:-1]

    return "%s\"" % s

## modified version for area rug
def dim_to_text(dim, num_decimal_pts, display_both_ft_inches=False):
    """
    Convert from a numerical dimension value to a textual representation
    :param dim: The numerical dimension value
    :param num_decimal_pts: The maximum number of decimal points to include.
    :return: The string representation of the dimension value, with " to indicate inches.
    :display_both_ft_inches: display in the format of 12ft 7inches...
    """
    if display_both_ft_inches:
        inches = int(dim)%12
        ft = int(dim) // 12
        # s = str(ft) + "'" + str(inches)
        return "%s" % ft + "'"+ "%s\"" % inches
    s = ("%%.%df" % num_decimal_pts) % dim

    while "." in s and s.endswith((".", "0")):
        s = s[:-1]
    return "%s\"" % s


def add_extra_border(img, border_top, border_left, xs, ys, min_edge_dist=25, border_color=(255, 255, 255)):
    """
    Check to make sure no element about to be superimposed will be off screen or too close to the border.
    If so, add extra border to stop this from happening.

    :param img: The image that is being drawn on.
    :param border_top: Indicates how much extra border has already been added to the top of the image.
    :param border_left: Indicates how much extra border has already been added to the left of the image.
    :param xs: The various x values for the line and dimension being added currently
    :param ys: The various y values for the line and dimension being added currently
    :param min_edge_dist: Superimposed elements should be at most this many pixels from the edge, defaults to 25
    :param border_color: what color to fill the extra border, defaults to pure white
    :return: The image with any extra border added, as a numpy array, and the updated extra border counts for the top and left sides.
    """
    x_min = float(min(xs) + border_left)
    x_max = float(max(xs) + border_left)
    y_min = float(min(ys) + border_top)
    y_max = float(max(ys) + border_top)

    if x_min < min_edge_dist:
        extra_border = int(math.ceil(min_edge_dist - x_min))
        img = cv2.copyMakeBorder(img, 0, 0, extra_border, 0, cv2.BORDER_CONSTANT, value=border_color)
        x_max += extra_border
        border_left += extra_border

    if x_max + min_edge_dist > img.shape[1] - 1:
        extra_border = int(math.ceil(x_max + min_edge_dist - img.shape[1] + 1))
        img = cv2.copyMakeBorder(img, 0, 0, 0, extra_border, cv2.BORDER_CONSTANT, value=border_color)

    if y_min < min_edge_dist:
        extra_border = int(math.ceil(min_edge_dist - y_min))
        img = cv2.copyMakeBorder(img, extra_border, 0, 0, 0, cv2.BORDER_CONSTANT, value=border_color)
        y_max += extra_border
        border_top += extra_border

    if y_max + min_edge_dist > img.shape[0] - 1:
        extra_border = int(math.ceil(y_max + min_edge_dist - img.shape[0] + 1))
        img = cv2.copyMakeBorder(img, 0, extra_border, 0, 0, cv2.BORDER_CONSTANT, value=border_color)

    return img, border_top, border_left


def add_label(img, p1, p2, config, dim_name, away_pt, label, font, margin, line_width, border_top, border_left, area_rugs_specific_requirements):
    """
    Draws a dimension line and value on the image.
    Will add extra border to the image if the shifted dimension line or value will be off the screen.

    :param img: The image that the line and value should be drawn on, as a numpy array
    :param p1: The first point defining the edge of the product.
    :param p2: The second point defining the edge of the product.
    :param away_pt: The line will be moved away from this point so that it doesn't lie exactly along the product's border
    :param label: The text value to write in the center of the line
    :param font: The font object used to control how the label is written
    :param margin: The number of pixels that the dimension line should be moved away from the away_pt
    :param line_width: The width of the dimension line
    :param border_top: Indicates how much extra border has already been added to the top of the image.
    :param border_left: Indicates how much extra border has already been added to the left of the image.
    :return: The image with the line and value drawn, as a numpy array, followed by the updated border sizes on the top and left.
    """
    if p1 == p2:
        raise ValueError("trying to add label, but two points are the same.  Label text: %s" % label)

    # Figure out how big the box will be around the dimension value
    text_size = [1.2 * x for x in ImageDraw.Draw(Image.fromarray(img)).multiline_textsize(label, font=font)]
    min_margin = 1.2 * math.sqrt(sum(x ** 2 for x in text_size)) / 2.
    margin = max(min_margin, margin)

    edge = Segment(p1, p2)
    m = edge.midpoint
    away_on_bisector = \
        intersection(edge.perpendicular_bisector().perpendicular_line(away_pt), edge.perpendicular_bisector())[0]

    # figure out how much to move in the x and y directions to move perpendicular to the segment.
    dx = p1.y - p2.y
    dy = p2.x - p1.x
    length = math.sqrt(dx ** 2 + dy ** 2)
    dx /= length
    dy /= length

    # make sure we are moving away from the away_pt, not towards it
    if dx * (away_on_bisector.x - m.x) > 0 or dy * (away_on_bisector.y - m.y) > 0:
        dx *= -1
        dy *= -1

    # displace the dimension line end points the right amount
    p1_disp = displace(p1, margin, dx, dy)
    p2_disp = displace(p2, margin, dx, dy)

    edge_disp = Segment(p1_disp, p2_disp)
    m_disp = edge_disp.midpoint

    bx1 = m_disp.x - text_size[0] / 2
    bx2 = bx1 + text_size[0]
    by1 = m_disp.y - text_size[1] / 2
    by2 = by1 + text_size[1]

    img, border_top, border_left = add_extra_border(img, border_top, border_left, [p1_disp.x, p2_disp.x, bx1, bx2],
                                                    [p1_disp.y, p2_disp.y, by1, by2])

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    ## draw dimension bar ends -- area rug specific requirements
    ## reuse the computed line end positions...
    bar_x_end_1 = p1_disp.x + border_left
    bar_y_end_1 = p1_disp.y + border_top
    bar_x_end_2 = p2_disp.x + border_left
    bar_y_end_2 = p2_disp.y + border_top

    delta_x = abs(bar_x_end_1 - bar_x_end_2)
    delta_y = abs(bar_y_end_1 - bar_y_end_2)
    mid_x = 0.5 * (bar_x_end_1 + bar_x_end_2)
    mid_y = 0.5 * (bar_y_end_1 + bar_y_end_2)

    if area_rugs_specific_requirements:
        #if dim_name == 'height':
        ## determine whether the bar is horizontal (depth & width) or vertical (height)
        if delta_x/mid_x < delta_y/mid_y: ## this is vertical case
            endpoint_horiz_delta = 0.5*config['dims_bar_ends_size_fraction']*delta_y
            draw.line([bar_x_end_1 - endpoint_horiz_delta, bar_y_end_1, bar_x_end_1 + endpoint_horiz_delta, bar_y_end_1],
                fill=(0, 0, 0), width=line_width)
            draw.line([bar_x_end_2 - endpoint_horiz_delta, bar_y_end_2, bar_x_end_2 + endpoint_horiz_delta, bar_y_end_2],
                fill=(0, 0, 0), width=line_width)
        else: ## this is horizontal case
            endpoint_verti_delta = 0.5 * config['dims_bar_ends_size_fraction']*delta_x
            if dim_name == 'depth':
                endpoint_verti_delta *= 2.0
            draw.line([bar_x_end_1, bar_y_end_1 - endpoint_verti_delta, bar_x_end_1, bar_y_end_1 + endpoint_verti_delta],
                fill=(0, 0, 0), width=line_width)
            draw.line([bar_x_end_2, bar_y_end_2 - endpoint_verti_delta, bar_x_end_2, bar_y_end_2 + endpoint_verti_delta],
                fill=(0, 0, 0), width=line_width)

        draw.line([bar_x_end_1, bar_y_end_1, bar_x_end_2, bar_y_end_2],
                  fill=(0, 0, 0), width=line_width)
    else:
        draw.line([bar_x_end_1, bar_y_end_1, bar_x_end_2, bar_y_end_2],
                  fill=(0, 0, 0), width=line_width)

    # draw a white rectangle where the text will be located.
    if dim_name == 'depth':
        bar_left_x = min(bar_x_end_1,bar_x_end_2)
        bar_left_y = 0.5*(bar_y_end_1+bar_y_end_2)
        depth_text_pos_x = 0.4*(bar_left_x + 0)
        depth_text_pos_y = bar_left_y + config['dims_bar_ends_size_fraction']*delta_x
        draw_text_at_pt(draw, label,(depth_text_pos_x, depth_text_pos_y), font, x_align="left")
    else:
        draw.rectangle([bx1 + border_left, by1 + border_top, bx2 + border_left, by2 + border_top], fill=(255, 255, 255))
        draw_text_at_pt(draw, label, (m_disp.x + border_left, m_disp.y + border_top), font)
    return np.array(pil_img), border_top, border_left


def displace(p, amount, dx, dy):
    """
    Move the provided point a certain distance in a certain direction

    :param p: The starting point
    :param amount: the distance to move the point
    :param dx: how big a step in the x direction to take (the length of the dx,dy vector should be 1)
    :param dy: how big a step in the y direction to take (the length of the dx,dy vector should be 1)
    :return: The shifted point.
    """
    return Point(p.x + amount * dx, p.y + amount * dy)


def draw_text_at_pt(draw, text, pt, font, color=(0, 0, 0), x_align="center", y_align="center"):
    """
    Draw the text onto the image.

    :param draw: The object used to draw on the image
    :param text: The text to write
    :param pt: The point where the text should be written
    :param font: The font object with which the text should be written
    :param color: The color with which the text should be written, defaults to pure black.
    :param x_align: One of (center, right, left) controlling how the text is written at the point's x value
    :param y_align: One of (center, bottom, top) controlling how the text is written at the point's y value
    """
    text_size = draw.multiline_textsize(text, font=font)

    x = pt[0]
    y = pt[1]
    if x_align == "center":
        x -= text_size[0] / 2.0
    elif x_align == "right":
        x -= text_size[0]
    elif x_align != "left":
        raise ValueError('x_align should be left, right, or center')

    if y_align == "center":
        y -= text_size[1] / 2.0
    elif y_align == "bottom":
        y -= text_size[1]
    elif y_align != "top":
        raise ValueError('y_align should be top, bottom, or center')

    draw.multiline_text((x, y), text, fill=color, font=font, align=x_align)
