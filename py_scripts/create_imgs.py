# try:
#     from model_mart_logging import *
# except ImportError as e:
#     print("In get_dimensions.py: Skipping model_mart_logging import: %s" % str(e))
#
# try:
#     from model_mart_file_utils import *
# except ImportError as e:
#     print("In get_dimensions.py: Skipping model_mart_file_utils import: %s" % str(e))

from model_mart_img_utils import *

from utils import *
from consts import *

from pyspark.sql import SparkSession

import traceback
import datetime
import math
import numpy as np
import argparse
import json
import os
import skimage.io as skio
from colorsys import hsv_to_rgb

from sympy.geometry import *

from PIL import ImageFont


class BorderEdge:
    """
    Defines an edge of the product using a set of segments which all lie on that same edge.

    Stores:
        self.seg_list: all the segments associated with this edge
        self.endpts: The two points at the extremes of the edge. Stored as EdgeEndPoint
    """
    def __init__(self, seg_list):
        self.seg_list = seg_list
        self.endpts = get_endpts(seg_list, self)

    def draw_all_segs(self, img, color, thickness=5):
        """
        Draws all segments defining this edge.

        :param img: the image to draw on
        :param color: the color for the segments
        :param thickness: the line thickness
        """
        for seg in self.seg_list:
            x1 = seg.points[0].x
            y1 = seg.points[0].y
            x2 = seg.points[1].x
            y2 = seg.points[1].y
            cv2.line(img, (x1, y1), (x2, y2), color, thickness=thickness)

    def length(self):
        """
        :return: The distance between the two endpoints of the edge.
        """
        return self.endpts[0].endpt.distance(self.endpts[1].endpt)

    def get_angle(self):
        """
        :return: The angle formed by the two endpoints of the edge.
        """
        e1 = self.endpts[0].endpt
        e2 = self.endpts[1].endpt

        return math.atan2(e2.y - e1.y, e2.x - e1.x)

    def draw_endsegs(self, img, color, thickness=5):
        """
        Draws the two end points

        :param img: The image to draw on
        :param color: The color for the segments
        :param thickness: the line thickness
        """
        for endpt in self.endpts:
            endpt.draw(img, color, thickness)

    # def get_closest_endpt_matches(self, other_border_edges):
    #     closest_endpt_matches = []
    #     for my_endpt in self.endpts:
    #         other_segs = []
    #         for other_border in other_border_edges:
    #             other_segs += [(my_endpt, other_endpt, other_endpt.seg.distance(my_endpt.endpt)) for other_endpt in
    #                            other_border.endpts]
    #
    #         closest_endpt_matches += sorted(other_segs, key=lambda x: x[2])[:2]
    #
    #     return closest_endpt_matches

    def get_other_endpt(self, e):
        """
        Given one endpoint of this BorderEdge, returns the other

        :param e: one endpoint
        :return: the other endpoint
        """
        if e not in self.endpts:
            raise ValueError("The provided endpoint wasn't one of the two for this edge")

        return self.endpts[0] if self.endpts[0] != e else self.endpts[1]


class EdgeEndPoint:
    """
    The endpoint of one BorderEdge

    Stores:
        self.endpt: the Point object of the most extreme point
        self.seg: The segment object which the endpt is one end of
        self.edge: The BorderEdge object which this is an endpoint of
        self.corners: A list of corners associated with intersections of this endpoint with others.
    """
    def __init__(self, endpt, seg, edge):
        self.endpt = endpt
        self.seg = seg
        self.edge = edge
        self.corners = []

    def add_corner(self, corner):
        """
        Add a new corner to the list of corners associated with this endpoint.

        :param corner: The corner to be added to the list.
        """
        self.corners.append(corner)

    def draw(self, img, color, thickness=5):
        """
        Draw the EndPoint

        :param img: the image to be drawn on
        :param color: The color to use when drawing
        :param thickness: The line thickness
        """
        x1 = self.seg.points[0].x
        y1 = self.seg.points[0].y
        x2 = self.seg.points[1].x
        y2 = self.seg.points[1].y
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=int(thickness / 2))

        x1 = self.endpt.x
        y1 = self.endpt.y
        x2 = self.endpt.midpoint(self.seg.midpoint).x
        y2 = self.endpt.midpoint(self.seg.midpoint).y
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=thickness)


def is_in_img(pt, img_size, border_factor=0.05):
    """
    Determines if the given point is outside of the image by too much

    :param pt: the point in question
    :param img_size: The size of the image
    :param border_factor: a factor controlling how much the pt is allowed to be outside the image.
    :return: True if the point is within the image.
    """
    border_x = img_size[1] * border_factor
    border_y = img_size[0] * border_factor

    return -border_x <= pt.x < img_size[1] + border_x and -border_y <= pt.y < img_size[0] + border_y


class BorderCorner:
    """
    Represents the corner formed by two BorderEdges intersecting.

    Stores:
        self.endpoints: The two BorderEdges defining this corner
        self.intersection_pt: The coordinates of the intersection point.
    """

    def __init__(self, e1, e2, img_size):
        e1.add_corner(self)
        e2.add_corner(self)

        self.endpoints = (e1, e2)

        i = intersection(Line(e1.seg), Line(e2.seg))
        if len(i) > 0 and is_in_img(i[0], img_size):
            # if the two edges intersect and the intersection is in the image, store the intersection point
            self.intersection_pt = i[0]
        else:
            # otherwise, store the intersection point as the average of the two endpoints
            self.intersection_pt = Point((e1.endpt.x + e2.endpt.x) / 2, (e1.endpt.y + e2.endpt.y) / 2)

    def draw(self, img, color_pt, color_lines=None, radius=10, thickness=5):
        """
        Draws the BorderCorner on the image

        :param img: the image that will be drawn on
        :param color_pt: The color the intersection point will be drawn with.
        :param color_lines: The color the two border lines will be drawn with
        :param radius: The radius of the intersection point
        :param thickness: the thickness of the border lines
        """
        if color_lines is not None:
            for e in self.endpoints:
                e.draw(img, color_lines, thickness=thickness)

        cv2.circle(img, (self.intersection_pt.x, self.intersection_pt.y), radius, color_pt, thickness=-1)

    def get_other_endpt(self, e):
        """
        Given one BorderEdge defining this BorderCorner, returns the other BorderEdge.

        :param e: one BorderEdge defining this BorderCorner
        :return: the other BorderEdge
        """
        if e not in self.endpoints:
            raise ValueError("The provided endpoint wasn't one of the two defining this corner")

        return self.endpoints[0] if self.endpoints[0] != e else self.endpoints[1]


def are_close_to_parallel(t1, t2, theta_threshold):
    """
    Determine if two angles are nearly parallel.
    Tries several values to deal with wrap-around.

    :param t1: an angle
    :param t2: an angle
    :param theta_threshold: How close the angles must be to have them labelled as parallel
    :return: True if the angles are close to parallel.
    """
    delta_t = min(abs(t1 - t2), abs(t1 - t2 + np.pi), abs(t1 - t2 - np.pi))
    return delta_t < theta_threshold


def is_similar_seg(s1, s2, seg_to_theta_map, dist_threshold=30, theta_threshold=8 * np.pi / 180):
    """
    Returns whether two segments should be put in the same group.

    :param s1: a Segment geometry object
    :param s2: a Segment geometry object
    :param seg_to_theta_map: a map from a segment geometry object to the angle
    :param dist_threshold: The threshold determining if the segments are close together.
                            When one is projected to a line, the other must be at least this close.
    :param theta_threshold: The threshold determining whether or not the segments are close to parallel. Defaults to 8 degrees
    :return: True if the two segments should be in the same group
    """
    t1 = seg_to_theta_map[s1]
    t2 = seg_to_theta_map[s2]

    if not are_close_to_parallel(t1, t2, theta_threshold):
        return False

    l1 = Line(s1)
    l2 = Line(s2)

    d_l1_s2 = max(l1.distance(s2.points[0]), l1.distance(s2.points[1])) if len(intersection(l1, s2)) == 0 else 0
    d_l2_s1 = max(l2.distance(s1.points[0]), l2.distance(s1.points[1])) if len(intersection(l2, s1)) == 0 else 0

    return max(d_l1_s2, d_l2_s1) < dist_threshold


def get_segs(silhouette):
    """
    Fit segments to the border of the product's silhouette

    :param silhouette: The silhouette mask of the product
    :return: A list of the segments, each of format (x1, y1, x2, y2)
    """
    edges = cv2.Canny(silhouette, 50, 150, apertureSize=3)

    vote_cutoff = int(np.sum(edges) / 27000)
    min_len = int(edges.shape[0] / 18 - 25)

    return np.squeeze(
        cv2.HoughLinesP(edges, 1, np.pi / 180, vote_cutoff, minLineLength=min_len, maxLineGap=min_len)).tolist()


def group_border_segs(segments, seg_to_theta_map):
    """
    Group the segments into BorderEdges if they are close together and nearly parallel

    :param segments: a list of Segment geometry objects
    :param seg_to_theta_map: a map from a segment geometry object to the angle
    :return: a list of BorderEdge objects
    """
    seg_groups = []
    seg_index_map = {}

    for i in range(len(segments)):
        found_match = False
        for j in range(i):
            # if seg i is similar to seg j (which is already in a group), add i to the same group as j
            if is_similar_seg(segments[i], segments[j], seg_to_theta_map):
                found_match = True
                seg_ind = seg_index_map[j]
                seg_index_map[i] = seg_ind
                seg_groups[seg_ind].append(i)
                break

        # if line i wasn't similar to any line already in a group, add line i to its own new group
        if not found_match:
            seg_index_map[i] = len(seg_groups)
            seg_groups.append([i])

    return [BorderEdge([segments[i] for i in g]) for g in seg_groups]


def get_w_d_pt(border_corners):
    """
    Find, among the BorderCorners, the point which is the intersection of the width and depth lines.
    This is assumed to be the corner which is lowest on the image.

    :param border_corners: The list of all BorderCorners
    :return: The single BorderCorner which is the intersection of the width and depth lines.
    """
    return max(border_corners, key=lambda corner: corner.intersection_pt.y)


def get_w_d_other_corners(w_d_pt):
    """
    Given the w_d_pt, find the two other corners defined by the width and depth lines.

    :param w_d_pt: The BorderCorner which is the intersection of the width and depth lines.
    :return: The two other BorderCorners lying at the extremes of the width and depth edges.
    """
    other_corners = []
    for e in w_d_pt.endpoints:
        other_endpoint = e.edge.get_other_endpt(e)
        if len(other_endpoint.corners) != 1:
            raise ValueError("Other endpoint has more than one corner")

        other_corners.append(other_endpoint.corners[0])

    return other_corners


def get_w_h_and_d_back_pts(w_d_other_corner, w_d_pt):
    """
    Decide which of the two other points is the other endpoint of the width line, and which is the other endpoint of the depth line.
    The other endpoint of the width is chosen by looking at the lengths of the edges, scaled by the cosine
    of the edges' angles so that edges which are more vertical are punished.

    :param w_d_other_corner: The two other BorderCorners lying at the extremes of the width and depth edges.
    :param w_d_pt: The BorderCorner which is the intersection of the width and depth lines.
    :return: The w_h_pt first, followed by the d_back_pt
    """
    def get_scaled_dist(c1, c2):
        angle = math.atan2(c1.intersection_pt.y - c2.intersection_pt.y, c1.intersection_pt.x - c2.intersection_pt.x)
        return c1.intersection_pt.distance(c2.intersection_pt) * abs(math.cos(angle))

    w_h_pt = max(w_d_other_corner, key=lambda c: get_scaled_dist(w_d_pt, c))

    return w_h_pt, w_d_other_corner[0] if w_d_other_corner[0] != w_h_pt else w_d_other_corner[1]


def get_h_top_pt(w_h_pt, w_d_pt):
    """
    Get the top of the height line corner.

    :param w_h_pt: The BorderCorner at the bottom of the height line.
    :param w_d_pt: The BorderCorner which is the intersection of the width and depth lines.
    :return: The BorderCorner defining the top of the height line.
    """
    w_h_h_endpt = [endpt for endpt in w_h_pt.endpoints if w_d_pt not in endpt.edge.get_other_endpt(endpt).corners][0]
    return w_h_h_endpt.edge.get_other_endpt(w_h_h_endpt).corners[0]


def get_segs_and_thetas(all_border_segs):
    """
    Convert the segments to Geometry objects, and compute the angle for each segment

    :param all_border_segs: A list of the segments, each of format (x1, y1, x2, y2)
    :return: a list of Segment geometry objects, and a map from a segment geometry object to the angle
    """
    segs = []
    seg_to_theta_map = {}
    for x1, y1, x2, y2 in all_border_segs:
        s = Segment(Point(x1, y1), Point(x2, y2))
        segs.append(s)
        seg_to_theta_map[s] = math.atan2(y2 - y1, x2 - x1)

    return segs, seg_to_theta_map


def get_color(i, num_colors):
    """
    Return the RGB value of the ith fully saturated color around the hue wheel.

    :param i: The index of the color
    :param num_colors: The number of colors that will be drawn
    :return: R, G, B
    """
    r, g, b = hsv_to_rgb(i * 1. / num_colors, 1., 1.)
    return int(r * 255), int(g * 255), int(b * 255)


def get_endpts(seg_group, border_edge):
    """
    Compute the two endpoints of a BorderEdge by finding the two points which are the most distant from the center.

    :param seg_group: All the segments associated with a BorderEdge
    :param border_edge: The BorderEdge we are computing the endpoints of.
    :return: The two EdgeEndPoint objects.
    """
    def get_farthest_seg_endpoint(segs, p):
        max_dist = -1
        farthest_pt = None
        farthest_seg = None

        for seg in segs:
            for i in range(2):
                d = p.distance(seg.points[i])
                if d > max_dist:
                    max_dist = d
                    farthest_pt = seg.points[i]
                    farthest_seg = seg
        return farthest_pt, farthest_seg

    xs = []
    ys = []

    for seg in seg_group:
        xs.append(seg.points[0].x)
        ys.append(seg.points[0].y)

    avg_pt = Point(np.mean(xs), np.mean(ys))
    endpt1, endseg1 = get_farthest_seg_endpoint(seg_group, avg_pt)
    endpt2, endseg2 = get_farthest_seg_endpoint(seg_group, endpt1.midpoint(avg_pt))

    return EdgeEndPoint(endpt1, endseg1, border_edge), EdgeEndPoint(endpt2, endseg2, border_edge)


def get_border_edge_corners(border_edges, img_size):
    """
    Compute the intersection points of the BorderEdges.

    :param border_edges: The list of BorderEdges
    :param img_size: The shape of the image, needed so that corners aren't computed if they are off the image.
    :return: a list of the computed BorderCorner objects.
    """
    all_endpt_dists = []

    for i in range(len(border_edges)):
        for j in range(i):
            for end_pt_i in border_edges[i].endpts:
                for end_pt_j in border_edges[j].endpts:
                    all_endpt_dists.append((end_pt_i, end_pt_j, end_pt_i.endpt.distance(end_pt_j.endpt)))

    all_endpt_dists = sorted(all_endpt_dists, key=lambda x: x[2])
    corners = []
    used_endpts = set()

    for i in range(len(all_endpt_dists)):
        if len(used_endpts) == 2 * len(border_edges):
            break

        e1, e2, _ = all_endpt_dists[i]

        if e1 in used_endpts or e2 in used_endpts:
            continue

        corners.append(BorderCorner(e1, e2, img_size))
        used_endpts.add(e1)
        used_endpts.add(e2)

    return corners


def sort_edges(border_edges):
    """
    Sort the border edges so that they form a circle around the outside of the product.

    :param border_edges: the list of all BorderEdge objects
    :return: the sorted list of BorderEdge objects
    """
    border_edges = [x for x in border_edges]
    # get the starting edge, which will be the longest
    longest_dist = -1
    longest_dist_idx = -1
    for i in range(len(border_edges)):
        d = border_edges[i].length()
        if d > longest_dist:
            longest_dist = d
            longest_dist_idx = i

    longest_edge = border_edges[longest_dist_idx]

    sorted_edge_endpts = [(longest_edge, longest_edge.endpts[0]), (longest_edge, longest_edge.endpts[1])]

    del border_edges[longest_dist_idx]

    # repeat for each edge.  Find the closest end point on another edge, and add those endpoints into the list
    while len(border_edges) > 0:
        min_dist = float('inf')
        min_dist_idx = None
        for i in range(len(border_edges)):
            for j in range(len(border_edges[i].endpts)):
                d = sorted_edge_endpts[-1][1].endpt.distance(border_edges[i].endpts[j].endpt)
                if d < min_dist:
                    min_dist = d
                    min_dist_idx = (i, j)

        next_edge = border_edges[min_dist_idx[0]]
        sorted_edge_endpts += [(next_edge, next_edge.endpts[min_dist_idx[1]]),
                               (next_edge, next_edge.get_other_endpt(next_edge.endpts[min_dist_idx[1]]))]
        del border_edges[min_dist_idx[0]]

    return sorted_edge_endpts


def filter_edge_and_endpts(sorted_edge_endpts, theta_threshold=8 * np.pi / 180, len_ratio=0.25):
    """
    Filter out the border edges which are outliers and don't fit in the cycle, as well as edges which
    are a part of a 4-edge corner artifact or a 3-edge corner artifact.

    A 4-edge corner artifact looks like
         |
        _|
    ___|

    A 3-edge corner artifact looks like
         |
         |
       |
    ___|

    :param sorted_edge_endpts: The sorted list of BorderEdges
    :param theta_threshold: the threshold that determines if edges are nearly parallel for the 3/4-edge corner artifacts
    :param len_ratio: a multiplicative factor to determine if edges are small enough to be removed for 3/4-edge corner artifacts.
    :return: The filtered, sorted list of BorderEdges
    """
    # First remove any outlier edges that get tacked on to the end bc they don't fit elsewhere
    smallest_loop_around_dist = float('inf')
    smallest_loop_around_idx = -1
    for i in range(len(sorted_edge_endpts) - 1, -1, -2):
        d = sorted_edge_endpts[0][1].endpt.distance(sorted_edge_endpts[i][1].endpt)
        if d < smallest_loop_around_dist:
            smallest_loop_around_dist = d
            smallest_loop_around_idx = i

    sorted_edge_endpts = sorted_edge_endpts[:smallest_loop_around_idx + 1]
    sorted_edges = [sorted_edge_endpts[i][0] for i in range(0, len(sorted_edge_endpts), 2)]

    # Filter out any short edges which are part of a 4-edge corner artifact
    start_idx = 0
    while start_idx < len(sorted_edges):
        indices = [(start_idx + i) % len(sorted_edges) for i in range(4)]
        thetas = [sorted_edges[idx].get_angle() for idx in indices]
        lens = [sorted_edges[idx].length() for idx in indices]

        if are_close_to_parallel(thetas[0], thetas[2], theta_threshold) and \
                are_close_to_parallel(thetas[1], thetas[3], theta_threshold) and \
                len_ratio * lens[0] > lens[2] and \
                len_ratio * lens[3] > lens[1]:
            del sorted_edges[indices[2]]
            del sorted_edges[indices[1]]

        start_idx += 1

    # Filter out any short edges which are part of a 3-edge corner artifact
    start_idx = 0
    while start_idx < len(sorted_edges):
        indices = [(start_idx + i) % len(sorted_edges) for i in range(3)]
        thetas = [sorted_edges[idx].get_angle() for idx in indices]
        lens = [sorted_edges[idx].length() for idx in indices]

        if (are_close_to_parallel(thetas[0], thetas[1], theta_threshold) and len_ratio * lens[0] > lens[1]) or \
                (are_close_to_parallel(thetas[2], thetas[1], theta_threshold) and len_ratio * lens[2] > lens[1]):
            del sorted_edges[indices[1]]

        start_idx += 1

    return sorted_edges


def create_dimension_image_side(img, config, line_thickness=10):
    """
    Get the coordinates of the dimension lines using the side-facing image algorithm,
    which fits segments to the border of the product, groups them into individual edges, and computes intersections.

    :param img: The side facing product image
    :param config: Specifics about drawing the image
    :param line_thickness: How thick the edges should be for the debug images.
    :return: the list of debug images, each a numpy array) and
            the mapping from a dimension name to a pair of points defining the two corners of the product lying along that dimension
    """
    silhouette = get_silhouette_mask(img, config)
    silhouette = (255 * np.ones((silhouette.shape[0], silhouette.shape[1], 3)) * np.expand_dims(silhouette, axis=2)).astype(np.uint8)

    # gets all the raw segments lying on the border of the product
    all_border_segs = get_segs(silhouette)

    # Converts each raw segment to a Geometry object Segment, and gets the angle for each segment
    segments, seg_to_theta_map = get_segs_and_thetas(all_border_segs)

    # Groups the segments based on their angles and distances from each other.
    # Creates a BorderEdge for each group of segments
    border_edges = group_border_segs(segments, seg_to_theta_map)

    # Sort the border edges to hopefully be in a complete cycle
    sorted_edge_endpts = sort_edges(border_edges)

    # Filter the sorted border edges and endpoints to remove corner artifacts and outliers
    filtered_edges = filter_edge_and_endpts(sorted_edge_endpts)

    # Computes the distances between each pair of end points of a BorderEdge.
    # Greedily chooses which corners to pair up based on smallest distance.
    # Creates a BorderCorner for each pair.
    border_corners = get_border_edge_corners(filtered_edges, silhouette.shape)

    """
             _____
            /|    ----___ d
           / |           |
          /  |           |
         |   |           |    w_d_pt    is point b
         |   |           |    w_h_pt    is point c
         |   |           |    d_back_pt is point a
         |   |           |    h_top_pt  is point d
        a \  |           |
           \ |        ___|
            \|____----    c
             b
    """

    w_d_pt = get_w_d_pt(border_corners)
    w_d_other_corners = get_w_d_other_corners(w_d_pt)
    w_h_pt, d_back_pt = get_w_h_and_d_back_pts(w_d_other_corners, w_d_pt)
    h_top_pt = get_h_top_pt(w_h_pt, w_d_pt)

    debug_imgs = [silhouette.copy()]

    debug_img = img.copy()
    for i in range(len(all_border_segs)):
        x1, y1, x2, y2 = all_border_segs[i]
        cv2.line(debug_img, (x1, y1), (x2, y2), get_color(i, len(all_border_segs)), thickness=line_thickness)
    debug_imgs.append(debug_img)

    debug_img = img.copy()
    for i in range(len(border_edges)):
        border_edges[i].draw_all_segs(debug_img, get_color(i, len(border_edges)), thickness=line_thickness)
    debug_imgs.append(debug_img)

    debug_img = img.copy()
    sorted_edges = [sorted_edge_endpts[i][0] for i in range(0, len(sorted_edge_endpts), 2)]
    for i in range(len(sorted_edges)):
        sorted_edges[i].draw_all_segs(debug_img, get_color(i, len(sorted_edges)), thickness=line_thickness)
    debug_imgs.append(debug_img)

    debug_img = img.copy()
    for i in range(len(filtered_edges)):
        filtered_edges[i].draw_all_segs(debug_img, get_color(i, len(filtered_edges)), thickness=line_thickness)
    debug_imgs.append(debug_img)

    debug_img = img.copy()
    for i in range(len(filtered_edges)):
        filtered_edges[i].draw_endsegs(debug_img, get_color(i, len(filtered_edges)), thickness=line_thickness)
    debug_imgs.append(debug_img)

    debug_img = img.copy()
    for corner in border_corners:
        corner.draw(debug_img, (0, 0, 255), color_lines=(255, 0, 0), thickness=line_thickness)
    debug_imgs.append(debug_img)

    debug_img = img.copy()
    w_d_pt.draw(debug_img, (0, 255, 0))
    w_h_pt.draw(debug_img, (0, 0, 255))
    d_back_pt.draw(debug_img, (0, 255, 255))
    h_top_pt.draw(debug_img, (255, 255, 0))
    debug_imgs.append(debug_img)

    w_d_pt = w_d_pt.intersection_pt
    w_h_pt = w_h_pt.intersection_pt
    d_back_pt = d_back_pt.intersection_pt
    h_top_pt = h_top_pt.intersection_pt

    center_pt = Point(np.mean([w_d_pt.x, w_h_pt.x, d_back_pt.x, h_top_pt.x]),
                      np.mean([w_d_pt.y, w_h_pt.y, d_back_pt.y, h_top_pt.y]))

    pts_map = {
        "height": (h_top_pt, w_h_pt),
        "width": (w_h_pt, w_d_pt),
        "depth": (w_d_pt, d_back_pt),
        "center": (center_pt,)
    }

    return debug_imgs, pts_map


def filter_silhouette(silo, min_prct_of_largest=0.15):
    """
    Filters the silhouette mask to only contain it's largest connected components

    :param silo: The original silhouette mask
    :param min_prct_of_largest: The only pieces of the silhouette which are kept are those whose size it at least
                                this factor times the size of the largest piece.
                                To only keep a single connected component, set this value to be 1.0.
                                Defaults to 0.15.
    :return: The filtered silhouette mask
    """
    labels, _ = ndi.label(np.sum(silo, axis=2))
    sizes = np.bincount(labels.ravel())
    max_size = np.max(sizes[1:])

    sizes_masked = sizes >= (min_prct_of_largest * max_size)
    sizes_masked[0] = False

    silo_mask = np.expand_dims(sizes_masked[labels], axis=-1)
    return silo * silo_mask


def get_line_specs(img, config):
    """
    Get the specs for the dimension lines based on the size of the image.

    :param img: The image that the lines will be drawn on.  Used so that the lines grow to match a larger image.
    :param config: A configuration map giving the parameters of the dimension lines.
    :return: the margin between the edge of the product and the dimension line, and the thickness of the dimension line
    """
    img_sz = max(img.shape[:2])
    return int(config['line_margin_ratio'] * img_sz), int(config['line_size_ratio'] * img_sz)


def get_font(font_local_filepath, img, config):
    """
    Create a font object of the right size

    :param font_local_filepath: Where the font file is located on the local machine
    :param img: The image that the font will be writing on.  Used so that the font size grows to match a larger image.
    :param config: A configuration map giving the ratio between the image size and the font size
    :return: An ImageFont object
    """
    return ImageFont.truetype(font_local_filepath, int(config['font_size_ratio'] * max(img.shape[:2])))


def create_dimension_image_front(img, config):
    """
    Get the coordinates of the dimension lines using the front-facing image algorithm,
    which basically just finds the bounding box of the product.

    :param img: The front facing product image.
    :param config: Specifics about drawing the image
    :return: the list of debug images, each a numpy array) and
    the mapping from a dimension name to a pair of points defining the two corners of the product lying along that dimension
    """
    def get_min_max_x_y(s):
        silhouette_y, silhouette_x, _ = s.nonzero()
        return min(silhouette_x), max(silhouette_x), min(silhouette_y), max(silhouette_y)

    # a variable with a "_bb" suffix refers to the larger bounding box around the entire foreground,
    # rather than just the filtered large connected components of the silhouette
    silhouette_bb = get_silhouette_mask(img, config)
    silhouette_bb = (255 * np.ones((silhouette_bb.shape[0], silhouette_bb.shape[1], 3)) * np.expand_dims(silhouette_bb, axis=2)).astype(np.uint8)
    silhouette = filter_silhouette(silhouette_bb)

    min_x, max_x, min_y, max_y = get_min_max_x_y(silhouette)
    min_x_bb, max_x_bb, min_y_bb, max_y_bb = get_min_max_x_y(silhouette_bb)

    debug_img = img.copy()
    cv2.rectangle(debug_img, (min_x_bb, min_y_bb), (max_x_bb, max_y_bb), (0, 255, 0), thickness=4)
    cv2.rectangle(debug_img, (min_x, min_y), (max_x, max_y), (0, 0, 255), thickness=2)
    debug_imgs = [debug_img]

    depth_angle = config['front']['depth_angle_deg'] * np.pi / 180
    start_depth_height_ratio = config['front']['start_depth_height_ratio']
    end_depth_height_ratio = config['front']['end_depth_height_ratio']

    upper_depth_y = (start_depth_height_ratio * min_y) + ((1. - start_depth_height_ratio) * max_y)
    lower_depth_y = (end_depth_height_ratio * min_y) + ((1. - end_depth_height_ratio) * max_y)
    depth_delta_x = (lower_depth_y - upper_depth_y) / abs(np.tan(depth_angle))

    if depth_angle < np.pi / 2.:  # angled /
        lower_depth_x = max_x_bb
        upper_depth_x = max_x_bb + depth_delta_x
        height_x = min_x_bb
    else:  # angled \
        lower_depth_x = min_x_bb
        upper_depth_x = min_x_bb - depth_delta_x
        height_x = max_x_bb

    center_pt = Point(np.mean([min_x, max_x]), np.mean([min_y, max_y]))
    height_bottom_pt = Point(height_x, max_y)
    height_top_pt = Point(height_x, min_y)
    width_left_pt = Point(min_x, max_y_bb)
    width_right_pt = Point(max_x, max_y_bb)
    depth_lower_pt = Point(lower_depth_x, lower_depth_y)
    depth_upper_pt = Point(upper_depth_x, upper_depth_y)

    pts_map = {
        "height": (height_bottom_pt, height_top_pt),
        "width": (width_left_pt, width_right_pt),
        "depth": (depth_lower_pt, depth_upper_pt),
        "center": (center_pt,)
    }

    return debug_imgs, pts_map


def add_labels_at_points(img, pts_map, dimension_map, config, font):
    """
    Draws the dimension lines and numbers on the image.

    :param img: The image that the dimensions should be drawn on, as a numpy array
    :param pts_map: Mapping from a dimension name to a pair of points defining the two corners of the product lying along that dimension
    :param dimension_map: Map from the dimension names to the values for this sku
    :param config: Specifics about drawing the image
    :param font: the font object that will write the dimension values.
    :return: the image with the dimension lines and values superimposed, as a numpy array
    """
    line_margin, line_sz = get_line_specs(img, config)

    center_pt = pts_map['center'][0]

    border_top = 0
    border_left = 0

    for dim_name in ["height", "width", "depth"]:
        text = dim_to_text(dimension_map[dim_name], config['num_dim_decimal_pts'])

        p1, p2 = pts_map[dim_name]

        img, border_top, border_left = add_label(img, p1, p2, center_pt, text, font, line_margin, line_sz, border_top,border_left)

    return img


def is_front_facing_depth_and_height(pts_map, dist_factor=1.1, cos_cutoff=0.98):
    """
    If you give the side-facing algorithm a front-facing image, it will tend to have the depth and the height labelled
    as the vertical edges of the rectangle.  This scenario can be detected by measuring if the dimension lines for
    height and depth are close to parallel and close to the same length.  If this is the case, return True.

    :param pts_map: Mapping from a dimension name to a pair of points defining the two corners of the product lying along that dimension
    :param dist_factor: a multiplicative factor determining if the dimension lines are "close to the same length".
                        The longer can be at most this factor times the length of the shorter.
                        Should be at least 1.0, and larger values make it easier to say they are similar.
    :param cos_cutoff: a threshold determining if the dimension lines are "close to parallel".
                        The cosine of the angle between the two must be at least this threshold.
                        Should be between 0.0 and 1.0, and a smaller value makes it easier to say they are parallel.
    :return: a boolean flag indicating whether it seems like this is actually a front-facing image.
    """
    h_p1, h_p2 = pts_map['height']
    d_p1, d_p2 = pts_map['depth']

    h_len = h_p1.distance(h_p2)
    d_len = d_p1.distance(d_p2)

    if min(h_len, d_len) * dist_factor < max(h_len, d_len):
        return False

    h_dx = h_p1.x - h_p2.x
    h_dy = h_p1.y - h_p2.y
    d_dx = d_p1.x - d_p2.x
    d_dy = d_p1.y - d_p2.y

    cos_theta = abs(((h_dx * d_dx) + (h_dy * d_dy)) / (h_len * d_len))

    return cos_theta >= cos_cutoff


def should_swap_hd_dims(pts_map, dimension_map, swap_coeff=0.6):
    """
    For some product types (e.g. cooktops), the images that we label front-facing are actually showing the top of the product.
    The easiest solution is to just swap what we are calling the "height" and the "depth".
    This function detects this case by computing the aspect ratio of the bounding box in the image and comparing it
    against the aspect ratio of IDM dimension values before and after this swap.
    If the error between the measured and IDM aspect ratios decreases enough, this function returns True and enables the swap.

    :param pts_map: Mapping from a dimension name to a pair of points defining the two corners of the product lying along that dimension
    :param dimension_map: Map from the dimension names to the values for this sku
    :param swap_coeff: a multiplicative factor controlling how much lower the error must be after swapping to make it worth it.
                        Should probably be between 0 and 1.
                        A lower values makes it harder to swap, since the error after swapping has to be less than
                        this factor times the error before swapping.
    :return: a boolean flag indicating that the empirical aspect ratio of the bounding box would be more closely matched
            if the height and depth for the product were switched.
    """
    def dist(pts_list): return float(pts_list[0].distance(pts_list[1]))

    def ratio_error(r_m, r_idm): return abs(r_m - r_idm) / r_idm

    # the measured lengths from the image
    w_len_m = dist(pts_map['width'])
    h_len_m = dist(pts_map['height'])

    # the lengths according to IDM
    w_len_idm = dimension_map['width']
    d_len_idm = dimension_map['depth']
    h_len_idm = dimension_map['height']

    if w_len_idm is None or d_len_idm is None or h_len_idm is None or np.isnan(w_len_idm) or np.isnan(
            d_len_idm) or np.isnan(h_len_idm):
        return False

    # the measured aspect ratio
    hw_ratio_m = h_len_m / w_len_m

    # the aspect ratio if we keep the IDM height
    hw_ratio_idm = h_len_idm / w_len_idm
    # the aspect ratio if we use the depth instead of the height
    dw_ratio_idm = d_len_idm / w_len_idm

    return ratio_error(hw_ratio_m, dw_ratio_idm) <= ratio_error(hw_ratio_m, hw_ratio_idm) * swap_coeff


def create_dimension_image(img, config, dimension_map, font_local_filepath, is_side_img, product_type,
                           allow_swap_to_front):
    """
    Create the dimension image and the debug images showing how it was formed.

    :param img: The input image that dimensions should be applied to
    :param config: Specifics about drawing the image
    :param dimension_map: Map from the dimension names to the values for this sku
    :param font_local_filepath: Where the font file is located on the local machine
    :param is_side_img: a boolean indicating if the image is labelled as a side-facing image
    :param product_type: The product type of this sku
    :param allow_swap_to_front: a boolean indicating whether or not the algorithm can decide to change an image labelled
                                as side-facing to one labelled as front-facing, based on the results of the side-facing algorithm
    :return: the dimension image and the list of debug images, all as numpy arrays
    """
    font = get_font(font_local_filepath, img, config)
    img_w_border = add_border(img, config['border_ratio'])

    if is_side_img:
        debug_imgs, pts_map = create_dimension_image_side(img_w_border, config)

        if allow_swap_to_front and is_front_facing_depth_and_height(pts_map):
            debug_imgs_front, pts_map = create_dimension_image_front(img_w_border, config)
            debug_imgs += debug_imgs_front
    else:
        debug_imgs, pts_map = create_dimension_image_front(img_w_border, config)

    if (not is_side_img) and product_type in config['allow_swap_height_depth_product_types'] and should_swap_hd_dims(
            pts_map, dimension_map):
        temp = pts_map['height']
        pts_map['height'] = pts_map['depth']
        pts_map['depth'] = temp

    dimension_image = add_labels_at_points(img_w_border, pts_map, dimension_map, config, font)

    return dimension_image, debug_imgs


# def save_img_to_storage(image, product_type, img_filename, img_save_dir, local_img_filename_no_ext="temp_img"):
#     """
#     Save the image to the specified location in GCS
#
#     :param image: The image to be save, as a numpy array
#     :param product_type: The product type for the product
#     :param img_filename: The filename that the image will be saved under
#     :param img_save_dir: The directory in GCS where the image will be saved
#     :param local_img_filename_no_ext: The local filename of the image before it is saved to GCS
#     """
#     _, extension = os.path.splitext(img_filename)
#
#     local_img_filename = local_img_filename_no_ext + extension
#
#     skio.imsave(local_img_filename, image)
#
#     transfer_local_file_to_storage(local_img_filename, "%s/%s/%s" % (img_save_dir, product_type, img_filename))


def create_and_save_all_images(spark, day, config_loc, output_dir, image_type, save_debug_imgs, allow_swap_to_front):
    logging.info("\n\nloading the drawing config file")
    drawing_config_local_filepath = "drawing_config.json"
    transfer_storage_file_to_local(config_loc, drawing_config_local_filepath)
    with open(drawing_config_local_filepath, 'r') as fp:
        config = json.load(fp)
    logging.info("\n\nConfig: %s" % config)

    # product_type, oms_id, image_guid, image_type
    logging.info("\n\nloading the merged image data")
    merged_img_df = get_merged_img_df(output_dir, day, spark)

    # oms_id, width, height, depth
    logging.info("\n\nloading the dimension data")
    dimension_df = get_dimension_df(output_dir, day, spark, config['num_dim_decimal_pts'])

    logging.info("\n\nloading the font file")
    font_local_filepath = "font.otf"
    transfer_storage_file_to_local(config['font_file_loc'], font_local_filepath)

    # product_type, oms_id, image_guid, image_type, width, height, depth
    logging.info("\n\ncombining images with dimensions")
    front_img_dim_pd_df = get_one_image_type_with_dims(merged_img_df, dimension_df, image_type).toPandas()

    logging.info("\n\nAdding dimensions to images")

    for product_type in front_img_dim_pd_df.product_type.unique():
        logging.info("\n\nGenerating Images for %s" % product_type)

        app_rows = front_img_dim_pd_df[front_img_dim_pd_df.product_type == product_type]
        print_step_size = int(math.ceil(len(app_rows) / 10))
        num_imgs_done = 0

        for i in app_rows.index:
            if num_imgs_done % print_step_size == 0:
                logging.info("\nGenerating Image %d out of %d at time %s" % (
                    num_imgs_done + 1, len(app_rows), str(datetime.datetime.now())))

            num_imgs_done += 1
            row = app_rows.loc[i]

            try:
                img = get_image(row.image_guid)

                dimension_map = {
                    "height": row.height,
                    "width": row.width,
                    "depth": row.depth
                }

                dimension_image, debug_imgs = create_dimension_image(img, config, dimension_map, font_local_filepath,
                                                                     row.image_type == "SIDE", row.product_type,
                                                                     allow_swap_to_front)

                # save_img_to_storage(dimension_image, row.product_type, "%s.jpg" % row.oms_id,
                #                     "%s/%s/%s" % (output_dir, format_day(day, '-'), IMAGES_OUTPUT_SUFFIX))
                # save_img(img, filename, storage_dir, local_filename=None, local_dir=None, image_sizes=None)
                save_img(dimension_image, "%s.jpg" % row.oms_id,
                         "%s/%s/%s/%s" % (output_dir, format_day(day, '-'), IMAGES_OUTPUT_SUFFIX, row.product_type),
                         local_filename="temp.jpg")

                if save_debug_imgs:
                    for j in range(len(debug_imgs)):
                        # save_img_to_storage(debug_imgs[j], row.product_type, "%s_%d.jpg" % (row.oms_id, j),
                        #                     "%s/%s/%s" % (output_dir, format_day(day, '-'), DEBUG_IMGS_SUFFIX))
                        save_img(debug_imgs[j], "%s_%d.jpg" % (row.oms_id, j),
                                 "%s/%s/%s/%s" % (output_dir, format_day(day, '-'), DEBUG_IMGS_SUFFIX, row.product_type),
                                 local_filename="temp.jpg")
            except Exception as e:
                print("Problem generating image for oms_id %s:" % row.oms_id)
                print(''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--date", required=False,
                        help="The date to generate images for in format YYYY-MM-DD, defaults to the most recent day",
                        default=None, dest="date")
    parser.add_argument("--config_loc", required=False,
                        help="The config file, default\"gs://hd-datascience-np-artifacts/jim/dimension_images/config/image_drawing_config.json\"",
                        default="gs://hd-datascience-np-artifacts/jim/dimension_images/config/image_drawing_config.json",
                        dest="config_loc")
    parser.add_argument("--output_dir", required=False,
                        help="file where the output will be written, default \"gs://hd-datascience-np-data/dimension_images\"",
                        default="gs://hd-datascience-np-data/dimension_images", dest="output_dir")
    parser.add_argument("--image_type", required=False,
                        help='The type of images to generate (should be either "FRONT" or "SIDE"), default is both',
                        default=None, dest="image_type")
    parser.add_argument("--save_debug_imgs", required=False, help='Include this flag to save debug images',
                        action="store_true", dest="save_debug_imgs")
    parser.add_argument("--allow_swap_to_front", required=False,
                        help='If this flag is included, the system will examine the results for side-facing images. If it looks like the depth and height are parallel and the same length, will regenerate as a front-facing image.',
                        action="store_true", dest="allow_swap_to_front")

    args = parser.parse_args()

    if args.date is None:
        day = get_most_recent_date(args.output_dir)
    else:
        day = parse_date(args.date, divider='-')
    logging.info("\n\nGetting dimensions for day %s" % day)

    spark = SparkSession.builder \
        .appName("Dimension Images (Front-Facing Images)") \
        .config("spark.sql.shuffle.partitions", "50") \
        .getOrCreate()

    create_and_save_all_images(spark, day, args.config_loc, args.output_dir, args.image_type, args.save_debug_imgs,
                               args.allow_swap_to_front)

    spark.stop()
