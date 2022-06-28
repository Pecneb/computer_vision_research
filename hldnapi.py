"""
    Darknet API to make it easy to use it with Opencv.
    Copyright (C) 2022 Bence Peter

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Contact email: ecneb2000@gmail.com
"""

from ctypes import *
import cv2 as cv
import darknet
import argparse

"""Configuration files for darknet"""
CONFIG = "./darknet_config_files/yolov4.cfg"
DATA = "./darknet_config_files/coco.data"
WEIGHTS = "./darknet_config_files/yolov4.weights"

"""Loading configuration files into darknet"""
network, class_names, colors = darknet.load_network(CONFIG, DATA, WEIGHTS)
darknet_width = darknet.network_width(network)
darknet_height = darknet.network_height(network)

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def cvimg2detections(image):
    """Fcuntion to make it easy to use darknet with opencv

    Args:
        image (Opencv image): input image to run darknet on

    Returns:
        detections(tuple): detected objects on input image (label, confidence, bbox(x,y,w,h))
    """
    # Convert frame color from BGR to RGB
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Resize image for darknet
    image_resized = cv.resize(image_rgb, (darknet_width, darknet_height), interpolation=cv.INTER_LINEAR)
    # Create darknet image
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    # Convert cv2 image to darknet image format
    darknet.copy_image_from_bytes(img_for_detect, image_resized.tobytes())
    # Load image into nn and get detections
    detections = darknet.detect_image(network, class_names, img_for_detect)
    darknet.free_image(img_for_detect)
    # Resize bounding boxes for original frame
    detections_adjusted = []
    for label, confidence, bbox in detections:
        bbox_adjusted = convert2original(image, bbox)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))        
    return detections_adjusted