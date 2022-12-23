import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import wget
import sys
import tempfile
import tqdm
import threading

import utils_pose as utils
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from movenet import Movenet
from data import BodyPart

import tensorflow as tf

from tensorflow import keras
import tensorflow as tf
import tensorflowjs as tfjs

import matplotlib.pyplot as plt
from itertools import cycle

movenet = Movenet('movenet_thunder')

def get_center_point(landmarks, left_bodypart, right_bodypart):
  """Calculates the center point of the two given landmarks."""

  left = tf.gather(landmarks, left_bodypart.value, axis=1)
  right = tf.gather(landmarks, right_bodypart.value, axis=1)
  center = left * 0.5 + right * 0.5
  return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
  """Calculates pose size.

  It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
  """
  # Hips center
  hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                 BodyPart.RIGHT_HIP)

  # Shoulders center
  shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                      BodyPart.RIGHT_SHOULDER)

  # Torso size as the minimum body size
  torso_size = tf.linalg.norm(shoulders_center - hips_center)

  # Pose center
  pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                     BodyPart.RIGHT_HIP)
  pose_center_new = tf.expand_dims(pose_center_new, axis=1)
  # Broadcast the pose center to the same size as the landmark vector to
  # perform substraction
  pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (17*2), 17, 2])

  # Dist to pose center
  d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")
  # Max dist to pose center
  max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

  # Normalize scale
  pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

  return pose_size


def normalize_pose_landmarks(landmarks):
  """Normalizes the landmarks translation by moving the pose center to (0,0) and
  scaling it to a constant pose size.
  """
  # Move landmarks so that the pose center becomes (0,0)
  pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                 BodyPart.RIGHT_HIP)
  pose_center = tf.expand_dims(pose_center, axis=1)
  # Broadcast the pose center to the same size as the landmark vector to perform
  # substraction
  pose_center = tf.broadcast_to(pose_center,
                                [tf.size(landmarks) // (17*2), 17, 2])
  landmarks = landmarks - pose_center

  # Scale the landmarks to a constant pose size
  pose_size = get_pose_size(landmarks)
  landmarks /= pose_size

  return landmarks


def landmarks_to_embedding(landmarks_and_scores):
  """Converts the input landmarks into a pose embedding."""
  # Reshape the flat input into a matrix with shape=(17, 3)
  reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

  # Normalize landmarks 2D
  landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])

  # Flatten the normalized landmark coordinates into a vector
  embedding = keras.layers.Flatten()(landmarks)

  return embedding

def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
  # Draw the detection result on top of the image.
  image_np = utils.visualize(image, [person])

  # Plot the image with detection results.
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
#   im = ax.imshow(image_np)

  if close_figure:
    plt.close(fig)

  if not keep_input_size:
    image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

  return image_np

def get_keypoint_landmarks(person):
    pose_landmarks = np.array(
      [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
        for keypoint in person.keypoints],
      dtype=np.float32)
    return pose_landmarks


def predict_pose(model, lm_list):

    global label
#     lm_list = np.array(lm_list)
#     lm = lm.reshape(lm,(1,5,34))
#     lm_list = np.expand_dims(lm_list, axis=0)
#     print(lm_list.shape)
    results = model.predict(lm_list)
    print(np.max(results))
    if np.max(results) >= 0.6 and np.max(results) < 16:
        label = class_names[np.argmax(results)]
    return label

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 0, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(input_tensor, inference_count=3):
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)

    for _ in range(inference_count - 1):
        detection = movenet.detect(input_tensor.numpy(),
                                reset_crop_region=False)

    return detection


class_names = ['pose_1', 'pose_10', 'pose_11', 'pose_12', 'pose_13', 'pose_14',
       'pose_15', 'pose_16', 'pose_2', 'pose_3', 'pose_4', 'pose_5',
       'pose_6', 'pose_7', 'pose_8', 'pose_9']



def is_available_dou(list , var1, var2):
    for i in range(len(list)-1):
        if list[i] == var1 and list[i+1] ==  var2:
            return True
    return False

def is_available(list , var):
    for i in list:
        if i == var:
            return True
    return False

def standard_pose16(list):
    print(len(list)-1)
    for i in range(len(list)-2):
        if list[i] == 'pose_16':
            list.pop(i)
    return list

def standardize(list):
    j = 0
    for i in range(0,len(list)):
        if list[i] == 'pose_16':
            j = i

    new_list_label = []
    for i in range(0,j):
        new_list_label.append(list[i])
#     print(new_list_label)
    return new_list_label

def normalsize(list):
    i = 0
    while i < len(list)-1:
        if list[i] == list[i+1]:
            list.pop(i)
#             print(i)
            i -= 1
#             print(i)
        i += 1
    return list

def list_except(list1 , list2):
    list = []
    for i in list2:
        if not is_available(list1 ,i):
            list.append(i)
    return list

def colab_list(list1, list2):
    seri_label = ['pose_1','pose_2','pose_3','pose_4','pose_5','pose_6','pose_7','pose_8',
              'pose_9','pose_10','pose_11','pose_12','pose_13','pose_14','pose_15','pose_16']
    list = []
    for i in seri_label:
        for l1 in list1:
            if i == l1:
                list.append(i)
        for l2 in list2:
            if i == l2:
                list.append(i)
    return list