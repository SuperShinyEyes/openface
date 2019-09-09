from PIL import Image
from openface.align_dlib import AlignDlib
import openface
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from moviepy.editor import VideoFileClip


video_path = 'unravel17_cropped.mp4'
clip = VideoFileClip(str(video_path))
fn = clip.duration * clip.fps
import pdb; pdb.set_trace()
frame = clip.get_frame(fn)