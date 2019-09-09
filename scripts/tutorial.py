import cv2
from openface.align_dlib import AlignDlib
import openface
import argparse
import itertools
import os
import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(precision=2)

imgDim = 96

img= cv2.imread("nikita.png")
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

align = openface.AlignDlib('/app/weights/shape_predictor_68_face_landmarks.dat')

bb = align.getLargestFaceBoundingBox(
    rgbImg=img,
    skipMulti=False,     # Skip image if more than one face detected.
)


x = bb.left()
y = bb.top()
w = bb.width()
h = bb.height()

#---------------------------------------------
# Apply alignment & crop transformation
alignedFace = align.align(
    imgDim=imgDim,
    rbgImg=img,
    bb=bb,           # Bounding box around the face to align. \
                     # Defaults to the largest face.
    landmarks=None,  # Detected landmark locations. \
                     # Landmarks found on `bb` if not provided.
    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE, # The indices to transform to.
    skipMulti=False  # Skip image if more than one face detected
)

# img1 = cv2.rectangle(img ,(x,y),(x+w,y+h),(255, 0, 0), 2)
cv2.imwrite('nikita-rec.png', alignedFace)

#---------------------------------------------
# Landmark list:  List[Tuple[int, int]]
# Length: 68
landmarks = align.findLandmarks(img, bb)
import pdb; pdb.set_trace()

#---------------------------------------------
# Load OpenFace neural net
net = openface.TorchNeuralNet('/app/weights/nn4.v1.t7', imgDim)
rep = net.forward(alignedFace)

# plt.
# plt.subplot(131)
# plt.imshow(img)
# plt.subplot(132)
# plt.imshow(img1)
# plt.subplot(133)
# crop_img = img1[y:y+h, x:x+w]
# cv2.imwrite('nikita-crop.png', crop_img)

