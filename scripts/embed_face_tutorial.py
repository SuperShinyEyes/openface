from openface.align_dlib import AlignDlib
import openface
import cv2
import numpy as np

#-----------------------------------------------------
# openFace models: https://cmusatyalab.github.io/openface/models-and-accuracies/
'''
Number of parameters
--------------------
- nn4.small2: 3733968
- nn4.small1: 5579520
- nn4:        6959088
- nn2:        7472144

Accuracies & AUC
----------------
nn4.small2.v1 (Default)	0.9292 +- 0.0134,   0.973
nn4.small1.v1           0.9210 +- 0.0160,   0.973
nn4.v2                  0.9157 +- 0.0152,   0.966
nn4.v1                  0.7612 +- 0.0189,   0.853
'''

# MODEL_NAME = 'nn4.v1'; landmarkIndices = openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
# MODEL_NAME = 'nn4.v2'; landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
# MODEL_NAME = 'nn4.small1.v1'; landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
MODEL_NAME = 'nn4.small2.v1'; landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
MODEL_PATH = '/app/weights/%s.t7' % MODEL_NAME


imgDim = 96
align = openface.AlignDlib('/app/weights/shape_predictor_68_face_landmarks.dat')
#---------------------------------------------
# Load OpenFace neural net
# Can't use CUDA:
# /root/torch/install/bin/luajit: /root/torch/install/share/lua/5.1/trepl/init.lua:384: module 'cutorch' not found:No LuaRocks module found for cutorch
net = openface.TorchNeuralNet(MODEL_PATH, imgDim, cuda=False)

def get_video_metadata(cap, video_id):
    fps = cap.get(cv2.cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT)
    print("Video #", video_id)
    print("FPS:", fps)
    print("Number of frames:", frame_count)

    return int(fps), int(frame_count)

for video_id in range(1, 18):

    video_path = '/app/videos/unravel%d_cropped.mp4' % video_id
    cap = cv2.VideoCapture(str(video_path))
    fps, frame_count = get_video_metadata(cap, video_id)
    i = 0

    embedding_array = np.zeros([frame_count, 128], dtype=np.float64)

    exist_frame, frame = cap.read()
    # import pdb;pdb.set_trace()
    while(exist_frame):
        print('Video #%d\t Frame #%d / %d' % (video_id, i, frame_count))
        # if i == 30: break
        # rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgbImg = frame

        bb = align.getLargestFaceBoundingBox(
            rgbImg=rgbImg,
            skipMulti=False,     # Skip image if more than one face detected.
        )

        if bb is not None:
            #---------------------------------------------
            # Apply alignment & crop transformation
            alignedFace = align.align(
                imgDim=imgDim,
                rgbImg=rgbImg,
                bb=bb,           # Bounding box around the face to align. \
                                # Defaults to the largest face.
                landmarks=None,  # Detected landmark locations. \
                                # Landmarks found on `bb` if not provided.
                landmarkIndices=landmarkIndices, # The indices to transform to.
                skipMulti=False  # Skip image if more than one face detected
            )

            # img1 = cv2.rectangle(img ,(x,y),(x+w,y+h),(255, 0, 0), 2)
            # cv2.imwrite('unravel%d_frame%07d_aligned.png' % (video_id, i), alignedFace)
            cv2.imwrite('results-aligned%d/%d.png' % (video_id, i), alignedFace)
            # plt.imsave('unravel17_frame500_aligned.png', alignedFace)

            #---------------------------------------------
            # Landmark list:  List[Tuple[int, int]]
            # Length: 68
            landmarks = align.findLandmarks(rgbImg, bb)

            rep = net.forward(alignedFace)

            embedding_array[i] = rep

            del frame
            del bb
            del alignedFace
            del rgbImg
            del landmarks
            del rep
            
            # import pdb; pdb.set_trace()

        exist_frame, frame = cap.read()
        i += 1

    np.save('results-aligned%d/unravel%d_embedding.npy' % (video_id, video_id), embedding_array)

    cap.release()
    cv2.destroyAllWindows()
    del cap