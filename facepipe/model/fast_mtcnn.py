# Based on https://www.kaggle.com/code/timesler/fast-mtcnn-detector-55-fps-at-full-resolution?scriptVersionId=29201326

import warnings

import cv2
import numpy as np
from facenet_pytorch import MTCNN

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def coordinates_to_measures(boxes, resize):
    for i, box in enumerate(boxes):
        if box is not None:
            new = []
            for face in box:
                new.append(get_measures(face, resize))
            boxes[i] = new
    return boxes


def get_measures(box, resize):
    de_resize = int(1 / resize)
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x1, y1, w, h = x1 * de_resize, y1 * de_resize, w * de_resize, h * de_resize
    return np.array([x1, y1, w, h])


class FastMTCNN(object):
    """Fast MTCNN implementation."""

    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.

        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.

        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]

        boxes, probs = self.mtcnn.detect(frames[::self.stride])
        faces_count = 0
        bounding_boxes = []
        boxes = coordinates_to_measures(boxes, self.resize)
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                bounding_boxes.append(None)
            else:
                current_faces = [{'box': box, 'confidence': prob} for box, prob in zip(boxes[box_ind], probs[box_ind])]
                bounding_boxes.append(current_faces)
                faces_count = faces_count + len(current_faces)

        return bounding_boxes
