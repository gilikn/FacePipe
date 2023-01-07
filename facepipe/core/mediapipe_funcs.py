# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

## Based on: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/crop_mouth_from_video.py

from collections import defaultdict
from collections import deque

import cv2
import numpy as np
import torch
import torchvision.ops.boxes as bops
from skimage import transform as tf

from facepipe.config.pre_processing_config import HORIZONTAL_MEAN_FACE_INDENT, VERTICAL_MEAN_FACE_INDENT, \
    MAX_FACES, STD_SIZE, stablePntsIDs, mediapipePntsIds, MEAN_FACE_PATH, WINDOW_MARGIN


def arrayed(landmarks, orig_width, orig_height):
    pts = np.array(landmarks)
    pts_lst = []
    for pt in pts:
        pts_lst.append(np.array([pt.x * orig_width, pt.y * orig_height]))
    return np.array(pts_lst)


def write_video_cv(rois, target_path, fps):
    video_out = cv2.VideoWriter(target_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (224, 224))
    for roi in rois:
        video_out.write(roi)
    video_out.release()
    return


def get_bounding_box(landmarks_coord_np: np.array):
    xmin = int(np.min(landmarks_coord_np[:, 0]))
    xmax = int(np.max(landmarks_coord_np[:, 0]))
    ymin = int(np.min(landmarks_coord_np[:, 1]))
    ymax = int(np.max(landmarks_coord_np[:, 1]))
    bounding_box = torch.tensor([[xmin, ymin, xmax, ymax]])
    return bounding_box


def canonize_faces_by_frame(hot_locations, landmarks, create_offset=True):
    canonized_landmarks = []
    land_idx = 0
    if create_offset:
        offset = hot_locations.tolist().index(1)
        hot_locations = hot_locations[offset:]
    else:
        offset = 0
    for can_land_idx, x in enumerate(hot_locations):
        if x == 0:
            canonized_landmarks.append(None)
        else:
            canonized_landmarks.append(landmarks[land_idx])
            land_idx = land_idx + 1
    return offset, canonized_landmarks


def keep_maximal_face(face_heat_map, landmarks_arr):
    max_idx = np.argmax(face_heat_map.sum(axis=0))
    return face_heat_map[:, np.argmax(face_heat_map.sum(axis=0))], landmarks_arr[max_idx]


def load_and_normalize_mean_face(mean_face_path):
    mean_face_landmarks = np.load(mean_face_path)
    mean_face_landmarks[:, 0] = mean_face_landmarks[:, 0] + HORIZONTAL_MEAN_FACE_INDENT
    mean_face_landmarks[:, 1] = mean_face_landmarks[:, 1] + VERTICAL_MEAN_FACE_INDENT
    return mean_face_landmarks


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks


def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def media_pipe_preprocess_video(output_video_path, mean_face_path, detector=None, frames=None, fps=30):
    last_bounding_boxes = {}
    landmarks = defaultdict(list)
    face_heat_map = np.zeros((len(frames), MAX_FACES))
    for frame_idx, frame in enumerate(frames):
        results = detector.process(frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_coord_np = arrayed([face_landmarks.landmark[i] for i in range(478)], frame.shape[1],
                                             frame.shape[0])
                bounding_box = get_bounding_box(landmarks_coord_np)
                max_iou = 0
                max_key = 0
                for key, last_box in last_bounding_boxes.items():
                    iou = bops.box_iou(bounding_box, last_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_key = key
                if max_iou > 0.2:
                    if face_heat_map[frame_idx, max_key] == 0:
                        landmarks[max_key].append(landmarks_coord_np)
                        last_bounding_boxes[max_key] = bounding_box
                        face_heat_map[frame_idx, max_key] = 1
                else:
                    potential_new_key = len(landmarks.keys())
                    if potential_new_key < MAX_FACES:  # remove this condition if filter is used
                        landmarks[potential_new_key].append(landmarks_coord_np)
                        last_bounding_boxes[potential_new_key] = bounding_box
                        face_heat_map[frame_idx, potential_new_key] = 1

    face_heat_map, landmarks = keep_maximal_face(face_heat_map, landmarks)
    if sum(face_heat_map) < 0.2 * len(frames):  # not enough faces were detected
        return None, None, None

    offset, landmarks_i = canonize_faces_by_frame(face_heat_map, landmarks, create_offset=False)
    preprocessed_landmarks = landmarks_interpolate(landmarks_i)
    rois = new_crop_patch(frames, preprocessed_landmarks, mean_face_path, offset=offset)
    write_video_cv(rois, f"{output_video_path}", fps)
    return rois, preprocessed_landmarks, offset


def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped


def new_crop_patch(frames, landmarks, mean_face_path, offset=0):
    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """
    mean_face_landmarks = load_and_normalize_mean_face(mean_face_path)
    frames = frames[offset:]
    num_frames = len(frames)
    frame_idx = 0
    margin = min(num_frames, WINDOW_MARGIN)
    for i in range(len(landmarks)):
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []
        frame = frames[i]
        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img(smoothed_landmarks[mediapipePntsIds, :],
                                          mean_face_landmarks[stablePntsIDs, :],
                                          cur_frame,
                                          STD_SIZE)
            sequence.append(trans_frame)
        if frame_idx == len(landmarks) - 1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
                sequence.append(trans_frame)
            return np.array(sequence)
        frame_idx += 1
    return None
