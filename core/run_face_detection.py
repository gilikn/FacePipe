import datetime
import logging
import os
import subprocess

import cv2
import mediapipe as mp
import numpy as np
import torch
from imutils.video import FileVideoStream
from tqdm import tqdm

from facepipe.model.fast_mtcnn import FastMTCNN
from facepipe.core.mediapipe_funcs import media_pipe_preprocess_video
from facepipe.config.pre_processing_config import CROP_DIRECTORY, AUDIO_DIRECTORY, ENLARGE_SCALAR, BASE_THRESHOLD, MAX_FACES

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def run_detection(fast_mtcnn, filename):
    """
    :param fast_mtcnn: Detector based on Fast-MTCNN implementation.
    :param filename: The path to the processed video.
    :return: List of the bounding boxes detected.
    """
    frames = []
    batch_size = 80
    bounding_boxes_total = []

    v_cap = FileVideoStream(filename).start()
    v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    for j in range(v_len):

        frame = v_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if len(frames) >= batch_size or j == v_len - 1:
            bounding_boxes = fast_mtcnn(frames)
            bounding_boxes_total = bounding_boxes_total + bounding_boxes
            frames = []

    v_cap.stop()
    return bounding_boxes_total


def keep_in_frame(x1, x2, y1, y2, vid_width, vid_height, return_padding=True):
    """
    :param x1: The x1 coordinate value.
    :param x2: The x2 coordinate value.
    :param y1: The y1 coordinate value.
    :param y2: The y2 coordinate value.
    :param vid_width: The video width.
    :param vid_height: The video length.
    :param return_padding: Whether padding values should be returned or not.
    :return: The coordinates and padding required in case the bounding box exceeds frame boundary.
    """
    pad_left, pad_right, pad_up, pad_down = 0, 0, 0, 0
    if x1 < 0:
        pad_left = np.abs(x1)
        x1 = 0
    if x2 > vid_width:
        pad_right = np.abs(x2 - vid_width)
        x2 = vid_width
    if y1 < 0:
        pad_up = np.abs(y1)
        y1 = 0
    if y2 > vid_height:
        pad_down = np.abs(vid_height - y2)
        y2 = vid_height
    if return_padding:
        return x1, x2, y1, y2, pad_left, pad_right, pad_up, pad_down
    else:
        return x1, x2, y1, y2


def modify_bounding_box(bounding_box):
    """
    :param bounding_box: A bounding box in x1, y1, width and height format.
    :return: An enlarged bounding box in x1, x2, y1, y2 format.
    """
    x1, y1, w, h = bounding_box
    center_x = int(x1 + w / 2)
    center_y = int(y1 + h / 2)
    side = max(w, h)
    new_side = ENLARGE_SCALAR * side
    x1 = center_x - int(new_side / 2)
    x2 = center_x + int(new_side / 2)
    y1 = center_y - int(new_side / 2)
    y2 = center_y + int(new_side / 2)
    return (x1, x2, y1, y2)


def find_face_MTCNN(box_list, area, vid_width, vid_height, avg_most_conf):
    """
    :param box_list: List of bounding boxes.
    :param area: The area in which the face should be detected in.
    :param vid_width: The video width.
    :param vid_height: The video height.
    :param avg_most_conf: The average size of the most confident face detected.
    :return: A bounding of a face if was detected and it passes the logical statements, or False otherwise.
    """
    for face in box_list:
        if face['confidence'] > BASE_THRESHOLD:
            if avg_most_conf * 0.80 <= calc_area_of_box(face['box']) <= avg_most_conf * 1.20:
                center = centralize(face['box'])
                if is_near_area(center['center'], area['center'], vid_width, vid_height):
                    return face
    return False


def add_audio(directory_path, file_name, video_path, input_path):
    """
    Adds audio to the requested video.
    """
    subprocess.call(
        f"ffmpeg -y -i {video_path} -i {file_name.split('@')[0].split('.')[0]}.mp4 -map 0:v -map 1:a -c copy {os.path.join(directory_path, AUDIO_DIRECTORY, file_name)}",
        shell=True, cwd=input_path)


def get_video_metadata(video_path):
    """
    :param video_path: The path to the video.
    :return: A tuple of parameters; the cv2 video capture obkect of the video, width, height, length and FPS.
    """
    video_capture = cv2.VideoCapture(video_path)
    vid_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    return video_capture, vid_width, vid_height, length, fps


def count_faces_in_video(bounding_boxes, avg_most_conf):
    """
    :param bounding_boxes: List of bounding boxes.
    :param avg_most_conf: average size of the most confident face detect.
    :return:
    """

    count_hist = {}
    for frame_faces in bounding_boxes:
        count_faces = 0
        if frame_faces is not None:
            for face in frame_faces:
                if face['confidence'] > BASE_THRESHOLD:
                    count_faces += 1
                elif face['confidence'] > 0.9 and (
                        avg_most_conf * 0.8 <= calc_area_of_box(face['box']) <= avg_most_conf * 1.2):
                    count_faces += 1
            count_hist[count_faces] = count_hist.get(count_faces) + 1 if count_faces in count_hist.keys() else 1
    return max(count_hist, key=count_hist.get)


def is_near_area(center1, center2, vid_width, vid_height):
    """
    :param center1: The x, y coordinates of the center of the first face.
    :param center2: The x, y coordinates of the center of the second face.
    :param vid_width: The width of the frame.
    :param vid_height: The height of the frame.
    :return: True or false, whether center2 is close to center1 by maximum of 20% of height and width of the video.
    """

    center1_x, center1_y = center1
    center2_x, center2_y = center2
    if np.abs(center2_x - center1_x) > vid_width * 0.2:
        return False
    if np.abs(center2_y - center1_y) > vid_height * 0.2:
        return False
    return True


def average_value_of_most_confident_face(bounding_boxes):
    """
    :param bounding_boxes:
    :return:
    """
    area_sizes = []
    for box_list in bounding_boxes:
        most_confident = box_list[0] if box_list is not None else None
        if most_confident is not None and most_confident['confidence'] > BASE_THRESHOLD:
            area_sizes.append(calc_area_of_box(most_confident['box']))
    return np.mean(area_sizes)


def calc_area_of_box(box):
    x1, y1, w, h = box
    area = w * h
    return area


def get_biggest_box(current_coordinates, last_coordinates):
    """
    :param current_coordinates: Coordiantes of the currently detected bounding box.
    :param last_coordinates: Coordiantes of the lastly detected bounding box.
    :return: The x1, x2, y1, y2 coordinates of the minimal box that contains both input-boxes.
    """
    x1, x2, y1, y2 = current_coordinates
    last_x1, last_x2, last_y1, last_y2 = last_coordinates
    x1 = min(x1, last_x1)
    x2 = max(x2, last_x2)
    y1 = min(y1, last_y1)
    y2 = max(y2, last_y2)
    return x1, x2, y1, y2


def crop_frame(file_path, bounding_boxes, write_y1, write_y2, write_x1, write_x2):
    """
    :param file_path: The path to the file.
    :param bounding_boxes: The bounding boxes
    :param write_y1: The y1 coordiante one wishes to crop by.
    :param write_y2: The y2 coordiante one wishes to crop by.
    :param write_x1: The x1 coordiante one wishes to crop by.
    :param write_x2: The x2 coordiante one wishes to crop by.
    :return: The list of cropped frames.
    """
    cropped_frames = []
    video_capture = cv2.VideoCapture(file_path)
    for i in bounding_boxes:
        _, color = video_capture.read()
        cropped = color[write_y1:write_y2, write_x1:write_x2]
        cropped_frames.append(cropped)
    return cropped_frames


def face_crop_multiple(directory_path, file_name, bounding_boxes, multi_face_mode, avg_most_conf, input_path,
                       detector):
    """
    :param directory_path: The base path of the working directory.
    :param file_name: The name of the file processed.
    :param bounding_boxes: List of bounding boxes detected.
    :param multi_face_mode: Number of faces detected in multi-face mode.
    :param avg_most_conf: Average size of the face detected most confidently.
    :param input_path: The path to the processed file.
    :param detector: The detector object (e.g. MediaPipe's FaceMesh).
    :return: None. Processed video is saved in the CROP_DIRECTORY under directory_path.
    """
    try:
        file_path = os.path.join(input_path, file_name)
        video_capture, vid_width, vid_height, length, fps = get_video_metadata(file_path)
        areas = faces_area(bounding_boxes, multi_face_mode)
        for num_face, area in enumerate(areas):
            output_video = os.path.join(directory_path, CROP_DIRECTORY,
                                        f"{file_name.split('.')[0]}@part{num_face + 1}.mp4")
            video_capture = cv2.VideoCapture(file_path)
            last_coordinates = modify_bounding_box(area['box'])
            last_area = None
            for f, box_list in enumerate(bounding_boxes):
                current_coordinates = last_coordinates
                if box_list is not None:
                    face = find_face_MTCNN(box_list, area, vid_width, vid_height, avg_most_conf)
                    if face:
                        current_coordinates = modify_bounding_box(face['box'])
                    else:
                        for face in box_list:
                            if face['confidence'] > BASE_THRESHOLD or (face['confidence'] > 0.95 and (
                                    avg_most_conf * 0.80 <= calc_area_of_box(face['box']) <= avg_most_conf * 1.20)):
                                center = centralize(face['box'])
                                if is_near_area(center['center'], area['center'], vid_width, vid_height):
                                    if last_area is not None and last_area * 0.6 <= calc_area_of_box(
                                            face['box']) <= last_area * 1.4:
                                        current_coordinates = modify_bounding_box(face['box'])
                                        last_area = calc_area_of_box(face['box'])
                                        break
                _, color = video_capture.read()
                write_x1, write_x2, write_y1, write_y2 = obtain_coordinates_for_cropping(current_coordinates,
                                                                                         last_coordinates, vid_width,
                                                                                         vid_height)
                last_coordinates = (write_x1, write_x2, write_y1, write_y2)
            video_capture.release()
            cropped_frames = crop_frame(file_path, bounding_boxes, write_y1, write_y2, write_x1, write_x2)
            media_pipe_preprocess_video(output_video, detector=detector, frames=cropped_frames, fps=fps)
            finish_crop(video_capture)
        return
    except Exception as e:
        logging.warning(f'Failed face crop due to {e}')


def faces_area(bounding_boxes, multi_face_mode):
    """
    :param bounding_boxes: List of bounding boxes
    :param multi_face_mode: The number of faces in the frame
    :return: List of bounding boxes with the center areas (the results of the centralize func)
    """
    for frame_faces in bounding_boxes:
        relevant_faces = []
        if frame_faces is not None:
            for face in frame_faces:
                if face['confidence'] > 0.99:
                    relevant_faces.append(centralize(face['box']))
        if len(relevant_faces) == multi_face_mode:
            return relevant_faces


def obtain_coordinates_for_cropping(current_coordinates, last_coordinates, vid_width, vid_height):
    """
    :param current_coordinates: Coordinates of the currently detected bounding box.
    :param last_coordinates: Coordinates of the lastly detected bounding box.
    :param vid_width: The width of the video.
    :param vid_height: The height of the video.
    :return: Final coordinates for cropping the frame, based on the minimal box that contains both last and current bounding boxes.
    """
    x1, x2, y1, y2 = current_coordinates
    current_coordinates = keep_in_frame(x1, x2, y1, y2, vid_width, vid_height, return_padding=False)
    x1, x2, y1, y2 = last_coordinates
    last_coordinates = keep_in_frame(x1, x2, y1, y2, vid_width, vid_height, return_padding=False)
    write_x1, write_x2, write_y1, write_y2 = get_biggest_box(current_coordinates, last_coordinates)
    return write_x1, write_x2, write_y1, write_y2


def centralize(box):
    """
    :param box: Bounding box in the format of x1, y1, width and height.
    :return: A dictionary including the bounding box and the center of the bounding box.
    """
    x1, y1, w, h = box
    center_x = int(x1 + w / 2)
    center_y = int(y1 + h / 2)
    return {'center': (center_x, center_y), 'box': box}


def finish_crop(video_capture):
    video_capture.release()
    return


def face_crop(directory_path, file_name, bounding_boxes, input_path, detector):
    """
    :param directory_path: The base path of the working directory.
    :param file_name: The name of the file processed.
    :param bounding_boxes: List of bounding boxes detected.
    :param input_path: The path to the processed file.
    :param detector: The detector object (e.g. MediaPipe's FaceMesh).
    :return: None. Processed video is saved in the CROP_DIRECTORY under directory_path.
    """
    try:
        if any(bounding_boxes):
            file_path = os.path.join(input_path, file_name)
            avg_most_conf = average_value_of_most_confident_face(bounding_boxes)
            multi_faces_mode = count_faces_in_video(bounding_boxes, avg_most_conf)
            if multi_faces_mode > 1:
                return face_crop_multiple(directory_path, file_name, bounding_boxes, multi_faces_mode,
                                          avg_most_conf, input_path, detector=detector)

            video_capture, vid_width, vid_height, length, fps = get_video_metadata(file_path)
            output_video = os.path.join(directory_path, CROP_DIRECTORY, f"{file_name}")
            non_empty_box = None
            i = 0
            while non_empty_box is None:
                non_empty_box = bounding_boxes[i]
                i += 1
            video_capture = cv2.VideoCapture(file_path)

            last_coordinates = modify_bounding_box(non_empty_box[0]['box'])
            last_area = calc_area_of_box(non_empty_box[0]['box'])
            for box_list in bounding_boxes:
                face = box_list[0] if box_list is not None else None
                if face is not None and face['confidence'] > 0.99:
                    if last_area * 0.6 <= calc_area_of_box(face['box']) <= last_area * 1.4:
                        current_coordinates = modify_bounding_box(face['box'])
                        last_area = calc_area_of_box(face['box'])
                else:
                    current_coordinates = last_coordinates
                write_x1, write_x2, write_y1, write_y2 = obtain_coordinates_for_cropping(current_coordinates,
                                                                                         last_coordinates, vid_width,
                                                                                         vid_height)
                last_coordinates = (write_x1, write_x2, write_y1, write_y2)

            video_capture.release()
            cropped_frames = crop_frame(file_path, bounding_boxes, write_y1, write_y2, write_x1, write_x2)
            media_pipe_preprocess_video(output_video, detector=detector, frames=cropped_frames, fps=fps)
            finish_crop(video_capture)
        return
    except Exception as e:
        logging.warning(f'Failed face crop due to {e}')


def run_face_detection_pipeline(directory_path, input_path=None):
    """
    :param directory_path: Processed data destination path.
    :param input_path: Input data path.
    """
    detector = FastMTCNN(
        stride=3,
        resize=1,
        margin=14,
        factor=0.6,
        keep_all=True,
        device=device
    )
    start_time = datetime.datetime.now()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=MAX_FACES, refine_landmarks=True, min_detection_confidence=0.3,
                                      min_tracking_confidence=0.8)

    input_path = directory_path if input_path is None else input_path
    all_files = [x for x in os.listdir(input_path) if '.mp4' in x]
    bounding_boxes = {}
    for file_name in tqdm(all_files):
        if file_name not in os.listdir(os.path.join(directory_path, CROP_DIRECTORY)):
            file_path = os.path.join(input_path, file_name)
            try:
                bounding_boxes[file_name] = run_detection(detector, file_path)
                face_crop(directory_path, file_name, bounding_boxes[file_name], input_path, detector=face_mesh)
            except Exception as e:
                logging.warning(f'Failed due to {e}')
    logging.warning('Face crop time:', datetime.datetime.now() - start_time)


def add_audio_pipeline(directory_path, input_path=None):
    """
    :param directory_path: Processed data destination path.
    :param input_path: Input data path.
    """
    input_path = directory_path if input_path is None else input_path
    for i, file_name in enumerate(os.listdir(input_path)):
        if '.mp4' in file_name:
            try:
                if file_name in os.listdir(os.path.join(directory_path, CROP_DIRECTORY)):
                    add_audio(directory_path, file_name, os.path.join(directory_path, CROP_DIRECTORY, file_name),
                              input_path)
                else:
                    for i in range(1, 3):
                        add_audio(directory_path, f"{file_name.split('.')[0]}@part{i}.mp4",
                                  os.path.join(directory_path, CROP_DIRECTORY,
                                               f"{file_name.split('.')[0]}@part{i}.mp4"), input_path)
            except Exception as e1:
                logging.warning(f'Could not add audio to {file_name}')
