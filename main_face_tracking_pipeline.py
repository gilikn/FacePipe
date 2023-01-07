import logging
import subprocess

from absl import app
from absl import flags

from facepipe.config.pre_processing_config import CROP_DIRECTORY, AUDIO_DIRECTORY, MEAN_FACE_DOWNLOAD_URL
from facepipe.core.run_face_detection import run_face_detection_pipeline, add_audio_pipeline

flags.DEFINE_string('base_directory_path', None,
                    'The path to the directory where original and results directories are held.')
flags.DEFINE_string('cropped_directory_name', CROP_DIRECTORY,
                    'Name of directory in which cropped (silent) videos will be saved (under base_directory_path).')
flags.DEFINE_string('audio_directory_name', AUDIO_DIRECTORY,
                    'Name of directory in which cropped videos will be saved with audio (under base_directory_path).')
flags.DEFINE_string('mean_face_path', None, f"reference mean face (download from {MEAN_FACE_DOWNLOAD_URL})")
FLAGS = flags.FLAGS


def main(argv):
    del argv
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    assert FLAGS.mean_face_path is not None, f"Please enter the path to the mean_face file! Download from {MEAN_FACE_DOWNLOAD_URL}"
    assert FLAGS.base_directory_path is not None, "Please enter the path to the videos!"
    directory_path = FLAGS.base_directory_path
    videos_dir_path = directory_path
    subprocess.call(f'mkdir {FLAGS.cropped_directory_name}', cwd=directory_path, shell=True)
    subprocess.call(f'mkdir {FLAGS.audio_directory_name}', cwd=directory_path, shell=True)
    run_face_detection_pipeline(directory_path, FLAGS.mean_face_path, videos_dir_path)
    add_audio_pipeline(directory_path, videos_dir_path)


if __name__ == '__main__':
    app.run(main)
