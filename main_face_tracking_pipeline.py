import logging
import os
import subprocess

from absl import app
from absl import flags

from facepipe.config.pre_processing_config import CROP_DIRECTORY, AUDIO_DIRECTORY, ORIGINAL_DIRECTORY
from facepipe.core.run_face_detection import run_face_detection_pipeline, add_audio_pipeline

flags.DEFINE_string('base_directory_path', '', 'The directory where original and tracked directories are held.')
flags.DEFINE_string('original_directory_name', ORIGINAL_DIRECTORY,
                    'Name of directory in which original videos are held.')
flags.DEFINE_string('cropped_directory_name', CROP_DIRECTORY,
                    'Name of directory in which cropped (silent) videos will be saved.')
flags.DEFINE_string('audio_directory_name', AUDIO_DIRECTORY,
                    'Name of directory in which cropped videos will be saved with audio.')
flags.DEFINE_string('mean_face_path', '', 'reference mean face (download from: '
                                     'https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/20words_mean_face.npy)')

FLAGS = flags.FLAGS


def main(argv):
    del argv
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    directory_path = FLAGS.base_directory_path
    videos_dir_path = os.path.join(directory_path, ORIGINAL_DIRECTORY)
    subprocess.call(f'mkdir {FLAGS.cropped_directory_name}', cwd=directory_path, shell=True)
    subprocess.call(f'mkdir {FLAGS.audio_directory_name}', cwd=directory_path, shell=True)
    run_face_detection_pipeline(directory_path, FLAGS.mean_face_path, videos_dir_path)
    add_audio_pipeline(directory_path, videos_dir_path)


if __name__ == '__main__':
    app.run(main)
