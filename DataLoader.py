import glob
import os

from sklearn.utils import shuffle
import numpy as np
import cv2
import matplotlib.pyplot as plt

from moviepy import editor


class DataLoader:
    def __init__(self):
        self.input_size = 128
        self.num_images = 10
        with open('videos_paths.txt', 'r') as file:
            self.videos = file.read().split('\n')

    def _process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (self.input_size, self.input_size))
        return frame.reshape((self.input_size, self.input_size, 1)).astype('float32') / 255.

    def _get_n_random_frames(self, video):
        video = editor.VideoFileClip(video)
        duration = video.duration - 0.5
        pts = np.random.uniform(0, 1, size=(self.num_images, ))
        return np.moveaxis(np.array([self._process_frame(video.get_frame(duration * s)) for s in pts]), 0, -2)

    def load_batch(self, batch_size):
        video_paths = np.random.choice(self.videos, batch_size, replace=False)
        assert batch_size % 2 == 0
        batch_size = batch_size // 2
        same = [[self._get_n_random_frames(vid), self._get_n_random_frames(vid)] for vid in video_paths[:batch_size]]
        video_paths = np.append(video_paths, video_paths[0])
        diff = [[self._get_n_random_frames(video_paths[n]), self._get_n_random_frames(video_paths[n + 1])] for n in range(batch_size, batch_size * 2, 1)]
        labels = [1] * batch_size + [0] * batch_size

        X = same + diff
        y, X = shuffle(labels, X)
        return np.array(X), np.array(y)


# d = DataLoader()
# d.load_batch(4)
