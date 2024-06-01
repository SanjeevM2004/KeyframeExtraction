import numpy as np
import json
import cv2
from keras.utils.data_utils import get_file
from keras import backend as K
import glob as gb
import os
from sklearn.cluster import KMeans

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    x = x.astype(float)
    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        x = x[:, :, :, ::-1]
    return x

def decode_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 512:
        raise ValueError('`decode_predictions` expects a batch of predictions (i.e. a 2D array of shape (samples, 1000)). Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json', CLASS_INDEX_PATH, cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(CLASS_INDEX[str(i)][1], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results

def extract_keyframes(video_path, sampling_rate=30):
    capture = cv2.VideoCapture(video_path)
    frames = []
    k = 0
    while(capture.isOpened()):
        if k % sampling_rate == 0:
            capture.set(1, k)
            ret, frame = capture.read()
            if frame is None:
                break
            frames.append(np.asarray(frame))
        k += 1
    capture.release()
    return np.array(frames)

def main(video_path):
    frames = extract_keyframes(video_path)
    print("Extracted frames from the video")
    
    # Assuming we have a feature extraction function `extract_features`
    features = extract_features(frames)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(features)
    
    frame_indices = []
    for center in kmeans.cluster_centers_:
        distances_to_center = np.linalg.norm(features - center, axis=1)
        closest_point_index = np.argmin(distances_to_center)
        frame_indices.append(closest_point_index)
    
    frame_indices = sorted(frame_indices)
    selected_frames = [frames[i] for i in frame_indices]
    
    return selected_frames

def extract_features(frames):
    # Dummy feature extraction, replace with actual feature extraction code
    return np.random.rand(len(frames), 512)

if __name__ == '__main__':
    video_path = 'path_to_your_video.mp4'
    keyframes = main(video_path)
    print("Selected keyframes indices:", keyframes)
