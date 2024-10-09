
import pandas as pd
from mtcnn.mtcnn import MTCNN
from fer import FER
import cv2
# from models.gaze_tracking.gaze_tracking import GazeTracking


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class FastStatistics:
    def __init__(self, use_cv2, process_statistic_every=30, frames_per_second=30):
        self.use_cv2 = use_cv2
        self.detector = MTCNN()
        self.cv2_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.data = []
        # self.gaze = GazeTracking()
        self.emotion_detector = FER()
        self.process_every = process_statistic_every
        self.frames_per_second = frames_per_second
        self.frame_id = 0

    def detect_emotions(self, face_img, box):
        _emotions = []
        try:
            _emotions = self.emotion_detector.detect_emotions(face_img, [box])
            if len(_emotions) == 0 or 'emotions' not in _emotions[0]:
                return {}
            emotions = {}
            for k, v in _emotions[0]['emotions'].items():
                emotions['emotion_' + k] = v
            return emotions
        except Exception as e:
            print(f'Error emotions {_emotions}', e)
            return {}

    def detect_gaze(self, face_img):
        try:
            import random
            is_center = random.choice([True, False])
            # self.gaze.refresh(face_img)
            # is_center = self.gaze.is_center()
            return is_center
        except Exception as e:
            print('Error gaze', e)
            return None

    def process_face(self, frame, box, detect_emotion_gaze=False):
        try:
            x, y, w, h = box
            face_img = frame[y:y+h, x:x+w]
            emotions = self.detect_emotions(frame, box) if detect_emotion_gaze else {}
            is_looking = self.detect_gaze(face_img) if detect_emotion_gaze else None
            return {
                'frame_id': self.frame_id,
                'second': self.frame_id // self.frames_per_second,
                'bbox': (x, y, w, h),
                'is_calculated': detect_emotion_gaze,
                'is_looking': is_looking,
                **emotions
            }
        except Exception as e:
            print('Error processing face', e)
            return None

    def detect_faces(self, frame):
        if self.use_cv2:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cv2_detector.detectMultiScale(
                frame_gray,
                scaleFactor=1.1,
                minNeighbors=5,
            )
            return [{'box': (x, y, w, h)} for (x, y, w, h) in faces]
        else:
            return self.detector.detect_faces(frame)

    def calculate_statistics(self, frame, detect_emotion_gaze=False):
        detections = self.detect_faces(frame)
        data = [self.process_face(frame, detection['box'], detect_emotion_gaze) for detection in detections]
        data = [d for d in data if d is not None]
        return data

    def update(self, frame):
        self.frame_id += 1
        detect_emotion_gaze = True
        data = self.calculate_statistics(frame, detect_emotion_gaze)
        if len(data) > 0:
            self.data += data
        return data

    def get_statistics(self):
        if len(self.data) == 0:
            print("non data")
            return None
        data = pd.DataFrame(self.data)
        if len(data) == 0:
            print("non data by dataframe")
            return None
        if 'is_looking' not in data.columns:
            print("non data by is_looking")
            return None
        valid_data = data[data.is_calculated]
        if len(valid_data) == 0:
            print("non data by valid_data")
            return None
        return valid_data


if __name__ == '__main__':
    import os

    dataset_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'dataset'))

    img1 = cv2.imread(os.path.join(dataset_directory, 'berkeley_faces.jpg'))
    img2 = cv2.imread(os.path.join(dataset_directory, 'PAFF_052114_firstimpressionfaces_newsfeature1.jpg'))

    face_tracker = FastStatistics(2, 2)
    face_tracker.update(img1)
    face_tracker.update(img2)



