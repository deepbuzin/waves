import cv2
import numpy as np
import mediapipe as mp

from rep_counter import _landmark_names

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calc_extrema(video_name, wrist_idx, shoulder_idx):
    dists = []

    cap = cv2.VideoCapture(video_name)
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print('End of video')
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Extract wrist to shoulder distance
            wrist = np.asarray([results.pose_landmarks.landmark[wrist_idx].x,
                                results.pose_landmarks.landmark[wrist_idx].y])

            shoulder = np.asarray([results.pose_landmarks.landmark[shoulder_idx].x,
                                   results.pose_landmarks.landmark[shoulder_idx].y])

            d = np.sqrt(np.sum(np.square(wrist - shoulder)))
            dists.append(d)
    cap.release()

    a = np.asarray(dists)

    # smoothen distances
    window_len = 3
    s = np.r_[a[window_len - 1:0:-1], a, a[-2:-window_len - 1:-1]]
    w = np.ones(window_len, 'd')
    a = np.convolve(w / w.sum(), s, mode='valid')

    # Calculate local minima and local maxima
    minima = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
    maxima = np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True]
    print(a[minima])
    print(a[maxima])


if __name__ == '__main__':
    _wrist_idx = _landmark_names.index('right_wrist')
    _shoulder_idx = _landmark_names.index('right_shoulder')
    # _wrist_idx = _landmark_names.index('left_wrist')
    # _shoulder_idx = _landmark_names.index('left_shoulder')

    _video_name = 'A.mp4'
    calc_extrema(_video_name, _wrist_idx, _shoulder_idx)

