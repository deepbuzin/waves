import cv2
import numpy as np
import mediapipe as mp
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

_landmark_names = ['nose',
                   'left_eye_inner', 'left_eye', 'left_eye_outer',
                   'right_eye_inner', 'right_eye', 'right_eye_outer',
                   'left_ear', 'right_ear',
                   'mouth_left', 'mouth_right',
                   'left_shoulder', 'right_shoulder',
                   'left_elbow', 'right_elbow',
                   'left_wrist', 'right_wrist',
                   'left_pinky_1', 'right_pinky_1',
                   'left_index_1', 'right_index_1',
                   'left_thumb_2', 'right_thumb_2',
                   'left_hip', 'right_hip',
                   'left_knee', 'right_knee',
                   'left_ankle', 'right_ankle',
                   'left_heel', 'right_heel',
                   'left_foot_index', 'right_foot_index']


class WaveCounter:
    def __init__(self, wrist_idx, shoulder_idx, inner_thresh, outer_thresh):
        self.wrist_idx = wrist_idx
        self.shoulder_idx = shoulder_idx
        self.inner_thresh = inner_thresh
        self.outer_thresh = outer_thresh
        self.wave_len = 2 * (outer_thresh - inner_thresh)
        self.passed_outermost = False
        self.reps = 0

    def parse_landmarks(self, landmarks):
        """
        The function from the assignment

        landmarks: Pose Estimation values outputted from MediaPipe Pose
        reps: The number of times a ‘wave’ has been completed
        percentage: The % completion of the current ‘wave’ action
        """
        wrist = np.asarray([landmarks[self.wrist_idx].x,  landmarks[self.wrist_idx].y])
        shoulder = np.asarray([landmarks[self.shoulder_idx].x, landmarks[self.shoulder_idx].y])
        d = np.sqrt(np.sum(np.square(wrist - shoulder)))

        # Compare distance to terminal point thresholds
        if d < self.inner_thresh:
            if self.passed_outermost:
                self.reps += 1
            self.passed_outermost = False
            percentage = 0.

        elif d > self.outer_thresh:
            self.passed_outermost = True
            percentage = 0.5

        elif self.inner_thresh <= d <= self.outer_thresh and not self.passed_outermost:
            percentage = (d - self.inner_thresh) / self.wave_len

        elif self.inner_thresh <= d <= self.outer_thresh and self.passed_outermost:
            percentage = 1. - (d - self.inner_thresh) / self.wave_len

        return self.reps, percentage


def extract_waves(video_name, wrist_idx, shoulder_idx, outer_thresh=0.07, inner_thresh=0.03):
    wave_counter = WaveCounter(wrist_idx, shoulder_idx, inner_thresh, outer_thresh)

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
            reps, percentage = wave_counter.parse_landmarks(results.pose_landmarks.landmark)
            print(f'{reps} reps, {int(percentage * 100)}%')

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count wave reps')
    parser.add_argument('video_name', type=str)
    args = parser.parse_args()
    _wrist_idx = _landmark_names.index('right_wrist')
    _shoulder_idx = _landmark_names.index('right_shoulder')
    extract_waves(args.video_name, _wrist_idx, _shoulder_idx)




