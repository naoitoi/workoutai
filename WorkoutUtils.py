import cv2

class WorkoutUtils:
    # Dictionary that maps from joint names to keypoint indices.

    KEYPOINT_DICT = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }

    @staticmethod
    def draw_line_between_keypoints(frame, start, end, color, thickness=3):
        cv2.line(frame, start, end, color, thickness)

    @staticmethod
    def show_frames_side_by_side(frame1, frame2):
        # Resize the frames to the same size
        frame1 = cv2.resize(frame1, (768, 768))
        frame2 = cv2.resize(frame2, (768, 768))

        # Combine the frames side by side
        combined = cv2.hconcat([frame1, frame2])

        # Display the combined frames
        cv2.imshow('Videos', combined)