import cv2
import numpy as np

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
    def show_frames_side_by_side(frame1, frame2, text = None):
        # Resize the frames to the same size
        frame1 = cv2.resize(frame1, (768, 768))
        frame2 = cv2.resize(frame2, (768, 768))

        # Combine the frames side by side
        combined = cv2.hconcat([frame1, frame2])

        # Add text at the bottom of the combined image
        if text is None:
            text = "Press H to show Horizontal Max.  V for Vertical Max.  Q to quit."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 255, 255)  # White color

        # Get the size of the text box
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # Create a box with text
        text_box = 255 * np.ones((50, combined.shape[1], 3), dtype=np.uint8)  # White box with height 50
        cv2.putText(text_box, text, (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Put text on the box

        # Vertically concatenate the combined images and the text box
        final_image = cv2.vconcat([combined, text_box])

        # Display the final image
        cv2.imshow('Videos with Text Box', final_image)