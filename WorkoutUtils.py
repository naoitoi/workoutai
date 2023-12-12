import cv2
import numpy as np
import os
import streamlit as st

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

    @staticmethod
    def show_frames_side_by_side_sl(container, frame1, frame2, text = None):
        # Resize the frames to the same size
        if frame1 is None:
            frame1 = np.zeros((768, 768, 3), dtype=np.uint8)
        if frame2 is None:
            frame2 = np.zeros((768, 768, 3), dtype=np.uint8)
        frame1 = cv2.resize(frame1, (768, 768))
        frame2 = cv2.resize(frame2, (768, 768))

        # Combine the frames side by side
        combined = cv2.hconcat([frame1, frame2])

        if text is not None:
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
        else:
            final_image = combined

        # Display the final image
        # cv2.imshow('Videos with Text Box', final_image)
        container.image(final_image, channels="BGR")

    @staticmethod
    def show_gifs_side_by_side_sl(file_name1, file_name2):
        # Am I running on HuggingFace Spaces?
        isRunningOnHF = False
        if (os.environ.get('HOME') == '/home/user' and
            os.environ.get('PYTHONPATH') == '/home/user/app' and
            os.environ.get('PWD') is None):
            isRunningOnHF = True
            print('Running on HuggingFace Spaces')
        else:
            print('Not running on HuggingFace Spaces')

        media_prefix = 'https://huggingface.co/spaces/naomaru/workout_ai/resolve/main/static/' if isRunningOnHF else 'app/static/'
        gif_url_1 = media_prefix + file_name1
        gif_url_2 = media_prefix + file_name2

        print("Show GIFs in %s and %s" % (gif_url_1, gif_url_2))

        # Inject custom CSS to adjust the container width
        custom_css = """
        <style>
        .stApp {
            max-width: 2000px; /* Adjust the width as needed */
            margin: auto; /* Center the container */
        }
        </style>
        """

        st.markdown(custom_css, unsafe_allow_html=True)

        # Show GIFs
        markdown = ('<head> \
            <meta charset="UTF-8"> \
            <meta http-equiv="X-UA-Compatible" content="IE=edge"> \
            <meta name="viewport" content="width=device-width, initial-scale=1.0"> \
            <title>Autoplay and Loop GIF</title> \
        </head> \
            <div id="placeholderDiv" style="display: flex; flex-direction: row;"> \
                <img src="{}" alt="Image 1" width="512" height="512" autoplay loop> \
                <img src="{}" alt="Image 2" width="512" height="512" autoplay loop>'.format(gif_url_1, gif_url_2))

        st.markdown(markdown, unsafe_allow_html=True)
