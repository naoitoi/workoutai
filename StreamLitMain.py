import streamlit as st
import time
import cv2
import json

from WorkoutUtils import WorkoutUtils

# Define the video URLs
video_url_1 = "JoynerSprintSquare"
video_url_2 = "SakikoSprintSquare"
image_url_1 = "JoynerRun.jpg"
image_url_2 = "SakikoRun.jpg"

def show_videos(file_names):
    my_container = st.empty()
    # Read the first video and metadata
    video1 = cv2.VideoCapture(file_names[0] + '-out.mp4')
    metadata1 = json.load(open(file_names[0] + '-meta.json'))

    # Read the second video and metadata
    video2 = cv2.VideoCapture(file_names[1] + '-out.mp4')
    metadata2 = json.load(open(file_names[1] + '-meta.json'))

    # Create a window to display the videos

    i = 0
    while True:
        # Capture frames from each video
        # Video 1 should be slowed down
        if i == 0:
            ret1, frame1 = video1.read()
            if not ret1:
                video1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret1, frame1 = video1.read()
            i = 1
        else:
            i = 0

        ret2, frame2 = video2.read()
        if not ret2:
            video2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = video2.read()

        # Display the frames side by side
        WorkoutUtils.show_frames_side_by_side_sl(my_container, frame1, frame2)

show_videos([video_url_1, video_url_2])