import os
import time

import streamlit as st
import cv2
import json

from Person import PersonVideo
from WorkoutUtils import WorkoutUtils

# Define the video URLs
video_url_1 = "JoynerSprintSquare"
video_url_2 = "SakikoSprintSquare"
#video_url_3 = "TetsuSprintSquare"
image_url_1 = "JoynerRun.jpg"
image_url_2 = "SakikoRun.jpg"

set_h = False
set_v = False
set_q = False
uploaded_file = None

def show_videos(file_names):
    print ("show_videos(%s, %s)" % (file_names[0], file_names[1]))
    global set_h, set_v, set_q
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
        if set_q:
            set_q = False
            return
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
        text = None

        # Horizontal Max button was pressed
        if set_h == True:
            video1.set(cv2.CAP_PROP_POS_FRAMES, metadata1['mhs_frame'])
            _, frame1 = video1.read()
            hs_ratio1 = metadata1['hs_ratio']
            video2.set(cv2.CAP_PROP_POS_FRAMES, metadata2['mhs_frame'])
            _, frame2 = video2.read()
            hs_ratio2 = metadata2['hs_ratio']
            text = ("Stride / Leg Len: Left = %d%%, Right = %d%%" %
                    (round(hs_ratio1 * 100), round(hs_ratio2 * 100)))

        # Vertical Max button was pressed
        if set_v == True:
            video1.set(cv2.CAP_PROP_POS_FRAMES, metadata1['mvs_frame'])
            _, frame1 = video1.read()
            vs_ratio1 = metadata1['vs_ratio']
            video2.set(cv2.CAP_PROP_POS_FRAMES, metadata2['mvs_frame'])
            _, frame2 = video2.read()
            vs_ratio2 = metadata2['vs_ratio']
            text = ("Vertical Len / Leg Len: Left = %d%%, Right = %d%%" %
                    (round(vs_ratio1 * 100), round(vs_ratio2 * 100)))
        # Display the frames side by side
        WorkoutUtils.show_frames_side_by_side_sl(my_container, frame1, frame2, text)

        if set_h == True or set_v == True:
            set_h = False
            set_v = False
            pressedKey = cv2.waitKey(5000) & 0xFF

# Print Python version
# print("Python version:", platform.python_version())
# Print Streamlit version
# print("Streamlit version:", st.__version__)

# Create two button in columns
col1, col2 = st.columns(2)
button_h = col1.button("Horizontal Max")
button_v = col2.button("Vertical Max")
# button_q = st.button("Quit")

# File uploader button
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

uploaded_file = st.file_uploader("Choose a file", type=["mp4"],
                                 key=st.session_state["file_uploader_key"])


if uploaded_file is not None:
    uploaded_file_without_extension, file_extension = os.path.splitext(uploaded_file.name)
    st.session_state["uploaded_file"] = uploaded_file_without_extension

    set_q = True
    with st.empty():
        st.write(f"Processing Uploaded File ... Please give it a few minutes.")
        #st.write(f"Filename: {uploaded_file.name}")
        #st.write(f"File Type: {uploaded_file.type}")
        #st.write(f"File Size: {uploaded_file.size} bytes")
        pv = PersonVideo(uploaded_file.name)

        uploaded_file = None
        print("Will draw_keypoints()")
        pv.draw_keypoints()
        print("Will save_video()")
        pv.save_video()
        st.write("")
        # Process the uploaded file as needed (e.g., read contents, analyze data)
        # For example, you can use Pandas to read a CSV file:
        # import pandas as pd
        # df = pd.read_csv(uploaded_file)
        # st.dataframe(df)
    print("Will call show_videos()")
        # video_url_2 =  video_url_3
    st.session_state["file_uploader_key"] += 1
        # show_videos([video_url_1, video_url_3])
    st.rerun()

# Check if the button is clicked
if button_h:
    set_h = True
if button_v:
    set_v = True
#if button_q:
#    set_q = True

def main():
    print ("file_uploader_key %d" % st.session_state["file_uploader_key"])
    if st.session_state["uploaded_file"] is not None:
        show_videos([video_url_1, st.session_state["uploaded_file"]])
    else:
        show_videos([video_url_1, video_url_2])

if __name__ == "__main__":
    # If the script is run directly, call the main function
    main()