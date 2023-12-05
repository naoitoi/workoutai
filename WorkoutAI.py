import json
import sys

import cv2
import tensorflow as tf

from Person import PersonVideo
from WorkoutUtils import WorkoutUtils


def show_videos(file_names):
    # Read the first video and metadata
    video1 = cv2.VideoCapture(file_names[0] + '-out.mp4')
    metadata1 = json.load(open(file_names[0] + '-meta.json'))

    # Read the second video and metadata
    video2 = cv2.VideoCapture(file_names[1] + '-out.mp4')
    metadata2 = json.load(open(file_names[1] + '-meta.json'))

    # Create a window to display the videos
    window = cv2.namedWindow('Videos', cv2.WINDOW_NORMAL)

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
        WorkoutUtils.show_frames_side_by_side(frame1, frame2)

        # Check if the user wants to quit
        pressedKey = cv2.waitKey(1) & 0xFF
        if chr(pressedKey).lower() == 'q':
            break
        elif chr(pressedKey).lower() == 'h':
            frame_index1 = metadata1['mhs_frame']
            video1.set(cv2.CAP_PROP_POS_FRAMES, frame_index1)
            ret, frame1 = video1.read()
            hs_ratio1 = metadata1['hs_ratio']

            frame_index2 = metadata2['mhs_frame']
            video2.set(cv2.CAP_PROP_POS_FRAMES, frame_index2)
            ret, frame2 = video2.read()
            hs_ratio2 = metadata2['hs_ratio']

            text = ("Stride / Leg Len: Left = %d%%, Right = %d%%" %
                    (round(hs_ratio1 * 100), round(hs_ratio2 * 100)))
            WorkoutUtils.show_frames_side_by_side(frame1, frame2, text)
            cv2.waitKey((5000))
        elif chr(pressedKey).lower() == 'v':
            frame_index1 = metadata1['mvs_frame']
            video1.set(cv2.CAP_PROP_POS_FRAMES, frame_index1)
            ret, frame1 = video1.read()
            vs_ratio1 = metadata1['vs_ratio']

            frame_index2 = metadata2['mvs_frame']
            video2.set(cv2.CAP_PROP_POS_FRAMES, frame_index2)
            ret, frame2 = video2.read()
            vs_ratio2 = metadata2['vs_ratio']

            text = ("Stride / Leg Len: Left = %d%%, Right = %d%%" %
                    (round(vs_ratio1 * 100), round(vs_ratio2 * 100)))

            WorkoutUtils.show_frames_side_by_side(frame1, frame2, text)
            cv2.waitKey((5000))

    # Close the videos
    video1.release()
    video2.release()

    # Close the window
    cv2.destroyAllWindows()

def main():
    num_args = len(sys.argv)
    # Get the arguments from the command line
    if (num_args == 1):
        print ("Load and analyze videos")
        load_and_analyze_videos()
        return

    if sys.argv[1] == 'play':
        file_names = sys.argv[2:]
        print ("Show videos: " + str(sys.argv[2:]))
        show_videos(file_names)
        return
    else:
        print("WorkoutAI.py <filename> [<filename> ...])")  # Analyze and show the video
        print("Usage: python3.11 WorkoutAI.py play <filename> [<filename> ...]")  # Play existing videos

def load_and_analyze_videos():
    personVideos = []
    for filename, slow_down_factor in [('JoynerSprintSquare.mp4', 2),
                                       ('SakikoSprintSquare.mp4', 1),
                                       ('NaoSprintSquare.mp4', 1)]:
        pv = PersonVideo(filename)
        personVideos.append(pv)
        pv.draw_keypoints()
        pv.save_video(slow_down_factor)
        #pv.analyze()
        if pv.show() == False:
            break

# This block checks if the script is being run directly
if __name__ == "__main__":
    # If the script is run directly, call the main function
    main()