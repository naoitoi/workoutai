import cv2
import tensorflow as tf

from Person import PersonVideo
from Person import PersonFrame

def analyze_frame(frame):

    height, width, _ = frame.shape

    # Crop the frame to a square
    # Calculate the starting point for the crop
    start_y = (height - 1080) // 2
    start_x = (width - 1080) // 2
    # Perform the crop
    cropped_image = frame[start_y:start_y + 1080, start_x:start_x + 1080, :]
    cv2.imshow('Video Frame', cropped_image)

    # Resize the frame to 256x256
    tf_image = tf.convert_to_tensor(cropped_image, dtype=tf.float32)
    tf_image = tf.expand_dims(tf_image, axis=0)
    tf_image = tf.cast(tf.image.resize_with_pad(tf_image, 256, 256), dtype=tf.int32)

    # Run model inference

    # Plot the keypoints on the frame

    cv2.imshow('Video Frame', cropped_image)

    return cropped_image
def main():

    personVideos = []
    for filename in ['JoynerSprintVideo.mp4', 'SakikoSprint.mp4', 'NaoSprint.mp4']:
        pv = PersonVideo(filename)
        personVideos.append(pv)
        pv.analyze()
        pv.show()


#        cap = cv2.VideoCapture(filename)
        # Loop through frames of a video
#        while cap.isOpened():
#            ret, frame = cap.read()
#            if not ret:
#                break
#            frame = analyze_frame(frame)

#            if cv2.waitKey(1) & 0xFF == ord('q'):
#                break

#    cv2.waitKey(1000)
#    cap.release()
#    cv2.destroyAllWindows()

# This block checks if the script is being run directly
if __name__ == "__main__":
    # If the script is run directly, call the main function
    main()