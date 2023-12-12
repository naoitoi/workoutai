import json

import cv2
import os

import imageio
import tensorflow_hub as hub
import tensorflow as tf
from WorkoutUtils import WorkoutUtils

# model = YOLO('yolov8n-pose.pt')
model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-thunder/versions/4")
movenet = model.signatures['serving_default']

class PersonFrame:

    # Constructor
    def __init__(self, frame):
        self.frame = frame
        self.keypoints = []

    # Show the frame
    def show(self):
        cv2.imshow('Video Frame', self.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    # Draw the person's keypoints on the frame
    # Returns a dictionary including:
    # hs: horizontal stride
    # vs: virtical stride
    # leg_len: length of the leg
    def draw_keypoints(self):
        height, width, _ = self.frame.shape

        # Crop the frame to a square
        # Assuming width > height
        start_y = 0
        start_x = width // 2 - (height // 2)

        # Perform the crop
        cropped_image = self.frame[start_y:start_y + height - 1, start_x:start_x + height, :]

        # Resize the frame to 256x256 so that MoveNet can process it
        tf_image = tf.convert_to_tensor(cropped_image, dtype=tf.float32)
        tf_image = tf.expand_dims(tf_image, axis=0)
        #tf_image = tf.cast(tf.image.resize_with_pad(tf_image, 256, 256), dtype=tf.int32)
        tf_image = tf.cast(tf.image.resize_with_pad(tf_image, 256, 256), dtype=tf.int32)

        # Run model inference
        outputs = movenet(tf_image)
        keypoints = outputs['output_0'].numpy()

        # Plot the keypoints on the frame

        image_size = cropped_image.shape[:2]

        # Define the color for the red dot in BGR format (OpenCV uses BGR instead of RGB)
        dot_color = (0, 0, 255)  # (B, G, R)

        key_point_coordinates = []
        for i in range(17):
            dot_coordinates = (int(keypoints[0][0][i][1] * image_size[1]), int(keypoints[0][0][i][0] * image_size[0]))
            key_point_coordinates.append(dot_coordinates)

        for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
             cv2.circle(cropped_image, key_point_coordinates[i], radius=5, color=dot_color, thickness=-1)  # -1 fills the circle
        # Define the coordinates for the red dot (assuming you want it at (x, y) = (100, 100))

        r_ankle = WorkoutUtils.KEYPOINT_DICT['right_ankle']
        l_ankle = WorkoutUtils.KEYPOINT_DICT['left_ankle']
        r_knee = WorkoutUtils.KEYPOINT_DICT['right_knee']
        l_knee = WorkoutUtils.KEYPOINT_DICT['left_knee']
        r_hip = WorkoutUtils.KEYPOINT_DICT['right_hip']
        l_hip = WorkoutUtils.KEYPOINT_DICT['left_hip']

        # Draw yellow line between the dots
        # Define the color for the red dot in BGR format (OpenCV uses BGR instead of RGB)
        line_color = (0, 255, 255)  # (B, G, R)
        pair = [13, 15]
        cv2.line(cropped_image,
                 (int(keypoints[0][0][pair[0]][1] * image_size[1]),
                      int(keypoints[0][0][pair[0]][0] * image_size[0])),
                 (int(keypoints[0][0][pair[1]][1] * image_size[1]),
                 int(keypoints[0][0][pair[1]][0] * image_size[0])),
                 line_color,3)

        # orange color
        line_color = (0, 165, 255)
        pair = [14, 16]
        cv2.line(cropped_image,
                 (int(keypoints[0][0][pair[0]][1] * image_size[1]),
                      int(keypoints[0][0][pair[0]][0] * image_size[0])),
                 (int(keypoints[0][0][pair[1]][1] * image_size[1]),
                 int(keypoints[0][0][pair[1]][0] * image_size[0])),
                 line_color,3)

        #horizontal_stride = abs(key_point_coordinates[r_ankle][0] - key_point_coordinates[l_ankle][0])
        l_horizontal_stride = key_point_coordinates[l_ankle][0] - key_point_coordinates[l_hip][0]
        r_horizontal_stride = key_point_coordinates[r_ankle][0] - key_point_coordinates[r_hip][0]
        r_leg_len = (cv2.norm(key_point_coordinates[r_hip], key_point_coordinates[r_knee]) +
                     cv2.norm(key_point_coordinates[r_knee], key_point_coordinates[r_ankle]))
        l_leg_len = (cv2.norm(key_point_coordinates[l_hip], key_point_coordinates[l_knee]) +
                     cv2.norm(key_point_coordinates[l_knee], key_point_coordinates[l_ankle]))

        if r_horizontal_stride > l_horizontal_stride:
            horizontal_stride = r_horizontal_stride
            leg_len = r_leg_len
        else:
            horizontal_stride = l_horizontal_stride
            leg_len = l_leg_len

        virtical_stride = abs(key_point_coordinates[r_ankle][1] - key_point_coordinates[l_ankle][1])
        # print ("horizontal_stride: %d, virtical_stride: %d" % (horizontal_stride, virtical_stride))

        # Draw a red dot on the image
        self.frame = cropped_image
        #cv2.imshow('Video Frame', self.frame)
        #cv2.waitKey(1)

        return {
            'hs': horizontal_stride,
            'vs': virtical_stride,
            'leg_len': leg_len
        }

    # Find region of interest (where the human is)
    # Use the square size to crop the frame
    def crop_human(self, square_size):
        # Find the person in the image
        height, width, _ = self.frame.shape

        crop_results = model(self.frame)
        for result in crop_results:
            boxes = result.boxes  # Boxes object for bbox outputs
            center_x, center_y = boxes[0].xywh[0][0:2]
            y1 = center_y - square_size/2
            y2 = center_y + square_size / 2
            if y1 < 0:
                margin = -y1
                y1 = 0
                y2 += margin

            x1 = center_x - square_size / 2
            x2 = center_x + square_size / 2
            if x1 < 0:
                margin = -x1
                x1 = 0
                x2 += margin

            self.frame = self.frame[int(y1):int(y2), int(x1):int(x2), :]
            break

        cv2.imshow('Video Frame', self.frame)
        cv2.waitKey(1)

class PersonVideo:

    # Constructor.  If filename is provided, load the video.
    def __init__(self, filename = None):
        self.keypoints = []
        self.frames = []
        self.np_frames = []
        self.metadata = {}

        if filename is not None:
            self.filename = filename
            self.load_video(filename)
            print ("Loaded video %s (cap %s)" % (self.filename, self.cap))
        filename_split = os.path.splitext(self.filename)[0]
        self.outfilename = filename_split + '-out.mp4'
        self.outgifname = filename_split + '-out.gif'
        self.out_mh_gif_name = filename_split + '-out-mh.gif'
        self.out_mv_gif_name = filename_split + '-out-mv.gif'
        self.metafilename = filename_split + '-meta.json'

    # Load a video.  If filename is provided, load that video.  Otherwise, use the filename provided in the constructor
    def load_video(self, filename = None):
        if filename is not None:
            self.filename = filename
        if filename is None:
            print("No filename provided")
            raise ValueError("No filename provided")

        self.cap = cv2.VideoCapture(filename)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            pf = PersonFrame(frame)
            self.frames.append(pf)
            # self.np_frames.append(cv2.cvtColor(pf.frame, cv2.COLOR_BGR2RGB))

    def save_video(self, slow_down_factor = 1):
        height, width, _ = self.frames[0].frame.shape

        # Save the metadata
        print ("Saving metadata: %s" % (self.metafilename))
        with open(self.metafilename, 'w') as json_file:
            json.dump(self.metadata, json_file)

        # Save the annotated video
        print ("Saving: %s (fps: %d)" % (self.outfilename, self.cap.get(cv2.CAP_PROP_FPS)))

        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(self.outfilename, fourcc,
                              self.cap.get(cv2.CAP_PROP_FPS) // slow_down_factor,
                              (width, height))

        for frame in self.frames:
            # Write the frame to the output file
            out.write(frame.frame)
        # Release everything if job is finished
        out.release()

    def save_gif(self, slow_down_factor = 1):
        height, width, _ = self.frames[0].frame.shape

        # Save the metadata
        print ("Saving metadata: %s" % (self.metafilename))
        with open(self.metafilename, 'w') as json_file:
            json.dump(self.metadata, json_file)

        fps = self.cap.get(cv2.CAP_PROP_FPS) // slow_down_factor
        # Save the annotated video
        print ("Saving: %s (fps: %d)" % (self.outgifname, self.cap.get(cv2.CAP_PROP_FPS)))
        # Write the frames to a GIF file
        imageio.mimsave(self.outgifname,
                        [cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB) for frame in self.frames],
                        fps=fps, compress='lossless', loop=65535)

        # Write Max Horizontal GIF
        frame_index = self.metadata['mhs_frame']
        imageio.imwrite(self.out_mh_gif_name,
                        cv2.cvtColor(self.frames[frame_index].frame, cv2.COLOR_BGR2RGB),
                        compress='lossless')

        # Write Max Virtical GIF
        frame_index = self.metadata['mvs_frame']
        imageio.imwrite(self.out_mv_gif_name,
                        cv2.cvtColor(self.frames[frame_index].frame, cv2.COLOR_BGR2RGB),
                        compress='lossless')

    def draw_keypoints(self):
        max_horizontal_stride = 0
        mhs_frame = None
        max_virtical_stride = 0
        mvs_frame = None

        i = 0 # Assume frames are numbered from 0 and increase by 1
        for frame in self.frames:
            dict = frame.draw_keypoints()
            horizontal_stride = dict['hs']
            virtical_stride = dict['vs']
            leg_len = dict['leg_len']

            if horizontal_stride > max_horizontal_stride:
                max_horizontal_stride = horizontal_stride
                hs_ratio = horizontal_stride / leg_len
                mhs_frame = i
            if virtical_stride > max_virtical_stride:
                max_virtical_stride = virtical_stride
                vs_ratio = virtical_stride / leg_len
                mvs_frame = i
            i = i + 1

        print ("Max horizontal stride: %d, ratio %f (frame %d)" % (max_horizontal_stride, hs_ratio, mhs_frame))
        print ("Max virtical stride: %d, ratio %f (frame %d)" % (max_virtical_stride, vs_ratio, mvs_frame))

        self.metadata = {
            'max_horizontal_stride': max_horizontal_stride,
            'hs_ratio': hs_ratio,
            'mhs_frame': mhs_frame,
            'max_virtical_stride': max_virtical_stride,
            'vs_ratio': vs_ratio,
            'mvs_frame': mvs_frame
        }

    def show(self):
        for frame in self.frames:
            rv = frame.show()
            if rv == False:
                return False
        return True
