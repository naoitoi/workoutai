import cv2
import os
import tensorflow_hub as hub
import tensorflow as tf

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

    # Find the smallest square that fits the person in the frame.
    # Return the height and width of the square
    def find_smallest_square(self):
        # Find the person in the image
        height, width, _ = self.frame.shape

        crop_results = model(self.frame)
        for result in crop_results:
            boxes = result.boxes  # Boxes object for bbox outputs
            if boxes is not None and boxes.cls[0] == 0:
                crop_x1 = round(boxes.data[0][0].item())
                crop_x2 = round(boxes.data[0][2].item())
                crop_y1 = round(boxes.data[0][1].item())
                crop_y2 = round(boxes.data[0][3].item())

                return (crop_y2 - crop_y1, crop_x2 - crop_x1)


    # Draw the person's keypoints on the frame
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

        for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
             dot_coordinates = (int(keypoints[0][0][i][1] * image_size[1]), int(keypoints[0][0][i][0] * image_size[0]))
             cv2.circle(cropped_image, dot_coordinates, radius=5, color=dot_color, thickness=-1)  # -1 fills the circle
        # Define the coordinates for the red dot (assuming you want it at (x, y) = (100, 100))

        # Draw yello line between the dots
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


        # Draw a red dot on the image
        self.frame = cropped_image

        # model_results = model(self.frame)
        # for result in model_results:
        #     boxes = result.boxes
        #     keypoints = result.keypoints  # Keypoints object for keypoint outputs
        #     # Make sure a person was found in the frame
        #     if keypoints is not None and result.boxes.cls[0] == 0:
        #         # Draw the keypoints on the frame
        #         image_size = self.frame.shape[:2]
        #
        #         # Define the color for the red dot in BGR format (OpenCV uses BGR instead of RGB)
        #         dot_color = (0, 0, 255)
        #
        #         for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        #             dot_coordinates = (int(keypoints.data[0][i][0].item()), int(keypoints.data[0][i][1].item()))
        #             # Define the coordinates for the red dot (assuming you want it at (x, y) = (100, 100))
        #
        #             # Draw a red dot on the image
        #             cv2.circle(self.frame, dot_coordinates, radius=5, color=dot_color, thickness=-1)
        #         cv2.imshow('Video Frame', self.frame)
        #         wait = cv2.waitKey(1)
        #         break

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

            # print ("Cropping to y (%d, %d), x (%d, %d)" % (y1, y2, x1, x2))
            self.frame = self.frame[int(y1):int(y2), int(x1):int(x2), :]
            break

        cv2.imshow('Video Frame', self.frame)
        cv2.waitKey(1)
        #
        # # Plot the keypoints on the frame
        #
        # image_size = cropped_image.shape[:2]
        #
        # # Define the color for the red dot in BGR format (OpenCV uses BGR instead of RGB)
        # dot_color = (0, 0, 255)  # (B, G, R)
        #
        # for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        #     dot_coordinates = (int(keypoints[0][0][i][1] * image_size[1]), int(keypoints[0][0][i][0] * image_size[0]))
        # Define the coordinates for the red dot (assuming you want it at (x, y) = (100, 100))

        # # Crop the frame to a square
        # # Calculate the starting point for the crop
        # start_y = (height - 1080) // 2
        # start_x = (width - 1080) // 2
        # # Perform the crop
        # cropped_image = frame[start_y:start_y + 1080, start_x:start_x + 1080, :]
        # cv2.imshow('Video Frame', cropped_image)
        #
        # # Resize the frame to 256x256
        # tf_image = tf.convert_to_tensor(cropped_image, dtype=tf.float32)
        # tf_image = tf.expand_dims(tf_image, axis=0)
        # tf_image = tf.cast(tf.image.resize_with_pad(tf_image, 256, 256), dtype=tf.int32)
        #
        # # Run model inference
        # outputs = movenet(tf_image)
        # keypoints = outputs['output_0'].numpy()
        #
        # # Plot the keypoints on the frame
        #
        # image_size = cropped_image.shape[:2]
        #
        # # Define the color for the red dot in BGR format (OpenCV uses BGR instead of RGB)
        # dot_color = (0, 0, 255)  # (B, G, R)
        #
        # for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        #     dot_coordinates = (int(keypoints[0][0][i][1] * image_size[1]), int(keypoints[0][0][i][0] * image_size[0]))
            # Define the coordinates for the red dot (assuming you want it at (x, y) = (100, 100))

            # Draw a red dot on the image
     #       cv2.circle(cropped_image, dot_coordinates, radius=5, color=dot_color, thickness=-1)  # -1 fills the circle


class PersonVideo:

    # Constructor.  If filename is provided, load the video.
    def __init__(self, filename = None):
        self.keypoints = []
        self.frames = []
        self.smallest_square_size = 0

        if filename is not None:
            self.filename = filename
            self.load_video(filename)
            print ("Loaded video %s (cap %s)" % (self.filename, self.cap))
        self.outfilename = os.path.splitext(self.filename)[0] + '-out.mp4'

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

    def save_video(self):
        height, width, _ = self.frames[0].frame.shape
        print ("Saving: %s" % self.outfilename)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.outfilename, fourcc, 20.0,
                              (width, height))

        for frame in self.frames:
            # Write the frame to the output file
            out.write(frame.frame)
        # Release everything if job is finished
        out.release()

    def analyze(self):
        max_height = 0
        max_width = 0
        for frame in self.frames:
            #frame.crop_human()
            height, width = frame.find_smallest_square()
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        print("Max height: %d, max width: %d" % (max_height, max_width))
        self.smallest_square_size = max(max_height, max_width)

    def draw_keypoints(self):
        for frame in self.frames:
            frame.draw_keypoints()

    def show(self):
        for frame in self.frames:
            # frame.crop_human(self.smallest_square_size)
            rv = frame.show()
            if rv == False:
                return False
        return True
