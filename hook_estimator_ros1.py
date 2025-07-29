import rospy
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from collections import deque
import tf
from tf import TransformListener
from filterpy.kalman import ExtendedKalmanFilter
from scipy.spatial.transform import Rotation
from datetime import datetime
import os
import csv
from itertools import groupby
from operator import itemgetter
from scipy.spatial.transform import Rotation


class KNN(object):
    def __init__(self):
        rospy.init_node('k_nearest_neighbors')

        self.subscription_front_camera = rospy.Subscriber(
            '/camera1/usb_cam/image_raw', Image, self.listener_callback_front, queue_size=10)

        self.subscription_rear_camera = rospy.Subscriber(
            '/camera2/usb_cam/image_raw', Image, self.listener_callback_rear, queue_size=10)

        self.subscription_accel = rospy.Subscriber(
            '/mavros/imu/data', Imu, self.imu_callback, queue_size=10)
        self.bridge = CvBridge()

        self.cv_image_front = None
        self.cv_image_rear = None
        self.front_stamp = None
        self.rear_stamp = None

        self.hook_timer = rospy.Timer(rospy.Duration(0.1), self.process_hook_estimation)

        # Initialization (in __init__ or once)
        self.est_Hook_pos_in_winch_frame = np.array([np.nan, np.nan, np.nan])
        self.est_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])
        self.est_Hook_pos_in_odom_frame = np.array([np.nan, np.nan, np.nan])
        self.ekf_Hook_pos_in_winch_frame = np.array([np.nan, np.nan, np.nan])
        self.ekf_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])
        self.ekf_Hook_pos_in_odom_frame = np.array([np.nan, np.nan, np.nan])
        self.est_Hook_pos_in_winch_frame_ekf = np.array([np.nan, np.nan, np.nan])
        self.Hook_pos_in_base_link_frame_to_plot = deque(maxlen=2000)
        self.Hook_pos_in_odom_frame_to_plot = deque(maxlen=2000)
        self.est_Hook_pos_in_base_link_frame_to_plot = deque(maxlen=2000)
        self.est_Hook_pos_in_odom_frame_to_plot = deque(maxlen=2000)
        self.ekf_Hook_pos_in_base_link_frame_to_plot = deque(maxlen=2000)
        self.ekf_Hook_pos_in_odom_frame_to_plot = deque(maxlen=2000)
        self.est_hook_timestamps = deque(maxlen=2000)  # Store timestamps of hook detections
        self.front_camera_sees_hook = deque(maxlen=2000)
        self.rear_camera_sees_hook = deque(maxlen=2000)

        self.plot_initialized = False
        self.ekf_est_hook_map_pos_to_save = deque(maxlen=2000)  # Store EKF hook estimated pos
        self.ekf_initialized = False
        self.last_ekf_time = None

        self.target_length = 2.0 # Length of the rope in meters, can be changed by winch control
        self.target_length_publish_count = 0
        self.max_publish_count = 2

        # EKF setup
        self.ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)  # State: [p, v]; Measured: [r]
        self.ekf.x = np.zeros(6)
        self.ekf.P *= 1e-1
        self.ekf.R = np.eye(3) * 1e-2                       # Measurement noise (camera)
        self.ekf.Q = np.eye(6) * 1e-1                       # Process noise

        # Initialize IMU values
        self.a_drone = None
        self.a_drone_odom = None

        # Initialize the tf buffer and listener
        self.tf_listener = tf.TransformListener()

    def listener_callback_front(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.cv_image_front = self.adjust_saturation(cv_image)
        self.front_stamp = msg.header.stamp

    def listener_callback_rear(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.cv_image_rear = self.adjust_saturation(cv_image)
        self.rear_stamp = msg.header.stamp

    def adjust_saturation(self, image, sat_factor=1.0):
        imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
        h, s, v = cv2.split(imghsv)
        s *= sat_factor
        s = np.clip(s, 0, 255)
        imghsv = cv2.merge([h, s, v])
        return cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    
    def filter_similar_lines_by_position_and_angle(self, lines, position_thresh=20, angle_thresh=5):
        """
        Filters out lines that are spatially close and have similar orientation.

        Args:
            lines (np.ndarray): Output from cv2.HoughLinesP, shape (N, 1, 4)
            position_thresh (float): Maximum distance between start/end points to consider lines similar
            angle_thresh (float): Maximum angle difference in degrees to consider lines similar

        Returns:
            List of filtered lines, each as [x1, y1, x2, y2]
        """
        if lines is None:
            return []

        lines = [line[0] for line in lines]  # unpack
        accepted = []

        def compute_angle(x1, y1, x2, y2):
            return np.degrees(np.arctan2(y2 - y1, x2 - x1))

        for i, line_i in enumerate(lines):
            x1_i, y1_i, x2_i, y2_i = line_i
            angle_i = compute_angle(x1_i, y1_i, x2_i, y2_i)
            keep = True

            for line_j in accepted:
                x1_j, y1_j, x2_j, y2_j = line_j
                angle_j = compute_angle(x1_j, y1_j, x2_j, y2_j)

                # Distance between corresponding points
                d_start = np.hypot(x1_i - x1_j, y1_i - y1_j)
                d_end   = np.hypot(x2_i - x2_j, y2_i - y2_j)

                if d_start < position_thresh and d_end < position_thresh and abs(angle_i - angle_j) < angle_thresh:
                    keep = False
                    break

            if keep:
                accepted.append(line_i)

        return accepted

    def estimate_hook_pixel_position(self, cv_image, rear_camera=False):
        """
        Estimates the hook's pixel position from the camera image using HSV color filtering and contour analysis.

        Args:
            cv_image (np.ndarray): The input image (in BGR format).
            rear_camera (bool): Whether the image is from the rear camera. If True, use argmin instead of argmax.

        Returns:
            tuple or None: (x, y) pixel coordinates of the hook tip, or None if not detected.
        """

        # if rear_camera:
        #     return None

        result = None

        # Convert to HSV
        cv_image_original = cv_image.copy()
        imghsv = cv2.cvtColor(cv_image_original, cv2.COLOR_BGR2HSV)

        cv2.imshow("HSV imghsv original", cv_image_original)

        # HSV range for hook detection (adjust as needed)
        lower_red = np.array([0, 180, 100])
        upper_red = np.array([220, 255, 225])
        hsv_thresh_hook = cv2.inRange(imghsv, lower_red, upper_red)

        # cv2.imshow("HSV Threshold Hook", hsv_thresh_hook)

        preview_hook = cv2.bitwise_and(cv_image, cv_image, mask=hsv_thresh_hook)

        # cv2.imshow("HSV preview_hook", preview_hook)
	# cv2.waitKey(1)

        # Find contours
        contours, _ = cv2.findContours(hsv_thresh_hook, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the biggest area (assumed to be the rope/hook)
        max_area = 0
        biggest_area = None
        cv_image_biggest_area = cv_image.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                biggest_area = cnt
                
        # if biggest_area is None:
        #     # Draw the biggest contour in green, thickness 2
        #     cv2.drawContours(cv_image_biggest_area, [biggest_area], -1, (0, 255, 0), 2)
        #     # Show the result
        #     cv2.imshow("Biggest Area Contour", cv_image_biggest_area)

        # Convert in greyscale for contour detection
        hsv_thresh_hook_gray = cv2.cvtColor(hsv_thresh_hook, cv2.COLOR_GRAY2BGR)

        # cv2.imshow("Grey Threshold Hook", hsv_thresh_hook_gray)

        # Create mask from the biggest contour
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)

        if biggest_area is not None and max_area > 50:  # Ensure the area is significant
            cv2.drawContours(mask, [biggest_area], -1, 255, thickness=cv2.FILLED)

            # Edge detection on the mask
            edges = cv2.Canny(mask, 50, 150, apertureSize=3)

            # cv2.imshow("Canny Edges", edges)

            # Hough Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=10)
            print(f"Detected lines: {len(lines) if lines is not None else 0}")
            filtered_lines = self.filter_similar_lines_by_position_and_angle(lines, position_thresh=20, angle_thresh=5)

            # Compute line lengths if lines are detected
            if filtered_lines is not None:
                line_lengths = [np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) for x1, y1, x2, y2 in filtered_lines]
                print(f"Line lengths: {line_lengths}")

            # Draw the lines
            cv_image_hough = cv_image.copy()
            if filtered_lines is not None:
                for line in filtered_lines:
                    if line.shape == (1, 4):
                        x1, y1, x2, y2 = line[0]
                    elif line.shape == (4,):
                        x1, y1, x2, y2 = line
                    cv2.line(cv_image_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green line, thickness 2

                # Show the image with lines
                # cv2.imshow("Detected Lines", cv_image_hough)
                self.gb_cv_image_hough = cv_image_hough.copy()
            
            # Get indices of the two longest lines or the longest line if less than two
            if filtered_lines is not None and len(line_lengths) > 1:
                longest_indices = np.argsort(line_lengths)[-2:]
                longest_lines = [lines[i] for i in longest_indices]
            elif lines is not None and len(line_lengths) == 1:
                longest_lines = [lines[0]]
            else:
                longest_lines = []

            # Calculate intersection point of the two longest lines
            if len(longest_lines) == 2:
                # Unpack the lines correctly
                if longest_lines[0].shape == (1, 4):
                    x1, y1, x2, y2 = longest_lines[0][0]
                    x3, y3, x4, y4 = longest_lines[1][0]
                elif longest_lines[0].shape == (4,):
                    x1, y1, x2, y2 = longest_lines[0]
                    x3, y3, x4, y4 = longest_lines[1]
                else:
                    rospy.logwarn("Unexpected line format.")
                    return None

                # All 4 endpoints
                points_line1 = [(x1, y1), (x2, y2)]
                points_line2 = [(x3, y3), (x4, y4)]

                # Find the closest pair of points
                min_distance = float('inf')
                closest_pair = None

                for pt1 in points_line1:
                    for pt2 in points_line2:
                        dist = (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2
                        if dist < min_distance:
                            min_distance = dist
                            closest_pair = (pt1, pt2)

                # Compute midpoint between closest points
                (xA, yA), (xB, yB) = closest_pair
                midpoint = ((xA + xB) // 2, (yA + yB) // 2)

                result = midpoint

            elif len(longest_lines) == 1:
                # If only one line is detected, return the exterme point nearest to the center of the image
                if longest_lines[0].shape == (1, 4):
                    x1, y1, x2, y2 = longest_lines[0][0]
                elif longest_lines[0].shape == (4,):
                    x1, y1, x2, y2 = longest_lines[0]

                # Compute image center
                img_h, img_w = cv_image.shape[:2]
                center_x, center_y = img_w // 2, img_h // 2

                # Compute distances from endpoints to center
                dist1 = np.hypot(x1 - center_x, y1 - center_y)
                dist2 = np.hypot(x2 - center_x, y2 - center_y)

                # Choose the endpoint closest to the center
                if dist1 < dist2:
                    nearest_point = (int(x1), int(y1))
                else:
                    nearest_point = (int(x2), int(y2))

                result = nearest_point

            if rear_camera:
                top10_by_x = sorted(biggest_area, key=lambda p: p[0][0])[:10]
                top10_by_y = sorted(top10_by_x, key=lambda p: p[0][1])[:10]
            else:
                top10_by_x = sorted(biggest_area, key=lambda p: p[0][0], reverse=True)[:10]
                top10_by_y = sorted(top10_by_x, key=lambda p: p[0][1], reverse=True)[:10]

            vertex = tuple(top10_by_y[0][0])

            if result is not None:
                if np.linalg.norm(np.array(vertex) - np.array(result)) < 10:
                    # If the vertex is close to the result, return it
                    output = result
                else:
                    output = vertex
            else:
                output = vertex

            height, width = cv_image.shape[:2]
            edge_margin = 2

            # Check if on edge
            on_edge = any([
                output[0] <= edge_margin, output[0] >= width - edge_margin,
                output[1] <= edge_margin, output[1] >= height - edge_margin,
            ])

            if on_edge:
                return None
            else:
                return output
            
        else:
            rospy.logwarn("No contours found for hook detection.")
            return None

    def update_transforms(self):
        """
        Computes and stores translation and rotation matrices between all relevant frames.
        Results are stored as self.A_pos_in_B_frame and self.A_rot_in_B_frame.
        """
        frame_list = [
            'base_link',
            'camera1_link',
            'camera2_link',
            'winch_link',
            'imu_link'
        ]

        # Prefix for frames (adjust as needed)
        full_frames = {name: '{}'.format(name) for name in frame_list}


        now = rospy.Time.now()

        for A_name, A_frame in full_frames.items():
            for B_name, B_frame in full_frames.items():
                if A_name == B_name:
                    continue  # skip same frame
                try:
                    # lookup_transform returns (trans, rot)
                    (trans, rot_quat) = self.tf_listener.lookupTransform(
                        B_frame,  # target_frame
                        A_frame,  # source_frame
                        now
                    )

                    # Convert translation to numpy array
                    pos = np.array(trans)

                    # Convert quaternion to rotation matrix
                    rot = Rotation.from_quat([rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]]).as_dcm()

                    # Save with dynamic attribute naming
                    pos_attr = "{}_pos_in_{}_frame".format(A_name, B_name)
                    rot_attr = "{}_rot_in_{}_frame".format(A_name, B_name)

                    setattr(self, pos_attr, pos)
                    setattr(self, rot_attr, rot)

                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    pos_attr = "{}_pos_in_{}_frame".format(A_name, B_name)
                    rot_attr = "{}_rot_in_{}_frame".format(A_name, B_name)

                    setattr(self, pos_attr, None)
                    setattr(self, rot_attr, None)
                    rospy.logwarn("TF lookup failed for {} -> {}: {}".format(A_frame, B_frame, e))
                
    def is_in_frame(self, pt, img_w, img_h):
        if pt is None or not (isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 2):
            return False
        x, y = pt
        return 0 <= x < img_w and 0 <= y < img_h


    def estimate_hook_position(self, tip_front_cam, tip_rear_cam):
        """
        Estimate the 3D position of the hook in winch frame coordinates using one or both camera views.
        Stores result in self.est_Hook_pos_in_winch_frame.
        """

        K = np.array([
                    [369.5, 0, 320],
                    [0, 415.69, 240],
                    [0, 0, 1]
                ])
        img_w = int(K[0, 2] * 2)
        img_h = int(K[1, 2] * 2)

        hook_positions = []

        # === FRONT CAMERA ===
        if tip_front_cam is not None and self.is_in_frame(tip_front_cam, img_w, img_h):
            self.front_camera_sees_hook.append(True)
            try:
                p_img = np.array(tip_front_cam)
                R_front = self.camera_link_rot_in_winch_link_frame
                T_front = self.camera_link_pos_in_winch_link_frame
                pos_front = self.reconstruct_hook_position(p_img, K, R_front, T_front, self.target_length)
                hook_positions.append(pos_front)
            except Exception as e:
                rospy.logwarn("[Front camera] Hook reconstruction failed: {}".format(e))
        else:
            self.front_camera_sees_hook.append(False)


        # === REAR CAMERA ===
        if tip_rear_cam is not None and self.is_in_frame(tip_rear_cam, img_w, img_h):
            self.rear_camera_sees_hook.append(True)
            try:
                p_img = np.array(tip_rear_cam)
                R_rear = self.camera_link_2_rot_in_winch_link_frame
                T_rear = self.camera_link_2_pos_in_winch_link_frame
                pos_rear = self.reconstruct_hook_position(p_img, K, R_rear, T_rear, self.target_length)
                hook_positions.append(pos_rear)
            except Exception as e:
                rospy.logwarn("[Rear camera] Hook reconstruction failed: {}".format(e))
        else:
            self.rear_camera_sees_hook.append(False)


        # === DECIDE FINAL POSITION ===
        if len(hook_positions) == 0:
            rospy.logwarn("No valid hook detections from either camera.")
            self.est_Hook_pos_in_winch_frame = None
        elif len(hook_positions) == 1:
            self.est_Hook_pos_in_winch_frame = hook_positions[0]
            rospy.loginfo("Hook estimated from a single camera.")
        else:
            self.est_Hook_pos_in_winch_frame = np.mean(hook_positions, axis=0)
            rospy.loginfo("Hook estimated by averaging both camera estimates.")

    def draw_hook_info(self, image, tip_rope, window_name='Hook Detection'):
        """
        Draws the hook tip and estimated 3D position (in base_link frame) on the image.

        Args:
            image (np.ndarray): BGR image where to draw.
            tip_rope (tuple or None): (x, y) pixel coordinates of the detected hook tip or None.
        """
        if image is None:
            return

        preview_hook = image.copy()

        if tip_rope is not None:
            # Draw the hook tip as a red filled circle
            cv2.circle(preview_hook, tuple(map(int, tip_rope)), 10, (0, 255, 255), 3)
            text_pos = (int(tip_rope[0] + 10), int(tip_rope[1]))
        else:
            # Default text position if no tip detected
            text_pos = (10, 30)

        # Prepare the position text based on estimated 3D hook position in base_link frame
        if hasattr(self, 'est_Hook_pos_in_base_link_frame') and \
        isinstance(self.est_Hook_pos_in_base_link_frame, np.ndarray) and \
        self.est_Hook_pos_in_base_link_frame.shape == (3,):
            pos = self.est_Hook_pos_in_base_link_frame
            pos_text = "Position (base_link): ({:.2f}, {:.2f}, {:.2f})".format(pos[0], pos[1], pos[2])

        else:
            pos_text = "Position (base_link): N/A"

        # Put the position text on the image
        cv2.putText(preview_hook, pos_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show the image window
        cv2.imshow(window_name, preview_hook)
        cv2.waitKey(1)

    def reconstruct_hook_position(self, p_img, K, R, T, L):
        """ Reconstructs the 3D position of the hook from a 2D image point.
        Parameters:
        - p_img: 2D image point (2,)
        - K: Camera intrinsic matrix (3x3)
        - R: Camera rotation matrix (3x3)
        - T: Camera translation vector (3,)
        - L: Rope length (scalar)

        Returns:
        - pos_3d: Reconstructed 3D position(s) of the hook in winch coordinates (up to 2 points)
        """
        # Convert image point to homogeneous coordinates
        p_img_hom = np.array([p_img[0], p_img[1], 1.0])

        # Compute direction of the ray in camera coordinates

        ray_dir_cam = np.dot(np.linalg.inv(K), p_img_hom)

        # Assuming the camera is rotated to align with winch coordinates
        ray_dir_cam = np.dot(np.array([
                                        [0, 0, 1],
                                        [0, 1, 0],
                                        [1, 0, 0]
                                    ]), ray_dir_cam)



        # Convert ray direction to winch coordinates
        ray_dir_world = np.dot(R, ray_dir_cam)



        # Camera center in winch coordinates
        C = T

        # The ray: P(s) = C + s * ray_dir_world
        # Intersect with sphere centered at origin (0, 0, 0) with radius L

        a = np.dot(ray_dir_world, ray_dir_world)
        b = 2 * np.dot(ray_dir_world, C)
        c = np.dot(C, C) - L**2

        # Solve quadratic: a*s^2 + b*s + c = 0
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            raise ValueError("No intersection with the pendulum sphere.")

        sqrt_disc = np.sqrt(discriminant)
        s = (-b + sqrt_disc) / (2*a)

        P = C + s * ray_dir_world

        return P

    def get_actual_frequency_and_dt(self):
        if len(self.est_hook_timestamps) < 2:
            return None, None
        intervals = np.diff(self.est_hook_timestamps)
        avg_dt = np.mean(intervals)
        freq = 1.0 / avg_dt if avg_dt > 0 else 0
        return freq, avg_dt

    def process_hook_estimation(self, msg=None):
        """
        Main hook estimation loop. Must be called repeatedly via timer or update event.
        Requires that self.cv_image_front and self.cv_image_rear are already populated.
        """

        if self.cv_image_front is None and self.cv_image_rear is None:
            rospy.logwarn("No camera images received yet.")
            return

        # Step 1: Estimate hook tip pixel positions
        tip_front = self.estimate_hook_pixel_position(self.cv_image_front, rear_camera=False) if self.cv_image_front is not None else None
        tip_rear = self.estimate_hook_pixel_position(self.cv_image_rear, rear_camera=True) if self.cv_image_rear is not None else None

        # Step 2: Update TF transforms (R, T for both cameras in winch frame)
        self.update_transforms()

        # Step 3: Estimate 3D hook position using one or both camera tips
        self.estimate_hook_position(tip_front, tip_rear)
        print("Estimated hook position in winch frame:", self.est_Hook_pos_in_winch_frame)

        # Step 4: Initialize EKF if needed
        if self.est_Hook_pos_in_winch_frame is not None:
	    self.est_Hook_pos_in_winch_frame_ekf = np.dot(np.array([
						    [0, 1, 0],
						    [-1, 0, 0],
						    [0, 0, 1]
						]), self.est_Hook_pos_in_winch_frame)

            try:
                # ======= EKF Initialization (once) =======
                if not self.ekf_initialized:
                    try:
                        pos = np.array(self.est_Hook_pos_in_winch_frame_ekf, dtype=float)
                        if pos.shape == (3,):
                            self.ekf.x[0:3] = pos.flatten()
                            self.ekf.x[3:6] = np.zeros(3)
                            # angles = self.cartesian_to_spherical(pos)
                            # self.ekf.x = np.array([angles[0], angles[1], 0.0, 0.0])
                            self.ekf_initialized = True
                            rospy.loginfo("EKF initialized with first hook position.")
                            print("EKF initial value:", self.ekf.x)
                        else:
                            rospy.logwarn("Invalid hook position shape during EKF init: {}".format(pos.shape))
                    except Exception as e:
                        rospy.logwarn("EKF initialization failed: {}".format(e))
            except Exception as e:
                rospy.logwarn("Error during EKF initialization: {}".format(e))
                self.ekf_initialized = False

        
        # Step 5: EKF prediction/update
        now = rospy.Time.now()

        if self.last_ekf_time is None:
            self.last_ekf_time = now
            now_sec = now.secs + now.nsecs * 1e-9
            self.est_hook_timestamps.append(now_sec)
            dt = 0.0  # No time has passed on the first update
        else:
            dt = now.secs + now.nsecs * 1e-9 - self.last_ekf_time.secs - self.last_ekf_time.nsecs * 1e-9
            self.last_ekf_time = now
            self.est_hook_timestamps.append(now.secs + now.nsecs * 1e-9)

        meas = (
                np.array(self.est_Hook_pos_in_winch_frame_ekf).reshape(3, 1)
                if self.est_Hook_pos_in_winch_frame_ekf is not None and not np.isnan(self.est_Hook_pos_in_winch_frame_ekf).any()
                else None
                )

        try:
            if meas is not None:
                self.update_ekf_constrained(dt, self.a_drone_odom, meas)
                #self.update_ekf_constrained( dt, self.a_drone_odom, meas, use_analytical=True)
            else:
                self.predict_ekf_constrained(dt, self.a_drone_odom)
                #self.predict_ekf_constrained( dt, self.a_drone_odom, use_analytical=True)
        except Exception as e:
            rospy.logwarn("EKF prediction\update failed: {}".format(e))

        print("self.ekf_Hook_pos_in_winch_frame:", self.ekf_Hook_pos_in_winch_frame)

        # Step 6: Transform hook position to base_link and odom frames
        try:
            pos_winch = self.est_Hook_pos_in_winch_frame  # shape (3,)
            rot_winch_in_base = getattr(self, "winch_link_rot_in_base_link_frame", None)
            pos_winch_in_base = getattr(self, "winch_link_pos_in_base_link_frame", None)
            rot_base_in_odom = getattr(self, "base_link_rot_in_odom_frame", None)
            pos_base_in_odom = getattr(self, "base_link_pos_in_odom_frame", None)
        except Exception as e:
            rospy.logwarn("Error transforming hook position to base_link and odom: {}".format(e))
            self.est_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])
            self.est_Hook_pos_in_odom_frame = np.array([np.nan, np.nan, np.nan])
            self.ekf_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])
            self.ekf_Hook_pos_in_odom_frame = np.array([np.nan, np.nan, np.nan])

        if self.est_Hook_pos_in_winch_frame is not None and not np.isnan(self.est_Hook_pos_in_winch_frame).any() and rot_winch_in_base is not None and pos_winch_in_base is not None:
            self.est_Hook_pos_in_base_link_frame = rot_winch_in_base @ pos_winch + pos_winch_in_base
            self.est_Hook_pos_in_base_link_frame = np.array([
                                                            [0, 1, 0],
                                                            [-1, 0, 0],
                                                            [0, 0, 1]
                                                        ]) @ self.est_Hook_pos_in_base_link_frame
        else:
            self.est_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])

                
        if not np.isnan(self.ekf_Hook_pos_in_winch_frame).any() and rot_winch_in_base is not None and pos_winch_in_base is not None:
            self.ekf_Hook_pos_in_base_link_frame = rot_winch_in_base @ self.ekf_Hook_pos_in_winch_frame + pos_winch_in_base
        else:
            self.ekf_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])

        if self.est_Hook_pos_in_base_link_frame is not None and not np.isnan(self.est_Hook_pos_in_base_link_frame).any() and rot_base_in_odom is not None and pos_base_in_odom is not None:
            self.est_Hook_pos_in_odom_frame = rot_base_in_odom @ self.est_Hook_pos_in_base_link_frame + pos_base_in_odom
        else:
            self.est_Hook_pos_in_odom_frame = np.array([np.nan, np.nan, np.nan])
        
        if not np.isnan(self.ekf_Hook_pos_in_winch_frame).any() and rot_base_in_odom is not None and pos_base_in_odom is not None:
            self.ekf_Hook_pos_in_odom_frame = rot_base_in_odom @ self.ekf_Hook_pos_in_base_link_frame + pos_base_in_odom
        else:
            self.ekf_Hook_pos_in_odom_frame = np.array([np.nan, np.nan, np.nan])

        # Step 5: Save positions for plotting
        if self.Hook_pos_in_base_link_frame is None:
            self.Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])
        if self.Hook_pos_in_odom_frame is None:
            self.Hook_pos_in_odom_frame = np.array([np.nan, np.nan, np.nan])
        self.Hook_pos_in_base_link_frame_to_plot.append(list(self.Hook_pos_in_base_link_frame))
        self.Hook_pos_in_odom_frame_to_plot.append(list(self.Hook_pos_in_odom_frame))
        self.est_Hook_pos_in_base_link_frame_to_plot.append(list(self.est_Hook_pos_in_base_link_frame))
        self.est_Hook_pos_in_odom_frame_to_plot.append(list(self.est_Hook_pos_in_odom_frame))
        self.ekf_Hook_pos_in_base_link_frame_to_plot.append(list(self.ekf_Hook_pos_in_base_link_frame))
        self.ekf_Hook_pos_in_odom_frame_to_plot.append(list(self.ekf_Hook_pos_in_odom_frame))

        # Step 6: Draw hook info on images if available
        if self.cv_image_front is not None:
            self.draw_hook_info(self.cv_image_front, tip_front, 'Front Camera Hook Detection')
        if self.cv_image_rear is not None:
            self.draw_hook_info(self.cv_image_rear, tip_rear, 'Rear Camera Hook Detection')


        if self.gb_cv_image_hough is not None:
            self.draw_hook_info(self.gb_cv_image_hough, tip_rear, 'Hough Lines and Hook Detection')


    def imu_callback(self, msg):
        self.a_drone = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Rotate acceleration vector from IMU frame to odom frame
        # Check if the rotation matrix exists and is valid
        rot = getattr(self, "imu_link_rot_in_base_link_frame", None)
        if rot is not None:
            self.a_drone_odom = np.dot(rot, self.a_drone)

        else:
            rospy.logwarn("Rotation matrix imu_link_rot_in_base_link_frame not available")
            self.a_drone_odom = None

    def fx(self, x, dt, a_drone):
        x = np.asarray(x).flatten()
        p = x[0:3]
        v = x[3:6]
        a_eff = self.g_vec - a_drone

        # Remove radial component of acceleration to stay on sphere
        acc = a_eff - np.dot(a_eff, p) / (np.linalg.norm(p)**2 + 1e-6) * p

        new_v = v + acc * dt
        new_p = p + new_v * dt
        return np.concatenate((new_p, new_v))

    def jacobian_F(self, x, dt, a_drone):
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        return F

    def hx(self, x):
        return x[0:3]

    def project_to_sphere(self, x, L):
        p = x[0:3].flatten()
        v = x[3:6].flatten()

        norm_p = np.linalg.norm(p)
        if norm_p > 1e-3:
            p_proj = p / norm_p * L
            v_proj = v - (np.dot(v, p_proj) / (L ** 2)) * p_proj
        else:
            p_proj = p
            v_proj = v

        return np.concatenate((p_proj, v_proj)).flatten()

    def project_covariance(self, P, x, L):
        p = x[0:3].flatten()
        norm_p = np.linalg.norm(p)
        if norm_p < 1e-3:
            return P

        p_unit = p / norm_p
        T = np.eye(3) - np.outer(p_unit, p_unit)

        P_proj = P.copy()
        #P_proj[0:3, 0:3] = T @ P[0:3, 0:3] @ T.T
        #P_proj[0:3, 3:6] = T @ P[0:3, 3:6]
	P_proj[0:3, 0:3] = np.dot(np.dot(T, P[0:3, 0:3]), T.T)
	P_proj[0:3, 3:6] = np.dot(T, P[0:3, 3:6])
        P_proj[3:6, 0:3] = P_proj[0:3, 3:6].T
        return P_proj

    # -------------------------------
    # ✅ Prediction Only
    # -------------------------------
    def predict_ekf(self, dt, a_drone):
        self.ekf.F = self.jacobian_F(self.ekf.x, dt, a_drone)
        self.ekf.x = self.fx(self.ekf.x, dt, a_drone)
        self.ekf.P = self.ekf.F @ self.ekf.P @ self.ekf.F.T + self.ekf.Q

        self.ekf_Hook_pos_in_winch_frame = self.ekf.x[0:3].copy()

    def predict_ekf_constrained(self, dt, a_drone):
        self.predict_ekf(dt, a_drone)
        self.ekf.x = self.project_to_sphere(self.ekf.x, self.target_length)
        self.ekf.P = self.project_covariance(self.ekf.P, self.ekf.x, self.target_length)

        self.ekf_Hook_pos_in_winch_frame = self.ekf.x[0:3].copy()
        print("self.ekf_Hook_pos_in_winch_frame:", self.ekf_Hook_pos_in_winch_frame)

    # -------------------------------
    # ✅ Predict + Update (with measurement)
    # -------------------------------
    def update_ekf(self, dt, a_drone, measurement):
        self.predict_ekf(dt, a_drone)

        H = np.hstack((np.eye(3), np.zeros((3, 3))))
        z = measurement.flatten()
        hx = self.hx(self.ekf.x).flatten()
        y = z - hx
        S = H @ self.ekf.P @ H.T + self.ekf.R
        K = self.ekf.P @ H.T @ np.linalg.inv(S)
        self.ekf.x = self.ekf.x + K @ y
        self.ekf.P = (np.eye(6) - K @ H) @ self.ekf.P


    def update_ekf_constrained(self, z_meas_pos, a_drone, dt):
        # ===== EKF PREDICTION =====
        self.ekf.F = self.F_jacobian(self.ekf.x, dt, a_drone)
        self.ekf.x = self.fx(self.ekf.x, dt, a_drone)
        #self.ekf.P = self.ekf.F @ self.ekf.P @ self.ekf.F.T + self.ekf.Q
	self.ekf.P = np.dot(np.dot(self.ekf.F, self.ekf.P), self.ekf.F.T) + self.ekf.Q
        self.ekf_Hook_pos_in_winch_frame = self.ekf.x[0:3].copy()


    def update_ekf_constrained(self, dt, a_drone, measurement):
        self.predict_ekf_constrained(dt, a_drone)

        H = np.hstack((np.eye(3), np.zeros((3, 3))))
        z = measurement.flatten()
        hx = self.hx(self.ekf.x).flatten()
        y = z - hx
        
	#S = H @ self.ekf.P @ H.T + self.ekf.R
        #K = self.ekf.P @ H.T @ np.linalg.inv(S)
        #self.ekf.x = self.ekf.x + K @ y
        #self.ekf.P = (np.eye(6) - K @ H) @ self.ekf.P

	S = np.dot(np.dot(H, self.ekf.P), H.T) + self.ekf.R
	K = np.dot(np.dot(self.ekf.P, H.T), np.linalg.inv(S))
	self.ekf.x = self.ekf.x + np.dot(K, y)
	self.ekf.P = np.dot(np.eye(6) - np.dot(K, H), self.ekf.P)

        self.ekf.x = self.project_to_sphere(self.ekf.x, self.target_length)
        self.ekf.P = self.project_covariance(self.ekf.P, self.ekf.x, self.target_length)

        self.ekf_Hook_pos_in_winch_frame = self.ekf.x[0:3].copy()

    def _align_arrays(self, *arrays):
        """
        Aligns arrays (3D vectors or scalars) to the same length by padding with NaNs.
        Handles empty arrays safely.
        """
        cleaned = []
        is_vector = []

        for arr in arrays:
            valid_rows = []
            for item in arr:
                if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 3:
                    try:
                        valid_rows.append([float(x) for x in item])
                    except (ValueError, TypeError):
                        continue
                elif isinstance(item, (int, float, np.float64)):
                    valid_rows.append(float(item))
            cleaned.append(valid_rows)
            is_vector.append(len(valid_rows) > 0 and isinstance(valid_rows[0], list))

        max_len = max(len(arr) for arr in cleaned)
        padded = []

        for arr, vec in zip(cleaned, is_vector):
            pad_value = [np.nan, np.nan, np.nan] if vec else np.nan
            padded_arr = arr + [pad_value] * (max_len - len(arr))
            padded.append(padded_arr)

        return tuple(np.array(arr) for arr in padded)

    def save_plot_data(self, save_dir='plot_data'):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	filename = os.path.join(save_dir, 'hook_base_link_{}.npz'.format(timestamp))



        try:
            np.savez(
                filename,
                timestamps=np.array(self.est_hook_timestamps),
                estimated=np.array(self.est_Hook_pos_in_base_link_frame_to_plot),
                ekf=np.array(self.ekf_Hook_pos_in_base_link_frame_to_plot),
                seen_front_f=np.array(self.front_camera_sees_hook),
                seen_rear_f=np.array(self.rear_camera_sees_hook),
            )

            rospy.loginfo("Saved plot data to: {}".format(filename))

        except Exception as e:
            rospy.logerr("Failed to save plot data: {}".format(e))

def main(args=None):
    rospy.init_node('k_nearest_neighbors')
    k_nearest_neighbors = KNN()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down node (Ctrl+C detected)")
    finally:
        k_nearest_neighbors.save_plot_data()
        print("Cleaning up...")


if __name__ == '__main__':
    main()
