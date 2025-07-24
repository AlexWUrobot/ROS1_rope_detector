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
import auv_detector.params_detector_2 as P
from scipy.spatial.transform import Rotation
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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
        self.est_Hook_pos_in_base_link_frame_to_plot = deque(maxlen=2000)
        self.ekf_Hook_pos_in_base_link_frame_to_plot = deque(maxlen=2000)
        self.est_hook_timestamps = deque(maxlen=2000)  # Store timestamps of hook detections

        self.plot_initialized = False
        self.ekf_est_hook_map_pos_to_save = deque(maxlen=2000)  # Store EKF hook estimated pos
        self.ekf_initialized = False

        self.target_length = 2.0 # Length of the rope in meters, can be changed by winch control
        self.target_length_publish_count = 0
        self.max_publish_count = 2

        # EKF setup
        self.ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)  # State: [p, v]; Measured: [r]
        self.ekf.x = np.zeros(6)
        self.ekf.P *= 1e-2
        self.ekf.R = np.eye(3) * 1e-2                       # Measurement noise (camera)
        self.ekf.Q = np.eye(6) * 1e-3                       # Process noise
        self.g_vec = np.array([0, 0, -9.81])  # Gravity vector in base_link frame

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

    def estimate_hook_pixel_position(self, cv_image, rear_camera=False):
        """
        Estimates the hook's pixel position from the camera image using HSV color filtering and contour analysis.

        Args:
            cv_image (np.ndarray): The input image (in BGR format).
            rear_camera (bool): Whether the image is from the rear camera. If True, use argmin instead of argmax.

        Returns:
            tuple or None: (x, y) pixel coordinates of the hook tip, or None if not detected.
        """

        # Convert to HSV
        imghsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # HSV range for hook detection (adjust as needed)
        lower_yellow = np.array([15, 60, 20])
        upper_yellow = np.array([70, 240, 255])
        hsv_thresh_hook = cv2.inRange(imghsv, lower_yellow, upper_yellow)
        preview_hook = cv2.bitwise_and(cv_image, cv_image, mask=hsv_thresh_hook)

        # Find contours
        contours, _ = cv2.findContours(hsv_thresh_hook, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the longest contour (assumed to be the rope/hook)
        max_len = 0
        longest_contour = None
        for cnt in contours:
            length = cv2.arcLength(cnt, False)
            if length > max_len:
                max_len = length
                longest_contour = cnt

        if longest_contour is not None:
            # Draw contour
            cv2.drawContours(preview_hook, [longest_contour], -1, (0, 255, 0), 1)

            if rear_camera:
                vertex_1 = tuple(longest_contour[longest_contour[:, :, 0].argmin()][0])
                vertex_2 = tuple(longest_contour[longest_contour[:, :, 1].argmin()][0])
            else:
                vertex_1 = tuple(longest_contour[longest_contour[:, :, 0].argmax()][0])
                vertex_2 = tuple(longest_contour[longest_contour[:, :, 1].argmax()][0])

            height, width = cv_image.shape[:2]
            edge_margin = 2

            # Check if on edge
            on_edge = any([
                vertex_1[0] <= edge_margin, vertex_1[0] >= width - edge_margin,
                vertex_1[1] <= edge_margin, vertex_1[1] >= height - edge_margin,
                vertex_2[0] <= edge_margin, vertex_2[0] >= width - edge_margin,
                vertex_2[1] <= edge_margin, vertex_2[1] >= height - edge_margin,
            ])

            if on_edge:
                rospy.logwarn("Hook tip detected on image edge, skipping hook position computation.")
                cv2.putText(preview_hook, "Hook tip on edge", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                self.est_hook_pos_in_base_link_frame_to_plot.append([np.nan, np.nan, np.nan])
                return None

            # Compute hook tip position
            tip_x = int((vertex_1[0] + vertex_2[0]) / 2)
            tip_y = int((vertex_1[1] + vertex_2[1]) / 2)
            tip_rope = (tip_x, tip_y)

	    cv2.putText(preview_hook, "{:.2f} m".format(self.target_length), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            return tip_rope

        else:
            rospy.logwarn("No hook contour found, skipping hook position computation.")
            cv2.putText(preview_hook, "Hook not found", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            self.est_hook_pos_in_base_link_frame_to_plot.append([np.nan, np.nan, np.nan])
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
        #full_frames = {name: f'{name}' for name in frame_list}
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
                    rot = Rotation.from_quat([rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]]).as_matrix()

                    # Save with dynamic attribute naming
                    #pos_attr = f"{A_name}_pos_in_{B_name}_frame"
		    pos_attr = "{}_pos_in_{}_frame".format(A_name, B_name)
                    #rot_attr = f"{A_name}_rot_in_{B_name}_frame"
		    rot_attr = "{}_rot_in_{}_frame".format(A_name, B_name)
                    setattr(self, pos_attr, pos)
                    setattr(self, rot_attr, rot)

                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    #rospy.logwarn(f"TF lookup failed for {A_frame} -> {B_frame}: {e}")
	            rospy.logwarn("TF lookup failed for {} -> {}: {}".format(A_frame, B_frame, e))
                    #pos_attr = f"{A_name}_pos_in_{B_name}_frame"
		    pos_attr = "{}_pos_in_{}_frame".format(A_name, B_name)
                    #rot_attr = f"{A_name}_rot_in_{B_name}_frame"
		    rot_attr = "{}_rot_in_{}_frame".format(A_name, B_name)
                    setattr(self, pos_attr, None)
                    setattr(self, rot_attr, None)

    def estimate_hook_position(self, tip_front_cam, tip_rear_cam):
        """
        Estimate the 3D position of the hook in winch frame coordinates using one or both camera views.
        Stores result in self.est_Hook_pos_in_winch_frame.
        """

        hook_positions = []

        # === FRONT CAMERA ===
        if tip_front_cam is not None:
            try:
                p_img = np.array(tip_front_cam)
                K = np.array([
                    [369.5, 0, 320],
                    [0, 415.69, 240],
                    [0, 0, 1]
                ])
                R_front = self.camera1_link_rot_in_winch_link_frame
                T_front = self.camera1_link_pos_in_winch_link_frame
                pos_front = self.reconstruct_hook_position(p_img, K, R_front, T_front, self.target_length)
                hook_positions.append(pos_front)
            except Exception as e:
                #rospy.logwarn(f"[Front camera] Hook reconstruction failed: {e}")
		rospy.logwarn("[Front camera] Hook reconstruction failed: {}".format(e))


        # === REAR CAMERA ===
        if tip_rear_cam is not None:
            try:
                p_img = np.array(tip_rear_cam)
                K = np.array([
                    [369.5, 0, 320],
                    [0, 415.69, 240],
                    [0, 0, 1]
                ])
                R_rear = self.camera2_link_rot_in_winch_link_frame
                T_rear = self.camera2_link_pos_in_winch_link_frame
                pos_rear = self.reconstruct_hook_position(p_img, K, R_rear, T_rear, self.target_length)
                hook_positions.append(pos_rear)
            except Exception as e:
                #rospy.logwarn(f"[Rear camera] Hook reconstruction failed: {e}")
		rospy.logwarn("[Front camera] Hook reconstruction failed: {}".format(e))

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
            cv2.circle(preview_hook, tuple(map(int, tip_rope)), 10, (0, 0, 255), -1)
            text_pos = (int(tip_rope[0] + 10), int(tip_rope[1]))
        else:
            # Default text position if no tip detected
            text_pos = (10, 30)

        # Prepare the position text based on estimated 3D hook position in base_link frame
        if hasattr(self, 'est_Hook_pos_in_base_link_frame') and \
        isinstance(self.est_Hook_pos_in_base_link_frame, np.ndarray) and \
        self.est_Hook_pos_in_base_link_frame.shape == (3,):
            pos = self.est_Hook_pos_in_base_link_frame
            #pos_text = f"Position (base_link): ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
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
        ray_dir_cam = np.linalg.inv(K) @ p_img_hom
        ray_dir_cam = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ]) @ ray_dir_cam  # Assuming the camera is rotated to align with winch coordinates

        # Convert ray direction to winch coordinates
        ray_dir_world = R @ ray_dir_cam

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

        # Step 4: Transform hook position from winch frame to base_link frame if available
        if self.est_Hook_pos_in_winch_frame is not None:
            self.est_Hook_pos_in_winch_frame_ekf = np.array([
                                                    [0, 1, 0],
                                                    [-1, 0, 0],
                                                    [0, 0, 1]
                                                ]) @ self.est_Hook_pos_in_winch_frame
            try:
                # ======= EKF Initialization (once) =======
                if not self.ekf_initialized:
                    try:
                        pos = np.array(self.est_Hook_pos_in_winch_frame_ekf, dtype=float)
                        if pos.shape == (3,):
                            self.ekf.x[0:3] = pos.flatten()
                            self.ekf.x[3:6] = np.zeros(3)
                            self.ekf_initialized = True
                            rospy.loginfo("EKF initialized with first hook position.")
                        else:
                            #rospy.logwarn(f"Invalid hook position shape during EKF init: {pos.shape}")
			    rospy.logwarn("Invalid hook position shape during EKF init: {}".format(pos.shape))
                    except Exception as e:
                        #rospy.logwarn(f"EKF initialization failed: {e}")
			rospy.logwarn("EKF initialization failed: {}".format(e))

                # ======= Timestamp collection (safe) =======
                avg_stamp_sec = None
                try:
                    if self.front_stamp is not None and self.rear_stamp is not None:
                        front_sec = self.front_stamp.to_sec()
                        rear_sec = self.rear_stamp.to_sec()
                        avg_stamp_sec = (front_sec + rear_sec) / 2.0
                        self.est_hook_timestamps.append(avg_stamp_sec)
                    elif self.front_stamp is not None:
                        self.est_hook_timestamps.append(self.front_stamp.to_sec())
                    elif self.rear_stamp is not None:
                        self.est_hook_timestamps.append(self.rear_stamp.to_sec())
                    else:
                        rospy.logwarn("Front or rear stamp missing; cannot append timestamp.")
                except Exception as e:
                    #rospy.logwarn(f"Timestamp computation failed: {e}")
		    rospy.logwarn("Timestamp computation failed: {}".format(e))

                # ======= EKF Update (if enough data and valid) =======
                if (
                    self.ekf_initialized and
                    self.est_Hook_pos_in_winch_frame_ekf is not None and
                    not np.isnan(self.est_Hook_pos_in_winch_frame_ekf).any() and
                    len(self.est_hook_timestamps) >= 2
                ):
                    t1 = self.est_hook_timestamps[-1]
                    t0 = self.est_hook_timestamps[-2]

                    if t1 is not None and t0 is not None:
                        try:
                            dt = t1 - t0
                            if dt is None or not isinstance(dt, (float, int)):
                                rospy.logwarn("Computed dt is None or not a number; skipping EKF update.")
                                self.ekf_Hook_pos_in_winch_frame = np.array([np.nan, np.nan, np.nan])
                            else:
                                a_drone = self.a_drone_odom
                                meas = np.array(self.est_Hook_pos_in_winch_frame_ekf, dtype=float).reshape(3, 1)
                                self.update_ekf_constrained(meas, a_drone, dt)
                        except Exception as e:
                            #rospy.logwarn(f"EKF update failed: {e}")
			    rospy.logwarn("EKF update failed: {}".format(e))
                            self.ekf_Hook_pos_in_winch_frame = np.array([np.nan, np.nan, np.nan])
                    else:
                        rospy.logwarn("Timestamps are None; skipping EKF update.")
                else:
                    self.ekf_Hook_pos_in_winch_frame = np.array([np.nan, np.nan, np.nan])

                # winch to base_link
                pos_winch = self.est_Hook_pos_in_winch_frame  # shape (3,)
                rot_winch_in_base = getattr(self, "winch_link_rot_in_base_link_frame", None)
                pos_winch_in_base = getattr(self, "winch_link_pos_in_base_link_frame", None)

                if rot_winch_in_base is not None and pos_winch_in_base is not None:
                    self.est_Hook_pos_in_base_link_frame = rot_winch_in_base @ pos_winch + pos_winch_in_base
                    self.est_Hook_pos_in_base_link_frame = np.array([
                                                                    [0, 1, 0],
                                                                    [-1, 0, 0],
                                                                    [0, 0, 1]
                                                                ]) @ self.est_Hook_pos_in_base_link_frame
                    if not np.isnan(self.ekf_Hook_pos_in_winch_frame).any():
                        self.ekf_Hook_pos_in_base_link_frame = rot_winch_in_base @ self.ekf_Hook_pos_in_winch_frame + pos_winch_in_base
                    else:
                        self.ekf_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])
                else:
                    self.est_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])
                    rospy.logwarn("Missing winch_link -> base_link transform for hook position.")

            except Exception as e:
                #rospy.logwarn(f"Error transforming hook position: {e}")
		rospy.logwarn("Error transforming hook position: {}".format(e))
                self.est_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])
                self.ekf_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])

        else:
            rospy.logwarn("Hook position estimation failed.")
            self.est_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])
            self.ekf_Hook_pos_in_base_link_frame = np.array([np.nan, np.nan, np.nan])

        # Step 5: Save positions for plotting
        self.est_Hook_pos_in_base_link_frame_to_plot.append(list(self.est_Hook_pos_in_base_link_frame))
        self.ekf_Hook_pos_in_base_link_frame_to_plot.append(list(self.ekf_Hook_pos_in_base_link_frame))

        # Step 6: Draw hook info on images if available
        if self.cv_image_front is not None:
            self.draw_hook_info(self.cv_image_front, tip_front, 'Front Camera Hook Detection')
        if self.cv_image_rear is not None:
            self.draw_hook_info(self.cv_image_rear, tip_rear, 'Rear Camera Hook Detection')
    def imu_callback(self, msg: Imu):
        self.a_drone = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Rotate acceleration vector from IMU frame to odom frame
        # Check if the rotation matrix exists and is valid
        rot = getattr(self, "imu_link_rot_in_odom_frame", None)
        if rot is not None:
            self.a_drone_odom = rot @ self.a_drone  # matrix multiply rotation * vector
        else:
            rospy.logwarn("Rotation matrix imu_link_rot_in_odom_frame not available")
            self.a_drone_odom = None

    def project_to_sphere(self, x, L):
        """ Project position to sphere and velocity to tangent space. """
        p = x[0:3].flatten()
        v = x[3:6].flatten()

        norm_p = np.linalg.norm(p)
        if norm_p > 1e-3:
            p_proj = p / norm_p * L
            # Project v to tangent of sphere at p_proj
            v_proj = v - (np.dot(v, p_proj) / (L ** 2)) * p_proj
        else:
            p_proj = p
            v_proj = v

        return np.concatenate((p_proj, v_proj)).flatten()

    def project_covariance(self, P, x, L):
        """ Project the covariance to the tangent space at position x[0:3]. """
        p = x[0:3].flatten()
        norm_p = np.linalg.norm(p)
        if norm_p < 1e-3:
            return P  # Avoid projection if position is near zero

        p_unit = p / norm_p
        T = np.eye(3) - np.outer(p_unit, p_unit)  # Tangent projector

        # Project block-wise
        P_proj = P.copy()
        P_proj[0:3, 0:3] = T @ P[0:3, 0:3] @ T.T
        P_proj[0:3, 3:6] = T @ P[0:3, 3:6]
        P_proj[3:6, 0:3] = P_proj[0:3, 3:6].T
        P_proj[3:6, 3:6] = P[3:6, 3:6]  # Leave velocity covariance as is

        return P_proj
        
    #"""
    # PREVIOUS KALMAN FILTER: PROBLEM WITHY THE Z AXIS
    def fx(self, x, dt, a_drone):
        r = x[0:3]
        v = x[3:6]
        
        r_norm = np.linalg.norm(r)
        r_hat = r / r_norm
        g_eff = self.g_vec - a_drone

        acc = - (np.dot(v, v) + np.dot(g_eff, r_hat)) * r_hat / self.target_length
        
        r_new = r + v * dt
        v_new = v + acc * dt
        return np.hstack([r_new, v_new])

    def F_jacobian(self, x, dt, a_drone):
        r = x[0:3]
        v = x[3:6]
        r_norm = np.linalg.norm(r)
        r_hat = r / r_norm
        g_eff = self.g_vec - a_drone

        I3 = np.eye(3)
        zeros = np.zeros((3, 3))

        d_acc_dr = (
            - (np.outer(g_eff, r_hat) + np.dot(g_eff, r_hat) * (np.eye(3) - np.outer(r_hat, r_hat)) / r_norm)
            - (2 * np.outer(v, v) / r_norm)
        ) / self.target_length

        d_acc_dv = -2 * np.outer(r_hat, v) / self.target_length

        F = np.block([
            [zeros, I3],
            [d_acc_dr, d_acc_dv]
        ])

        return np.eye(6) + F * dt

    def hx(self, x):
        return x[0:3]  # we observe position only

    def update_ekf_constrained(self, z_meas_pos, a_drone, dt):
        # ===== EKF PREDICTION =====
        self.ekf.F = self.F_jacobian(self.ekf.x, dt, a_drone)
        self.ekf.x = self.fx(self.ekf.x, dt, a_drone)
        self.ekf.P = self.ekf.F @ self.ekf.P @ self.ekf.F.T + self.ekf.Q

        # ===== PROJECT TO SPHERE (optional but recommended) =====
        self.ekf.x = self.project_to_sphere(self.ekf.x, self.target_length)
        self.ekf.P = self.project_covariance(self.ekf.P, self.ekf.x, self.target_length)

        # ===== EKF UPDATE =====
        H = np.hstack((np.eye(3), np.zeros((3, 3))))
        z = z_meas_pos.flatten()
        hx = self.hx(self.ekf.x).flatten()
        y = z - hx
        S = H @ self.ekf.P @ H.T + self.ekf.R
        K = self.ekf.P @ H.T @ np.linalg.inv(S)
        self.ekf.x = self.ekf.x + K @ y
        self.ekf.P = (np.eye(6) - K @ H) @ self.ekf.P

        # ===== PROJECT AGAIN (after update) =====
        self.ekf.x = self.project_to_sphere(self.ekf.x, self.target_length)
        self.ekf.P = self.project_covariance(self.ekf.P, self.ekf.x, self.target_length)

        # Store the filtered position estimate
        self.ekf_Hook_pos_in_winch_frame = self.ekf.x[0:3].copy()
    #"""

    def _align_arrays(self, *arrays):
        """
        Filters and aligns arrays to have the same valid length, padding shorter arrays with NaN values.
        Assumes each array is a list of 3D vectors or timestamps.
        """
        cleaned = []

        for arr in arrays:
            valid_rows = []
            for item in arr:
                if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 3:
                    try:
                        valid_rows.append([float(x) for x in item])
                    except (ValueError, TypeError):
                        continue
                elif isinstance(item, (int, float)):  # probably a timestamp
                    valid_rows.append(float(item))
            cleaned.append(valid_rows)

        max_len = max(len(arr) for arr in cleaned)
        padded = []
        for arr in cleaned:
            while len(arr) < max_len:
                arr.append([np.nan, np.nan, np.nan] if isinstance(arr[0], list) else np.nan)
            padded.append(arr)

        return tuple(np.array(arr) for arr in padded)

    def save_plot_data(self, save_dir='plot_data'):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_dir, f'hook_base_link_{timestamp}.npz')

        try:
            np.savez(
                filename,
                timestamps=np.array(self.est_hook_timestamps),
                estimated=np.array(self.est_Hook_pos_in_base_link_frame_to_plot),
                ekf=np.array(self.ekf_Hook_pos_in_base_link_frame_to_plot)
            )
            #rospy.loginfo(f"Saved plot data to: {filename}")
	    rospy.loginfo("Saved plot data to: {}".format(filename))
        except Exception as e:
            #rospy.logerr(f"Failed to save plot data: {e}")
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
