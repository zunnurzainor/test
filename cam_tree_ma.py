import rclpy
import cv2
import numpy as np
import math

from rclpy.node import Node 
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from math import atan2,sin,cos,tan,inf,sqrt
from sklearn.metrics import r2_score

import configparser
import numpy as np
from pathlib import Path

# ============================================================
# LOAD CONFIGURATION
# ============================================================

CONFIG_FILE = "/home/orin_nano/ros2_ws/src/robot_action_python/robot_action_python/config/tree_detection.config"

config = configparser.ConfigParser()
config.read(CONFIG_FILE)


# ============================================================
# TREE DETECTION PARAMETERS
# ============================================================

MIN_DEPTH = config.getfloat("TreeDetection", "MIN_DEPTH")
MAX_DEPTH = config.getfloat("TreeDetection", "MAX_DEPTH")

TREEWIDTH_PIXEL = config.getint(
    "TreeDetection",
    "TREEWIDTH_PIXEL"
)

PIXEL_THRESHOLD = config.getfloat(
    "TreeDetection",
    "PIXEL_THRESHOLD"
)


# ============================================================
# HSV PARAMETERS
# ============================================================

LOWER1 = np.array([
    int(x) for x in config.get("TreeDetection", "LOWER1").split(",")
])

UPPER1 = np.array([
    int(x) for x in config.get("TreeDetection", "UPPER1").split(",")
])

LOWER2 = np.array([
    int(x) for x in config.get("TreeDetection", "LOWER2").split(",")
])

UPPER2 = np.array([
    int(x) for x in config.get("TreeDetection", "UPPER2").split(",")
])

MA_WINDOW_SIZE  = 5

# ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i  resolution:=HD720  publish_tf:=true  positional_tracking:=true  enable_mapping:=false  imu_fusion:=true
# ros2 run cam_test cam_tree

class Tree_identification(Node):

    def __init__(self):
        # initialize node name
        super().__init__('Tree_Identification')

        # subscript selected rostopic
        self.subs1 = self.create_subscription(
            CompressedImage,
            '/zed/zed_node/rgb/color/rect/image/compressed',
            self.rgbimage_callback,
            10
        )
        self.subs2 = self.create_subscription(
            Image,
            '/zed/zed_node/depth/depth_registered',
            self.depthimage_callback,
            10
        )

        self.subs3 = self.create_subscription(
                    PoseStamped,
                    '/zed/zed_node/pose',
                    self.robot_pose,
                    10
                )
        

        ### 
        # For subscription robot position rostopic
        ###

        # Declare important variable
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.x = self.y = self.z = self.roll = self.pitch = self.yaw = 0
        self.min_depth = MIN_DEPTH # inner border (m)
        self.max_depth = MAX_DEPTH # outer border (m)
        self.center_line = int(540/2) # frame size 960 x 540
        self.treewidth_pixel = TREEWIDTH_PIXEL
        self.thresh_diff = 1 # zone around tree (m)
        self.pixel_threshold = PIXEL_THRESHOLD # H_score
        self.ma_window_size = MA_WINDOW_SIZE  # moving average window size (odd number recommended)
        
        # Camera Intrinsic matrix
        #self.fx = 972.1294555664062
        #self.fy = 972.1294555664062
        #self.cx = 481.70263671875
        #self.cy = 270.33123779296875

        # Camera Intrinsic matrix (from echo)
        self.fx = 527.5696411132812
        self.fy = 527.5696411132812
        self.cx = 499.6092529296875
        self.cy = 284.9339599609375

        # Canny edge parameter
        self.canny_thresh = ([100, 200])

        # Define red color ranges in HSV (Cone)
        self.lower_1 = LOWER1
        self.upper_1 = UPPER1
        self.lower_2 = LOWER2
        self.upper_2 = UPPER2

        # Bright orange-red cone
        #self.lower_1 = np.array([0,  150, 150])   
        #self.upper_1 = np.array([15, 255, 255])   
        #self.lower_2 = np.array([15, 150, 150])   
        #self.upper_2 = np.array([25, 255, 255])   

        # Define brown color ranges in HSV (Tree)
        #self.lower_1 = np.array([5, 30, 40])
        #self.upper_1 = np.array([25, 100, 80])
        #self.lower_2 = np.array([0, 10, 50])
        #self.upper_2 = np.array([20, 50, 90])

        # Define brown color (for dark brown-grey trunk):
        #self.lower_1 = np.array([10,  20,  30])
        #self.upper_1 = np.array([30,  80, 100])
        #self.lower_2 = np.array([40,  10,  40])
        #Self.upper_2 = np.array([80,  50, 120])

        # Define brown color (18/03/2026):
        #self.lower_1 = np.array([0,  25,  100])
        #self.upper_1 = np.array([179,  70, 230])
        #self.lower_2 = np.array([10,  55,  40])
        #self.upper_2 = np.array([22,  160, 160])

        # Wider HSV for dark palm trunk
        #self.lower_1 = np.array([0,   0,   10])
        #self.upper_1 = np.array([40,  60, 100])
        #self.lower_2 = np.array([0,   0,   80])
        #self.upper_2 = np.array([40,  50, 160])

        # New HSV range (28/03/26)
        #self.lower_1 = np.array([0,  15,  20])
        #self.upper_1 = np.array([25, 80,  90])
        #self.lower_2 = np.array([0,  10,  80])
        #self.upper_2 = np.array([30, 55, 130])

        # Pose of robot relative to map/odom
        self.MvC = np.array([
            [self.x],
            [self.y],
            [self.z]
        ])
        self.MrC = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        self.TrW = np.array([
                    [self.x],
                    [self.y],
                    [self.z]
                ])
        
        self.RrW = np.array([
            [math.cos(self.robot_yaw), -math.sin(self.robot_yaw), 0.0],
            [math.sin(self.robot_yaw), math.cos(self.robot_yaw), 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Notify Node activation
        self.get_logger().info("Tree Identification Node Started")

    def rgbimage_callback(self, msg):
        try:
            self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

        except Exception as e:
            self.get_logger().error("Masalah process RGB image: %s", str(e))

    def depthimage_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error("Masalah process Depth image: %s", str(e))

    def robot_pose(self, msg):
            self.x = msg.pose.position.x
            self.y = msg.pose.position.y
            self.z = 0.0
    
             # Quaternion
            rot_q = msg.pose.orientation
    
            # Quaternion -> yaw
            siny_cosp = 2 * (rot_q.w * rot_q.z + rot_q.x * rot_q.y)
            cosy_cosp = 1 - 2 * (rot_q.y**2 + rot_q.z**2)
    
            self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
            # self.robot_yaw = msg.pose.orientation.z
            
            self.TrW = np.array([
                [self.x],
                [self.y],
                [self.z]
            ])
    
            self.RrW = np.array([
                [math.cos(self.robot_yaw), -math.sin(self.robot_yaw), 0.0],
                [math.sin(self.robot_yaw), math.cos(self.robot_yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])

    def calculated_pixel_ratio(self, mask):
        # Count non-zero pixels
        color_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]

        # avoid division by zero
        if total_pixels == 0:
            return 0.0
        
        ratio = color_pixels / total_pixels
        return ratio

    def getpoint(self, d, theta):
        xt= d*cos(theta)
        yt= d*sin(theta)
        zt= 0
        Cvt = np.array([
            [xt],
            [yt],
            [zt]
        ])
        Mvt= self.MvC+np.dot(self.MrC,Cvt)  
        return [Mvt[0][0],Mvt[1][0],Mvt[2][0]]

    def process_image(self, color_image, depth_image, drawing):
        detect_l = 0
        detect_r = 0
        tree = 0
        tree_temp = []
        tree_data = []

        if color_image is not None and depth_image is not None:

            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            # ---------------------------------------------------------
            # DEPTH MASKING
            # ---------------------------------------------------------
            depth_mask = cv2.inRange(
                depth_image,
                self.min_depth,
                self.max_depth
            )

            masked_depth = cv2.bitwise_and(
                depth_image,
                depth_image,
                mask=depth_mask
            )

            depth_8u = cv2.convertScaleAbs(
                masked_depth,
                alpha=25
            )

            # ---------------------------------------------------------
            # GET DEPTH VALUES AT CENTER LINE
            # ---------------------------------------------------------
            center_row = depth_8u[self.center_line, :]

            # ---------------------------------------------------------
            # MOVING AVERAGE FILTER
            # ---------------------------------------------------------
            # Smooth the depth signal before calculating differences.
            # This helps reduce noise and false edge detections.
            ma_kernel = np.ones(self.ma_window_size) / self.ma_window_size

            center_row_smooth = np.convolve(
                center_row.astype(np.float32),
                ma_kernel,
                mode='same'
            )

            # ---------------------------------------------------------
            # CALCULATE PIXEL-TO-PIXEL DEPTH DIFFERENCE
            # ---------------------------------------------------------
            diffs = np.diff(center_row_smooth)

            # Colored depth image
            depth_colored = cv2.applyColorMap(
                depth_8u,
                cv2.COLORMAP_JET
            )

            # ---------------------------------------------------------
            # INITIALIZE DETECTION VARIABLES
            # ---------------------------------------------------------
            start_idx = None

            roi_l_temp = None
            roi_r_temp = None

            bounding_boxes = []
            distance_boxes = []

            # ---------------------------------------------------------
            # SEARCH FOR OBJECT / TREE EDGES
            # ---------------------------------------------------------
            for i in range(1, len(diffs)):

                # -----------------------------------------------------
                # LEFT EDGE
                # -----------------------------------------------------
                if diffs[i] > self.thresh_diff:

                    start_idx = i

                # -----------------------------------------------------
                # RIGHT EDGE
                # -----------------------------------------------------
                elif diffs[i] < -self.thresh_diff:

                    end_idx = i

                    if start_idx is not None and end_idx is not None:

                        # -------------------------------------------------
                        # CHECK TREE WIDTH IN PIXELS
                        # -------------------------------------------------
                        if abs(start_idx - end_idx) > self.treewidth_pixel:

                            bounding_boxes.append(
                                (start_idx, end_idx)
                            )

                            # Get depth at left and right edges
                            start_depth = masked_depth[
                                self.center_line,
                                start_idx + 1
                            ]

                            end_depth = masked_depth[
                                self.center_line,
                                end_idx
                            ]

                            distance_boxes.append(
                                (start_depth, end_depth)
                            )

                            # =================================================
                            # LEFT ROI
                            # =================================================
                            roi_l = [
                                start_idx - 30,
                                start_idx + 20,
                                self.center_line - 200,
                                self.center_line + 200
                            ]

                            roi_l_temp = color_image[
                                roi_l[2]:roi_l[3],
                                roi_l[0]:roi_l[1]
                            ]

                            if roi_l_temp.size == 0:

                                pass

                            else:

                                # Convert ROI to grayscale
                                roi_l_gray = cv2.cvtColor(
                                    roi_l_temp,
                                    cv2.COLOR_BGR2GRAY
                                )

                                # Canny edge detection
                                roi_l_edge = cv2.Canny(
                                    roi_l_gray,
                                    self.canny_thresh[0],
                                    self.canny_thresh[1]
                                )

                                # Crop depth mask
                                roi_l_depth_mask = depth_mask[
                                    roi_l[2]:roi_l[3],
                                    roi_l[0]:roi_l[1]
                                ]

                                # Filter Canny edges using depth
                                roi_l_filtered_edge = cv2.bitwise_and(
                                    roi_l_edge,
                                    roi_l_depth_mask
                                )

                                # Get edge points
                                roi_l_pointcloud = np.column_stack(
                                    np.where(
                                        roi_l_filtered_edge > 0
                                    )
                                )

                                detect_l = 1

                            # =================================================
                            # RIGHT ROI
                            # =================================================
                            roi_r = [
                                end_idx - 30,
                                end_idx + 20,
                                self.center_line - 200,
                                self.center_line + 200
                            ]

                            roi_r_temp = color_image[
                                roi_r[2]:roi_r[3],
                                roi_r[0]:roi_r[1]
                            ]

                            if roi_r_temp.size == 0:

                                pass

                            else:

                                # Convert ROI to grayscale
                                roi_r_gray = cv2.cvtColor(
                                    roi_r_temp,
                                    cv2.COLOR_BGR2GRAY
                                )

                                # Canny edge detection
                                roi_r_edge = cv2.Canny(
                                    roi_r_gray,
                                    self.canny_thresh[0],
                                    self.canny_thresh[1]
                                )

                                # Crop depth mask
                                roi_r_depth_mask = depth_mask[
                                    roi_r[2]:roi_r[3],
                                    roi_r[0]:roi_r[1]
                                ]

                                # Filter Canny edges using depth
                                roi_r_filtered_edge = cv2.bitwise_and(
                                    roi_r_edge,
                                    roi_r_depth_mask
                                )

                                # Get edge points
                                roi_r_pointcloud = np.column_stack(
                                    np.where(
                                        roi_r_filtered_edge > 0
                                    )
                                )

                                detect_r = 1

                            # =================================================
                            # IF BOTH LEFT AND RIGHT EDGES ARE DETECTED
                            # =================================================
                            if detect_l and detect_r:

                                roi_c = [
                                    start_idx - 30,
                                    end_idx + 30,
                                    self.center_line - 200,
                                    self.center_line + 200
                                ]

                                roi_c_temp = color_image[
                                    roi_c[2]:roi_c[3],
                                    roi_c[0]:roi_c[1]
                                ]

                                if roi_c_temp.size == 0:

                                    pass
                                    print("1")

                                else:

                                    # -------------------------------------------------
                                    # HSV COLOR CHECK
                                    # -------------------------------------------------
                                    hsv = cv2.cvtColor(
                                        roi_c_temp,
                                        cv2.COLOR_BGR2HSV
                                    )

                                    # Create two HSV masks
                                    mask1 = cv2.inRange(
                                        hsv,
                                        self.lower_1,
                                        self.upper_1
                                    )

                                    mask2 = cv2.inRange(
                                        hsv,
                                        self.lower_2,
                                        self.upper_2
                                    )

                                    # Combine masks
                                    color_mask = cv2.bitwise_or(
                                        mask1,
                                        mask2
                                    )

                                    # Calculate percentage of valid pixels
                                    pixel = self.calculated_pixel_ratio(
                                        color_mask
                                    )

                                    # =================================================
                                    # TREE GEOMETRY CALCULATION
                                    # =================================================

                                    # Angle to left edge of tree
                                    alpha1 = atan2(
                                        start_idx - self.cx,
                                        self.fx
                                    )

                                    # Angle to right edge of tree
                                    alpha2 = atan2(
                                        end_idx - self.cx,
                                        self.fx
                                    )

                                    # Total angular width of tree
                                    beta = alpha2 - alpha1

                                    # Angle to center of tree
                                    tree_alpha = alpha1 + beta * 0.5

                                    # Pixel column corresponding to tree center
                                    tree_X = int(
                                        self.fx * tan(tree_alpha)
                                        + self.cx
                                    )

                                    # Radial distance from camera to
                                    # front surface of tree
                                    tree_r = (
                                        masked_depth[
                                            self.center_line
                                        ][tree_X]
                                        / cos(tree_alpha)
                                    )

                                    # -------------------------------------------------
                                    # TREE DIAMETER
                                    # -------------------------------------------------
                                    tree_diameter = (
                                        (2 * sin(beta * 0.5))
                                        /
                                        (1 - sin(beta * 0.5))
                                    ) * tree_r

                                    # -------------------------------------------------
                                    # TREE RADIUS
                                    # -------------------------------------------------
                                    tree_radius = 0.5 * tree_diameter

                                    # -------------------------------------------------
                                    # DISTANCE FROM CAMERA TO TREE CENTER
                                    # -------------------------------------------------
                                    tree_distance = (
                                        tree_r
                                        + 0.5 * tree_diameter
                                    )

                                    # =================================================
                                    # TREE DETECTION USING PIXEL THRESHOLD
                                    # =================================================
                                    if pixel > self.pixel_threshold:
                                        tree = 1
                                    else:
                                        tree = 0

                                    # =================================================
                                    # TREE CONFIRMED
                                    # =================================================
                                    if tree:

                                        # -------------------------------------------------
                                        # GET TREE POSITION
                                        # -------------------------------------------------
                                        tree_temp = self.getpoint(
                                            d=tree_distance,
                                            theta=tree_alpha
                                        )

                                        # -------------------------------------------------
                                        # STORE TREE DATA
                                        #
                                        # tree_temp     = XYZ position
                                        # tree_distance = camera -> tree center
                                        # tree_alpha    = tree angle
                                        # tree_diameter = tree diameter
                                        # -------------------------------------------------
                                        tree_data.append(
                                            (
                                                tree_temp,
                                                tree_distance,
                                                tree_alpha,
                                                tree_diameter
                                            )
                                        )

                                        # =================================================
                                        # DRAWING / VISUALIZATION
                                        # =================================================
                                        if drawing:

                                            # -------------------------------------------------
                                            # TREE DIAMETER
                                            # -------------------------------------------------
                                            cv2.putText(
                                                color_image,
                                                f"tree_d(m): {tree_diameter:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 250
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 0, 0),
                                                5
                                            )

                                            cv2.putText(
                                                color_image,
                                                f"tree_d(m): {tree_diameter:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 250
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 255, 0),
                                                2
                                            )

                                            # -------------------------------------------------
                                            # TREE RADIUS
                                            # -------------------------------------------------
                                            cv2.putText(
                                                color_image,
                                                f"tree_r(m): {tree_radius:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 220
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 0, 0),
                                                5
                                            )

                                            cv2.putText(
                                                color_image,
                                                f"tree_r(m): {tree_radius:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 220
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 255, 0),
                                                2
                                            )

                                            # -------------------------------------------------
                                            # TREE XYZ
                                            # -------------------------------------------------
                                            cv2.putText(
                                                color_image,
                                                f"xyz(m): "
                                                f"{tree_temp[0]:.2f}, "
                                                f"{tree_temp[1]:.2f}, "
                                                f"{tree_temp[2]:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 190
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 0, 0),
                                                5
                                            )

                                            cv2.putText(
                                                color_image,
                                                f"xyz(m): "
                                                f"{tree_temp[0]:.2f}, "
                                                f"{tree_temp[1]:.2f}, "
                                                f"{tree_temp[2]:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 190
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 255, 0),
                                                2
                                            )

                                            # -------------------------------------------------
                                            # RADIAL DISTANCE TO TREE SURFACE
                                            # -------------------------------------------------
                                            cv2.putText(
                                                color_image,
                                                f"dist cam-tree: {tree_r:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 160
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 0, 0),
                                                5
                                            )

                                            cv2.putText(
                                                color_image,
                                                f"dist cam-tree: {tree_r:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 160
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 255, 0),
                                                2
                                            )

                                            # -------------------------------------------------
                                            # TREE CENTER DISTANCE
                                            # -------------------------------------------------
                                            cv2.putText(
                                                color_image,
                                                f"d: {tree_distance:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 130
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 0, 0),
                                                5
                                            )

                                            cv2.putText(
                                                color_image,
                                                f"d: {tree_distance:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 130
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 255, 0),
                                                2
                                            )

                                            # -------------------------------------------------
                                            # TREE ANGLE
                                            # -------------------------------------------------
                                            cv2.putText(
                                                color_image,
                                                f"alpha: {tree_alpha:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 100
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 0, 0),
                                                5
                                            )

                                            cv2.putText(
                                                color_image,
                                                f"alpha: {tree_alpha:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 100
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 255, 0),
                                                2
                                            )

                                            # -------------------------------------------------
                                            # PIXEL SCORE
                                            # -------------------------------------------------
                                            cv2.putText(
                                                color_image,
                                                f"Pixel Score: {pixel:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 70
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 0, 0),
                                                5
                                            )

                                            cv2.putText(
                                                color_image,
                                                f"Pixel Score: {pixel:.2f}",
                                                (
                                                    roi_r[0] - 380,
                                                    self.center_line - 70
                                                ),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (255, 255, 0),
                                                2
                                            )

                                            # -------------------------------------------------
                                            # TREE WIDTH ARROW
                                            # -------------------------------------------------
                                            cv2.arrowedLine(
                                                color_image,
                                                (
                                                    start_idx,
                                                    self.center_line
                                                ),
                                                (
                                                    end_idx,
                                                    self.center_line
                                                ),
                                                color=[255, 0, 255],
                                                thickness=2,
                                                line_type=cv2.LINE_8
                                            )

                                            cv2.arrowedLine(
                                                color_image,
                                                (
                                                    end_idx,
                                                    self.center_line
                                                ),
                                                (
                                                    start_idx,
                                                    self.center_line
                                                ),
                                                color=[255, 0, 255],
                                                thickness=2,
                                                line_type=cv2.LINE_8
                                            )

                                            # =================================================
                                            # LEFT EDGE ROI VISUALIZATION
                                            # =================================================
                                            cv2.rectangle(
                                                color_image,
                                                (
                                                    roi_l[0],
                                                    roi_l[2]
                                                ),
                                                (
                                                    roi_l[1],
                                                    roi_l[3]
                                                ),
                                                (0, 0, 255),
                                                2
                                            )

                                            if roi_l_pointcloud.shape[0] >= 2:

                                                y_coords_l = (
                                                    roi_l_pointcloud[:, 0]
                                                )

                                                x_coords_l = (
                                                    roi_l_pointcloud[:, 1]
                                                )

                                                roi_l_coef = np.polyfit(
                                                    y_coords_l,
                                                    x_coords_l,
                                                    1
                                                )

                                                roi_l_fit = np.polyval(
                                                    roi_l_coef,
                                                    y_coords_l
                                                )

                                                roi_l_r2 = r2_score(
                                                    x_coords_l,
                                                    roi_l_fit
                                                )

                                                # Draw fitted line
                                                pt1 = (
                                                    int(
                                                        roi_l_coef[0] * 0
                                                        + roi_l_coef[1]
                                                    ),
                                                    0
                                                )

                                                pt2 = (
                                                    int(
                                                        roi_l_coef[0]
                                                        * (
                                                            roi_l[3]
                                                            - roi_l[2]
                                                        )
                                                        + roi_l_coef[1]
                                                    ),
                                                    roi_l[3] - roi_l[2]
                                                )

                                                cv2.line(
                                                    roi_l_temp,
                                                    pt1,
                                                    pt2,
                                                    color=(255, 0, 127),
                                                    thickness=2
                                                )

                                            else:

                                                continue

                                            # =================================================
                                            # RIGHT EDGE ROI VISUALIZATION
                                            # =================================================
                                            cv2.rectangle(
                                                color_image,
                                                (
                                                    roi_r[0],
                                                    roi_r[2]
                                                ),
                                                (
                                                    roi_r[1],
                                                    roi_r[3]
                                                ),
                                                (255, 0, 0),
                                                2
                                            )

                                            if roi_r_pointcloud.shape[0] >= 2:

                                                y_coords_r = (
                                                    roi_r_pointcloud[:, 0]
                                                )

                                                x_coords_r = (
                                                    roi_r_pointcloud[:, 1]
                                                )

                                                roi_r_coef = np.polyfit(
                                                    y_coords_r,
                                                    x_coords_r,
                                                    1
                                                )

                                                roi_r_fit = np.polyval(
                                                    roi_r_coef,
                                                    y_coords_r
                                                )

                                                roi_r_r2 = r2_score(
                                                    x_coords_r,
                                                    roi_r_fit
                                                )

                                                # Draw fitted line
                                                pt1 = (
                                                    int(
                                                        roi_r_coef[0] * 0
                                                        + roi_r_coef[1]
                                                    ),
                                                    0
                                                )

                                                pt2 = (
                                                    int(
                                                        roi_r_coef[0]
                                                        * (
                                                            roi_r[3]
                                                            - roi_r[2]
                                                        )
                                                        + roi_r_coef[1]
                                                    ),
                                                    roi_r[3] - roi_r[2]
                                                )

                                                cv2.line(
                                                    roi_r_temp,
                                                    pt1,
                                                    pt2,
                                                    color=(255, 0, 127),
                                                    thickness=2
                                                )

                                            else:

                                                continue

                                    # =========================================================
                                    # OBJECT IS NOT A TREE
                                    # =========================================================
                                    else:

                                        cv2.putText(
                                            color_image,
                                            "Not Tree....",
                                            (
                                                roi_r[0] - 100,
                                                self.center_line - 230
                                            ),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (0, 0, 0),
                                            5
                                        )

                                        cv2.putText(
                                            color_image,
                                            "Not Tree",
                                            (
                                                roi_r[0] - 100,
                                                self.center_line - 230
                                            ),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (255, 255, 0),
                                            2
                                        )

                                        # Pixel score
                                        cv2.putText(
                                            color_image,
                                            f"Pixel Score: {pixel:.2f}",
                                            (
                                                roi_r[0] - 100,
                                                self.center_line - 210
                                            ),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (0, 0, 0),
                                            5
                                        )

                                        cv2.putText(
                                            color_image,
                                            f"Pixel Score: {pixel:.2f}",
                                            (
                                                roi_r[0] - 100,
                                                self.center_line - 210
                                            ),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (255, 255, 0),
                                            2
                                        )

                            # -----------------------------------------------------
                            # RESET START INDEX
                            # -----------------------------------------------------
                            else:

                                pass

                            start_idx = None

                        # =============================================================
                        # TREE WIDTH DID NOT PASS THRESHOLD
                        # =============================================================
                        else:

                            start_idx = None

                    # =============================================================
                    # START / END INDEX DOES NOT EXIST
                    # =============================================================
                    else:

                        pass

                # =============================================================
                # DRAW ALL DETECTED BOUNDING BOXES
                # =============================================================
                for i in range(len(bounding_boxes)):

                    start, end = bounding_boxes[i]

                    start_depth, end_depth = distance_boxes[i]

                    cv2.rectangle(
                        color_image,
                        (
                            start - 30,
                            self.center_line - 200
                        ),
                        (
                            end + 30,
                            self.center_line + 200
                        ),
                        (0, 255, 255),
                        2
                    )

                # =============================================================
                # CENTER LINE
                # =============================================================
                if drawing:

                    cv2.line(
                        color_image,
                        (0, self.center_line),
                        (1280, self.center_line),
                        (0, 255, 255),
                        1
                    )

        # =============================================================
        # RETURN
        # =============================================================
        return color_image, tree_data
    
def main(args=None):
    rclpy.init(args=args)
    node = Tree_identification()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.rgb_image is not None and node.depth_image is not None:
                processed = node.process_image(node.rgb_image.copy(), node.depth_image.copy(), draw_mode=1)
                cv2.imshow("ZED RGB Compressed", processed[0])
                cv2.waitKey(1)

    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    #rclpy.shutdown()

if __name__ == '__main__':
    main()