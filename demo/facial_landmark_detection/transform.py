import random

import cv2
import numpy as np


def calculate_pitch_yaw_roll(landmarks_2D,
                             cam_w=256,
                             cam_h=256,
                             radians=False):
    """ Return the the pitch yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """

    assert landmarks_2D is not None, 'landmarks_2D is None'

    # Estimated camera matrix values.
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    # dlib (68 landmark) trached points
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    # wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    # X-Y-Z with X pointing forward and Y on the left and Z up.
    # The X-Y-Z coordinates used are like the standard coordinates of ROS (robotic operative system)
    # OpenCV uses the reference usually used in computer vision:
    # X points to the right, Y down, Z to the front
    landmarks_3D = np.float32([
        [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT,
        [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT,
        [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
        [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
        [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
        [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
        [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
        [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
        [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
        [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
        [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
        [0.000000, -7.415691, 4.070434],  # CHIN
    ])
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)

    # Applying the PnP solver to find the 3D pose of the head from the 2D position of the landmarks.
    # retval - bool
    # rvec - Output rotation vector that, together with tvec, brings points from the world coordinate system to the camera coordinate system.
    # tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix,
                                 camera_distortion)
    # Get as input the rotational vector, Return a rotational matrix

    # const double PI = 3.141592653;
    # double thetaz = atan2(r21, r11) / PI * 180;
    # double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / PI * 180;
    # double thetax = atan2(r32, r33) / PI * 180;

    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    # euler_angles contain (pitch, yaw, roll)
    return tuple(map(lambda k: k[0], euler_angles))


def crop(image, landmark):
    x0, y0 = np.floor(landmark.min(0)).astype(int)
    x1, y1 = np.ceil(landmark.max(0)).astype(int)
    bbox = [x0, y0, x1, y1]
    landmark[:, 0] -= bbox[0]
    landmark[:, 1] -= bbox[1]
    image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    return image, landmark


def resize(image, size, landmark):
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, list):
        size = tuple(size)

    h, w, _ = image.shape
    new_w, new_h = size
    landmark[:, 0] *= new_w / w
    landmark[:, 1] *= new_h / h

    image = cv2.resize(image, size)
    return image, landmark


def random_horizontal_flip(image, landmark, mirror_indexes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = cv2.flip(image, 1)
        landmark[:, 0] = w - landmark[:, 0]
        landmark = landmark[mirror_indexes]
    return image, landmark


def random_rotate(image, landmark, angle_range):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    angle = random.choice(angle_range)
    rotation_matrix = cv2.getRotationMatrix2D(
        center, angle, 1).astype(np.float32)
    image = cv2.warpAffine(image, rotation_matrix, (w, h))
    landmark = rotation_matrix[:, :2].dot(landmark.T) + rotation_matrix[:, 2:]
    return image, landmark.T


def random_occlude(image, occlude_size):
    h, w, _ = image.shape
    occ_w, occ_h = occlude_size
    x = random.randint(0, w - occ_w)
    y = random.randint(0, h - occ_h)
    image[y:y+occ_h, x:x+occ_w, :] = 0
    return image


def normalize(image):
    image = image.astype(np.float32)
    image = image / 255.0
    return image


class Crop(object):
    def __call__(self, image, label):
        label['landmark'] = np.asarray(label['landmark'], dtype=np.float32).reshape((-1, 2))
        image, label['landmark'] = crop(image, label['landmark'])
        return image, label


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        image, label['landmark'] = resize(image, self.size, label['landmark'])
        label['size'] = self.size
        return image, label


class RandomHorizontalFlip(object):
    mirror_indexes = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34,
                      33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65]
    
    def __call__(self, image, label):
        image, label['landmark'] = random_horizontal_flip(image, label['landmark'], self.mirror_indexes)
        return image, label


class RandomRotate(object):
    def __init__(self, angle_range):
        self.angle_range = angle_range
        
    def __call__(self, image, label):
        image, label['landmark'] = random_rotate(image, label['landmark'], self.angle_range)
        return image, label


class RandomOcclude(object):
    def __init__(self, occlude_size):
        self.occlude_size = occlude_size
        
    def __call__(self, image, label):
        image = random_occlude(image, self.occlude_size)
        return image, label


class Normalize(object):
    def __call__(self, image, label):
        image = normalize(image)
        label['landmark'][:, 0] /= label['size'][0]
        label['landmark'][:, 1] /= label['size'][1]
        return image, label


class CalculateEulerAngles(object):
    tracked_points = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    
    def __call__(self, image, label):
        euler_angles = calculate_pitch_yaw_roll(label['landmark'][self.tracked_points])
        label['euler_angles'] = np.asarray(euler_angles, dtype=np.float32)
        return image, label


class ToTuple(object):
    def __call__(self, image, label):
        return image, (label['landmark'], label['euler_angles'])


class Compose(object):
    """Composes several transforms together.

    Parameters
    ----------
    transforms : list of 'transform' objects
        list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label
