import numpy as np
import math
import cv2

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip 34
    (0.0, -330.0, -65.0),        # Chin 9
    (-225.0, 170.0, -135.0),     # Left eye left corner 37
    (225.0, 170.0, -135.0),      # Right eye right corner 46
    (-150.0, -150.0, -125.0),    # Left Mouth corner 49
    (150.0, -150.0, -125.0)      # Right mouth corner 55
])

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def getHeadTiltAndCoords(size, image_points, frame_height):
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [
        0, focal_length, center[1]], [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                  camera_matrix, dist_coeffs, 
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    # Project a 3D point (0, 0 , 1000.0) onto the image plane
    (nose_end_point2D, _) = cv2.projectPoints(np.array(
        [(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # Get rotation matrix from the rotation vector
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Calculate head tilt angle in degrees
    head_tilt_degree = abs(
        [-180] - np.rad2deg([rotationMatrixToEulerAngles(rotation_matrix)[0]]))

    # Calculate starting and ending points for the two lines for illustration
    starting_point = (int(image_points[0][0]), int(image_points[0][1]))
    ending_point = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    ending_point_alternate = (ending_point[0], frame_height // 2)

    return head_tilt_degree, starting_point, ending_point, ending_point_alternate