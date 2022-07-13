import numpy as np
import pandas as pd
from cv2 import calibrateCamera, Rodrigues, CALIB_USE_INTRINSIC_GUESS

# constants for AI deck Himax camera and Frontnet image dataset
# field of view retrieved from datasheet
FIELD_OF_VIEW_X = FIELD_OF_VIEW_Y = 87.0
# images are of shape (96, 160), principal point is therefore:
C_X = 48.     
C_Y = 80.


def read_csv_file(path):
    """
    Function to read a .csv file from the given path
    Parameters:
        ----------
        path: String, path to the .csv file
    Returns:
        a pandas dataframe containing the data from the .csv file

    For camera calibration, this .csv file should include the 3D coordinates 
    of certain points in camera frame and the corresponding 2D pixel coordinates in the images.
    Column names for 3D coordinates: 'x', 'y', 'z'
    Column names for 2D coordinates: 'img_x', 'img_y'
    """                       
    return pd.read_csv(path)

def get_3d_coords(dataframe, column_x='x', column_y='y', column_z='z'):
    """
    Function to get the 3D coordinates from the pandas dataframe.
    Parameters:
        ----------
        dataframe: pandas dataframe
        column_x: optional, String, name of the column containing the x coordinates of the 3D points
        column_y: optional, String, name of the column containing the y coordinates of the 3D points
        column_z: optional, String, name of the column containing the z coordinates of the 3D points
    Returns:
        a (n, 3)-shaped numpy array including all 3D coordinates from the dataframe
    """
    return np.array([dataframe[column_x], dataframe[column_y], dataframe[column_z]])

def get_2d_coords(dataframe, column_x='img_x', column_y='img_y'):
    """
    Function to get the 2D coordinates from the pandas dataframe.
    Parameters:
        ----------
        dataframe: pandas dataframe
        column_x: optional, String, name of the column containing the x coordinates of the 2D points
        column_y: optional, String, name of the column containing the y coordinates of the 2D points
    Returns:
        a (n, 2)-shaped numpy array including all 3D coordinates from the dataframe
    """
    return np.array([dataframe[column_x], dataframe[column_y]])


def calc_focal_lengths_from_fov(fov_x, fov_y, c_x, c_y):
    """
    Function to calculate the focal lengths for the camera intrinsics matrix from field of view.
    Parameters:
        ----------
        fov_x: field of view in x direction
        fov_y: field of view in y direction
        c_x: principal point x coordinate
        c_y: principal point y coordinate
    Returns:
        focal length in x and focal length in y direction
    """
    if fov_y is None:                   # if fov_y is not set, it is assumed it is equal to fov_x
        fov_y = fov_x

    focal_length_x = c_x * np.tan(np.degrees(fov_x / 2.))  
    focal_length_y = c_y * np.tan(np.degrees(fov_y / 2.))

    return focal_length_x, focal_length_y


def initial_guess_camera_matrix(fov_x=FIELD_OF_VIEW_Y, fov_y=FIELD_OF_VIEW_Y, c_x=C_X, c_y=C_Y):
    """
    Construct an initial camera intrinsics matrix from given field of view information 
    and principal point.
    Parameters:
        ----------
        fov_x: field of view in x direction
        fov_y: field of view in y direction
        c_x: principal point x coordinate
        c_y: principal point y coordinate
    Returns:
        a (3,3) numpy array, the initial guess of the camera intrinsics matrix
    """

    focal_length_x, focal_length_y = calc_focal_lengths_from_fov(fov_x, fov_y, c_x, c_y)

    matrix = np.zeros((3,3))
    matrix[0,0] = focal_length_x
    matrix[1,1] = focal_length_y
    
    matrix[0,2] = c_x
    matrix[1,2] = c_y

    matrix[2,2] = 1.

    return matrix

def calibrate_camera(points_3d, points_2d, initial_guess, img_size=[96., 160.]):
    """
    Wrapper for the cv2 calibrateCamera() function.
    Parameters:
        ----------
        points_3d: numpy array of shape (n, 3), 3D coordinates from known objects
        points_2d: numpy array of shape (n, 2), 2D coordinates of known objects in images
        initial_guess: numpy array of shape (3,3), cv2 needs an initial guess if the objects are not on a plane 
        If your objects are on a plane, you can set it initial_guess = None and remove the flag
        img_size: tuple, height and width of the images
    Returns:
        camera_matrix: a (3,3) numpy array, the calculated camera intrinsics matrix
        rvecs: a (3,1) numpy array, the rotation vector
        tvecs: a (3,1) numpy array, the translation vector
        -> both vectors are needed for generating the transformation matrix, see function get_transformation_matrix()
        dist_coeffs: a (n,) numpy array, with n distortion coefficients
        error: the error for the calculated matrix, lower is better
    """
    print(img_size)
    error, camera_matrix, dist_coeffs, rvecs, tvecs = calibrateCamera(points_3d, points_2d, img_size, initial_guess, None, flags=CALIB_USE_INTRINSIC_GUESS)

    return camera_matrix, rvecs[0], tvecs[0], dist_coeffs[0], error


if __name__ == "__main__":
    path = '~/Documents/Coding/adversarial_frontnet/adversarial_frontnet/camera_intrinsics/ground_truth_pose.csv'
    init_guess = initial_guess_camera_matrix()

    df = read_csv_file(path)
    points_3d = get_3d_coords(df, ' pred_x', ' pred_y', ' pred_z')
    points_2d = get_2d_coords(df, ' img_x', ' img_y')

    print(calibrate_camera(points_3d, points_2d, init_guess))