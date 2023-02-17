import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata


def deform_frame(frame, num_pts=5, std_displacement=20, mean_displacement=0):

    """
    Warp an image, using the method outlined in
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    :param frame: 2D numpy array with image
    :param num_pts: The number of points along a dimension to displace to determine warp
    :param mean_displacement: Mean of displacement for these points (random, gaussian distributed displacement)
    :param std_displacement: Standard deviation of displacement for these points
    :return: warped frame image with the same dimensions
    """

    # get the displacement values; x, y
    def_grid = np.random.randn(num_pts, num_pts, 2) * std_displacement + mean_displacement

    # get the coordinates of the frame
    x, y = np.meshgrid(np.arange(0, frame.shape[1]), np.arange(0, frame.shape[0]))

    # get the locations of the displacement values
    locations_x = np.linspace(0, frame.shape[1], num_pts)
    locations_y = np.linspace(0, frame.shape[0], num_pts)
    warped_x, warped_y = np.meshgrid(locations_x, locations_y)
    orig_x = warped_x + def_grid[:, :, 1]
    orig_y = warped_y + def_grid[:, :, 0]

    # Interpolate the warp coordinates on the image grid
    grid_z = griddata(np.concatenate((np.expand_dims(orig_x.flatten(), axis=1),
                                      np.expand_dims(orig_y.flatten(), axis=1)), axis=1),
                      np.concatenate((np.expand_dims(warped_x.flatten(), axis=1),
                                      np.expand_dims(warped_y.flatten(), axis=1)), axis=1),
                      (x, y),
                      method='cubic')
    # separate back into x and y
    map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(x.shape)
    map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(x.shape)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    # map the image to the new coordinates
    deformed_frame = cv2.remap(frame, map_y_32, map_x_32, cv2.INTER_CUBIC)

    return deformed_frame


if __name__ == '__main__':

    frame_path = r'\Users\rmillin\test.jpg'
    frame = cv2.imread(frame_path)
    frame = frame[:, :, 0]
    deformed_frame = deform_frame(frame)
    plt.imshow(frame)
    plt.show()
    plt.imshow(deformed_frame)
    plt.show()
