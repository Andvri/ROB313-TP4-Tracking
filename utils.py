import cv2
import numpy as np
import math

RATE = .85
INTERVAL_SIZE = 180
INTERVAL = (2*math.pi)/INTERVAL_SIZE


def get_shapes(img_gray, show = False):
    x ,y = np.gradient(img_gray)
    orientation = np.arctan2(y, x)
    norm = np.hypot(y, x)

    threshold = np.quantile(norm, q=RATE)
    mask = np.where(norm > threshold, 1, 0)

    if show:
        mask_ = np.dstack((
            np.where(mask, orientation, 0),
            np.where(mask, orientation, 0),
            np.where(mask, orientation, 255)
        ))  # stacks 3 h x w arrays -> h x w x 3
    else:
        mask_ = mask

    
    return orientation, norm, mask_

def get_interval_number(pixel_orientation):
    return int((pixel_orientation+math.pi) // INTERVAL) - (0 if pixel_orientation < math.pi else 1)

def get_relative_coordinates(local_origin, local_center, elem_coor):
    return (
        local_center[0]- elem_coor[0] - local_origin[0],
        local_center[1]- elem_coor[1] - local_origin[1]
    )


def get_rtable(frame, origin, center):
    orientation, norm, mask = get_shapes(frame[:,:,2])
    
    rtable = {k: [] for k in range(INTERVAL_SIZE)}

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                id_ = get_interval_number(orientation[i, j])
                local_coor = get_relative_coordinates(origin, center, (j, i))
                rtable[id_].append(local_coor)

    return rtable



if __name__ == "__main__":
    img_path = './media/Input_TdH_Example.png'
    img = cv2.imread(img_path)
    get_rtable(img, 1)
