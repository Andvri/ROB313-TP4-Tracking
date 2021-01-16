import cv2
import numpy as np

RATE = .85

def get_shapes(img_gray):
    x ,y = np.gradient(img_gray)
    orientation = np.arctan2(y, x)
    norm = np.hypot(y, x)

    threshold = np.quantile(norm, q=RATE)
    mask = np.where(norm > threshold, 1, 0)

    mask_ = np.dstack((
        np.where(mask, orientation, 0),
        np.where(mask, orientation, 0),
        np.where(mask, orientation, 255)
    ))  # stacks 3 h x w arrays -> h x w x 3

    return orientation, norm, mask_


if __name__ == "__main__":
    img_path = './media/Input_TdH_Example.png'
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,2]
    

    orientation, norm, mask = get_shapes(gray)

    while True:
        print(norm / norm.max().max())
        cv2.imshow('Original', img)
        # cv2.imshow('Gray', gray)
        cv2.imshow('Orientation', orientation)
        cv2.imshow('Norm', norm / norm.max().max())
        cv2.imshow('Mask', mask.astype(float))
     
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

    cv2.destroyAllWindows()
