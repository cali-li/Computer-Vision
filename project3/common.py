"""Common functions."""
import cv2


def read_img(path):
    """Read image."""
    image = cv2.imread(path, cv2.COLOR_BGR2RGB)
    return image


def save_img(img, path):
    """Save image."""
    cv2.imwrite(path, img)
    print(path, "is saved!")
