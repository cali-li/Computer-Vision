import os

import numpy as np

from common import read_img, save_img
from filters import *


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    # Given an input image, x_offset, y_offset, and window_size,
    # return the function E(u,v) for window size W
    # corner detector score for that pixel.
    # Input- image: H x W
    #        u: a scalar for x offset
    #        v: a scalar for y offset
    #        window_size: a tuple for window size
    #
    # Output- results: a image of size H x W
    # Use zero-padding to handle window values outside of the image.
    #
    # set up
    width = int(window_size[0]/2)
    height = int(window_size[1]/2)
    diff = 0
    output = np.zeros(image.shape)
    # create image_offset containing all info from the image
    image_offset = np.zeros((image.shape[0]+window_size[0]-1, image.shape[1]+window_size[1]-1))
    image_offset[width:width+image.shape[0], height:height+image.shape[1]] = image

    # loop over
    for i in range(image.shape[0]):
        if i%50 == 0:
            print('The',i, 'th iteration...',i,':',image.shape[0])
        for j in range(image.shape[1]):
            for x in range(-width, width+1):
                for y in range(-height, height+1):
                    if i+width+x+u >= image_offset.shape[0] or j+height+y+v >= image_offset.shape[1] or i+width+x+u < 0 or j+height+y+v < 0:
                        pass
                    else:
                        diff += (image_offset[i+width+x+u,j+height+y+v] - image_offset[i+width+x,j+height+y])**2
            output[i,j] = diff
            diff = 0

    return output


def harris_detector(image, window_size=(5, 5),alpha = 0.06):
    # Given an input image, calculate the Harris Detector score for all pixels
    # Input- image: H x W
    # Output- results: a image of size H x W
    #
    # You can use same-padding for intensity (or 0-padding for derivatives)
    # to handle window values outside of the image.

    # compute the derivatives
    
    kx = np.array([[-1, 0, 1]]) * 0.5 # 1 x 3
    ky = np.transpose(kx)  # 3 x 1    

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    Ixx = Ix*Ix
    Ixy = Ix*Iy
    Iyy = Iy*Iy

#     k_gauss = np.ones(window_size)

    # gaussian kernel works better than the average kernel
    k_gauss = gaussian_kernel(window_size[0],1)
    M = np.zeros((image.shape[0],image.shape[1],3))
    M[:,:,0] = convolve(Ixx,k_gauss)
    M[:,:,1] = convolve(Ixy,k_gauss)
    M[:,:,2] = convolve(Iyy,k_gauss)
    
    R =  M[:,:,0]*M[:,:,2] - M[:,:,1]**2 - alpha*((M[:,:,0]+M[:,:,2]))**2 
    response = R

    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # Define offsets and window size and calulcate corner score
    u, v, W = 0, 2, (5, 5)

    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score.png")

    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")


if __name__ == "__main__":
    main()
