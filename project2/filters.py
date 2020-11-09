import os

import numpy as np

from common import read_img, save_img


def image_patches(image, patch_size=(16, 16)):
    # Given an input image and patch_size,
    # return the corresponding image patches made
    # by dividing up the image into patch_size sections.
    # Input- image: H x W
    #        patch_size: a scalar tuple M, N
    # Output- results: a list of images of size M x N
    output = []
    # raise errors
    if image.shape[0] < patch_size[0]:
        raise ValueError('The width of required patch is larger than the orignal image. Please reconsider your input!')
    if image.shape[1] < patch_size[1]:
        raise ValueError('The width of required patch is larger than the orignal image. Please reconsider your input!')
    # slide orignal image x / np.linalg.norm(x)
    for i in range(0, image.shape[0]-patch_size[0]+1):
        for j in range(0, image.shape[1]-patch_size[1]+1):
            patch = (image[i:i+patch_size[0], j:j+patch_size[1]]) / np.linalg.norm(image[i:i+patch_size[0], j:j+patch_size[1]])
            output.extend([patch])
    return output


def convolve(image, kernel):
    # Return the convolution result: image * kernel.
    # Reminder to implement convolution and not cross-correlation!
    # You can use zero- or wrap-padding.
    #
    # Input- image: H x W
    #        kernel: h x w
    #
    # Output- convolve: H x W
    output = np.zeros(image.shape)
    kernel = np.flipud(np.fliplr(kernel))
    width = int(kernel.shape[0]/2)
    height = int(kernel.shape[1]/2)
    if (kernel.shape[0] % 2) == 0 or (kernel.shape[1] % 2) == 0:
        raise ValueError('The kernel is invaild. Please check your input!')
    image_con = np.zeros((image.shape[0]+2*width, image.shape[1]+2*height))
    image_con[width:width+image.shape[0], height:height+image.shape[1]] = image
    print('Compute the center of the image...')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img = image_con[i:i+2*width+1, j:j+2*width+1]
            output[i,j] = np.sum(img*kernel)

    return output

def gaussian_kernel(size=3, sigma=np.sqrt(1/(2*np.log(2)))):
    """Returns a 2D Gaussian kernel.
    Parameters
    ----------
    size : float, the kernel size (will be square)

    sigma : float, the sigma Gaussian parameter

    Returns
    -------
    out : array, shape = (size, size)
      an array with the centered gaussian kernel
    """
    x = np.linspace(- (size // 2), size // 2, size) 
    x /= np.sqrt(2)*sigma
    x2 = x**2
    kernel = np.exp(- x2[:, None] - x2[None, :])
    return kernel / kernel.sum()


def edge_detection(image):
    # Return the gradient magnitude of the input image
    # Input- image: H x W
    # Output- grad_magnitude: H x W

    kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) # 1 x 3
    ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) # 3 x 1

    Ix = convolve(image, kx)*0.5
    Iy = convolve(image, ky)*0.5

    grad_magnitude = np.sqrt(Ix**2 + Iy**2)

    return grad_magnitude, Ix, Iy

def sobel_operator(image):
    # Return Gx, Gy, and the gradient magnitude.
    # Input- image: H x W
    # Output- Gx, Gy, grad_magnitude: H x W

    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]) # 1 x 3
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) # 3 x 1

    Gx = convolve(image, kx)*0.5
    Gy = convolve(image, ky)*0.5

    grad_magnitude = np.sqrt(Gx**2 + Gy**2)

    return Gx, Gy, grad_magnitude


def steerable_filter(image, angles=(np.pi * np.arange(6, dtype=np.float) / 6)):
    # Given a list of angels used as alpha in the formula,
    # return the corresponding images based on the formula given in pdf.
    # Input- image: H x W
    #        angels: a list of scalars
    # Output- results: a list of images of H x W

    # TODO: Use convolve() to complete the function
    output = []

    return output


def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- Image Patches --
    # Q1
    patches = image_patches(img)
    # TODO choose a few patches and save them
    chosen_patches = None
    save_img(chosen_patches, "./image_patches/q1_patch.png")

    # Q2: No code
    """ Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- Gaussian Filter --
    # Q1: No code

    # Q2

    # TODO: Calculate the kernel described in the question.
    # There is tolerance for the kernel.
    kernel_gaussian = gaussian_kernel()
    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # Q3
    edge_detect, _, _ = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    edge_with_gaussian, _, _ = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")

    # -- Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # Q1: No code

    # Q2
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")
    # Q3
    steerable_list = steerable_filter(img)
    for i, steerable in enumerate(steerable_list):
        save_img(steerable, "./sobel_operator/q3_steerable_{}.png".format(i))

    print("Sobel Operator is done. ")
    """ LoG Filter """
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # Q1
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")

    # Q2: No code
    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
