"""
main.py for HW3.

feel free to include libraries needed
"""
import numpy as np
from matplotlib import pyplot as plt
from common import read_img, save_img

def homography_transform(X, H):
    # Perform homography transformation on a set of points X
    # using homography matrix H
    #
    # Input - a set of 2D points in an array with size (N,2)
    #         a 3*3 homography matrix
    # Output - a set of 2D points in an array with size (N,2)

    X_ = np.hstack((X,np.ones((X.shape[0],1))))
    Y = np.matmul(H,X_.T).T
    K = Y[:,2]
    K = K.reshape((K.shape[0],1))
    Y = Y/K
    return Y


def fit_homography(XY):
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    #
    # Input - an array with size(N,4), each row contains two
    #         points in the form[x^T_i,y^T_i]1×4
    # Output - a 3*3 homography matrix
    
    H = None
    return H


def p1():
    # code for Q1.2.3 - Q1.2.5
    # 1. load points X from p1/transform.npy

    # 2. fit a transformation y=Sx+t

    # 3. transform the points
    
    # 4. plot the original points and transformed points

    # code for Q1.2.6 - Q1.2.8
    case = 8  # you will encounter 8 different transformations
    for i in range(case):
        XY = np.load('p1/points_case_'+str(i)+'.npy')
        # 1. generate your Homography matrix H using X and Y
        #
        #    specifically: fill function fit_homography()
        #    such that H = fit_homography(XY)
        H = fit_homography(XY)
        # 2. Report H in your report
        print(H)
        # 3. Transform the points using H
        #
        #    specifically: fill function homography_transform
        #    such that Y_H = homography_transform(X, H)
        Y_H = homography_transform(XY[:, :2], H)
        # 4. Visualize points as three images in one figure
        # the following code plot figure for you
        plt.scatter(XY[:, 1], XY[:, 0], c="red")  # X
        plt.scatter(XY[:, 3], XY[:, 2], c="green")  # Y
        plt.scatter(Y_H[:, 1], Y_H[:, 0], c="blue")  # Y_hat
        plt.savefig('./case_'+str(i))
        plt.close()


def stitchimage(imgleft, imgright):
    # 1. extract descriptors from images
    #    you may use SIFT/SURF of opencv

    # 2. select paired descriptors

    # 3. run RANSAC to find a transformation
    #    matrix which has most innerliers

    # 4. warp one image by your transformation
    #    matrix
    #
    #    Hint:
    #    a. you can use function of opencv to warp image
    #    b. Be careful about final image size

    # 5. combine two images, use average of them
    #    in the overlap area

    pass


def p2(p1, p2, savename):
    # read left and right images
    imgleft = read_img(p1)
    imgright = read_img(p2)
    # stitch image
    output = stitchimage(imgleft, imgright)
    # save stitched image
    save_img(output, './{}.jpg'.format(savename))


if __name__ == "__main__":
    # Problem 1
    p1()

    # Problem 2
    p2('p2/uttower_left.jpg', 'p2/uttower_right.jpg', 'uttower')
    p2('p2/bbb_left.jpg', 'p2/bbb_right.jpg', 'bbb')

    # Problem 3
    # add your code for implementing Problem 3
    #
    # Hint:
    # you can reuse functions in Problem 2
