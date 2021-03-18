import cv2
import numpy as np
from skimage.transform import pyramid_gaussian, pyramid_laplacian
def blend(images, masks, n=5):
    """
    Image blending using Image Pyramids. We calculate Gaussian Pyramids using OpenCV.add()
    Once we have the Gaussian Pyramids, we take their differences to find Laplacian Pyramids
    or DOG(Difference of Gaussians). Then we add all the Laplacian Pyramids according to the
    seam/edge of the overlapping image. Finally we upscale all the Laplasian Pyramids to
    reconstruct the final image.
    images: array of all the images to be blended
    masks: array of corresponding alpha mask of the images
    n: max level of pyramids to be calculated.
    [NOTE: that image size should be a multiple of 2**n.]
    """

    assert(images[0].shape[0] % pow(2, n) ==
           0 and images[0].shape[1] % pow(2, n) == 0)

    # for i in range(len(images)):
    #     images[i] = images[i].astype(float)
    g_pyramids = {}
    l_pyramids = {}

    H, W, C = images[0].shape

    # Calculating pyramids for various images before hand
    for i in range(len(images)):

        # Gaussian Pyramids
        G = images[i].copy()
        g_pyramids[i] = [G]
        for _ in range(n):
            # G = cv2.resize(G,(G.shape[1]//2, G.shape[0]//2), interpolation=cv2.INTER_NEAREST)
            G = cv2.pyrDown(G)
            g_pyramids[i].append(np.float32(G))

        # Laplacian Pyramids
        l_pyramids[i] = [G]
        for j in range(len(g_pyramids[i])-2, -1, -1):
            # G_up = cv2.resize(G,(G.shape[1]*2, G.shape[0]*2), interpolation=cv2.INTER_NEAREST)

            G_up = cv2.pyrUp(G)
            G = g_pyramids[i][j]
            L = G - G_up
            l_pyramids[i].append(L)
           
    for i in range(len(masks)):
        masks[i] = masks[i].astype('bool')
    # Blending Pyramids
    common_mask = masks[0].copy()
    common_image = images[0].copy()
    common_pyramids = [l_pyramids[0][i].copy()
                        for i in range(len(l_pyramids[0]))]

    ls_ = None
    # We take one image, blend it with our final image, and then repeat for
    # n images
    for i in range(1, len(images)):

        y1, x1 = np.where(common_mask == 1)
        y2, x2 = np.where(masks[i] == 1)

        # cv2.imshow("cm",common_image)
        # cv2.imshow("curr",images[i])
        # cv2.waitKey(0)

        # print(x1,x2)
        if np.max(x1) > np.max(x2):
            left_py = l_pyramids[i]
            right_py = common_pyramids

        else:
            left_py = common_pyramids
            right_py = l_pyramids[i]

        # for j in range(len(left_py)):
            # cv2.imshow("l", left_py[j])
            # cv2.imshow("r", right_py[j])
            # cv2.waitKey(0)
        # To check if the two pictures need to be blended are overlapping or not
        mask_intersection = np.bitwise_and(common_mask, masks[i])
        # cv2.imshow("op", op)s
        if True in mask_intersection:
            # If images blend, we need to find the center of the overlap
            y, x = np.where(mask_intersection == 1)
            x_min, x_max = np.min(x), np.max(x)

            # We get the split point
            split = ((x_max-x_min)/2 + x_min)/W
            # Finally we add the pyramids
            LS = []
            for la, lb in zip(left_py, right_py):
                rows, cols, dpt = la.shape
                # print(rows, cols)
                ls = np.hstack((la[:, 0:int(split*cols)], lb[:, int(split*cols):]))
                LS.append(ls)

        else:
            LS = []
            for la, lb in zip(left_py, right_py):
                _, cols, _ = la.shape
                ls = la + lb
                LS.append(ls)

        # Reconstructing the image
        ls_ = LS[0]
        for j in range(1, n+1):
            ls_ = cv2.pyrUp(ls_)
            ls_ = ls_ + LS[j]
            ls_[ls_>255] = 255; ls_[ls_<0] = 0
            

        # Preparing the commong image for next image to be added
        common_image = ls_
        common_mask = np.sum(common_image.astype(bool), axis=2).astype(bool)
        common_pyramids = LS

    return np.uint8(ls_)