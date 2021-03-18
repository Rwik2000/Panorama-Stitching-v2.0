import cv2
import numpy as np


def LaplacianBlend(images, masks, n=5):
    """
    Blending the input the input images using Laplacian Blending. Essentially, we reduce the 
    the image to smaller sizes using OpenCV()'s pyrDown() and obtaining a gaussian pyramid.
    Then upsample the images using pyrUp() and finding the difference of Gaussian and hence 
    the laplacian pyramid. Image mask are used to find the partition and the Laplacian pyramid 
    images are joined to get the desired result's laplacian pyramid. Next, the pyramid is upsampled 
    again and add the Gaussians at each level to get the desired result.

    """

    # Ensuring the input images are a multiple of 2^n where n is the number of 
    # layers of the laplacian pyramid.
    assert(images[0].shape[0] % pow(2, n) ==
           0 and images[0].shape[1] % pow(2, n) == 0)

    # Empty list of Gaussian pyramids and laplacian pyramids for each of the input images
    g_pyramids = [None]*len(images)
    l_pyramids = [None]*len(images)

    _, W, _ = images[0].shape

    # Calculating pyramids of the images
    for i in range(len(images)):

        # Gaussian Pyramids
        G = images[i].copy()
        g_pyramids[i] = [G] #Storing the pyramids in a list to the corresponding image index
        for _ in range(n):
            G = cv2.pyrDown(G)
            g_pyramids[i].append(np.float32(G))

        # Laplacian Pyramids
        l_pyramids[i] = [G]  
        for j in range(len(g_pyramids[i])-2, -1, -1):
            G_up = cv2.pyrUp(G)
            G = g_pyramids[i][j]
            L = G - G_up # Difference of Gaussian (DOG)
            l_pyramids[i].append(L) #Storing the pyramids in a list to the corresponding image index 
    
    # Making the masks boolean for further operations
    for i in range(len(masks)):
        masks[i] = masks[i].astype('bool')
    
    common_mask = masks[0].copy() #All masks will be iterated and will be combined to the common mask
    common_image = images[0].copy() #All images will be iterated and will be combined to the common image
    common_pyramids = [l_pyramids[0][i].copy()
                        for i in range(len(l_pyramids[0]))] #All pyramids will be iterated and will be combined to the common pyr

    final_image = None
    # Iterating on the images.
    for i in range(1, len(images)):

        _, x1 = np.where(common_mask == 1)
        _, x2 = np.where(masks[i] == 1)

        #  Sorting the common image and the image to be added to left and right
        if np.max(x1) > np.max(x2):
            left_py = l_pyramids[i]
            right_py = common_pyramids

        else:
            left_py = common_pyramids
            right_py = l_pyramids[i]

        # Finding the region of intersection between the common image and the current image
        mask_intersection = np.bitwise_and(common_mask, masks[i])
        if True in mask_intersection:
            _, x = np.where(mask_intersection == 1)
            # finding the coordinate for the verticle line which would helpin overlapping the left and the right image.
            x_min, x_max = np.min(x), np.max(x)
            midPt = ((x_max-x_min)/2 + x_min)/W
            # Finally we add the pyramids
            LS = []
            for lap_L, lap_R in zip(left_py, right_py):
                _, cols, _ = lap_L.shape
                ls = np.hstack((lap_L[:, 0:int(midPt*cols)], lap_R[:, int(midPt*cols):]))
                LS.append(ls)
        # If there is no intersection, simply add the images
        else:
            LS = []
            for lap_L, lap_R in zip(left_py, right_py):
                _, cols, _ = lap_L.shape
                ls = lap_L + lap_R
                LS.append(ls)

        # Reconstruct the image
        final_image = LS[0]
        for j in range(1, n+1):
            final_image = cv2.pyrUp(final_image)
            final_image = final_image + LS[j]
            final_image[final_image>255] = 255; final_image[final_image<0] = 0 
            cv2.waitKey(0)
            
        common_image = final_image

        common_mask = np.sum(common_image.astype(bool), axis=2).astype(bool)
        common_pyramids = LS

    return np.uint8(final_image)