
import numpy as np
import cv2
import os
import imutils
import random


class Warp():
    def __init__(self):
        self.xOffset = 0
        self.yOffset = 0
    # AVOID USING THIS FUNCTION
    def warpPerspective(self, im, H, output_shape):
        """ Warps an image im using (3,3) homography Matrix matrix H. 
            Warping is done using forward homography, This gives large 
            distortions in the transformed images."""
        warpImage = np.zeros(( output_shape[0], output_shape[1],3))
        for i in range(0, im.shape[1]):
            for j in range(0, im.shape[0]): 
            
                M = H.dot(np.array([i,j,1]).T)
                M = M/M[2]
                p, q = M[0], M[1] 
                rp = int(round(p))
                rq = int(round(q))
                try:
                    if rq>=0 - self.xOffset and rp>=0 - self.yOffset:
                        warpImage[rq + self.xOffset, rp + self.yOffset] = im[j, i]
                        # Makeshift filling algorithm
                        for k in range(4):
                            warpImage[rq + self.xOffset, rp + k + self.yOffset] = im[j, i]
                            warpImage[rq + self.xOffset, rp - k + self.yOffset] = im[j, i]
                            warpImage[rq + k + self.xOffset, rp + self.yOffset] = im[j, i]
                            warpImage[rq - k + self.xOffset, rp + self.yOffset] = im[j, i]


                except:
                    continue

        return warpImage

    def InvWarpPerspective(self, im, invA, H,output_shape):
        """ Warps an image im using (3,3) homography Matrix matrix H. 
            Warping is done using inverse homography, to ensure 
            distortions aren't there in the output.
        """
        # obtaining the coordinates of the final transformed image and
        # only iterating on the range to reduce time. 

        x1 = [0,0,1]
        x2 = [im.shape[1], im.shape[0],1]
        x1_trnsf = H.dot(np.array(x1).T)
        x1_trnsf = list(x1_trnsf/x1_trnsf[2])
        x2_trnsf = H.dot(np.array(x2).T)
        x2_trnsf = list(x2_trnsf/x2_trnsf[2])

        y_min = int(min(x1_trnsf[1], x2_trnsf[1]))
        y_max = int(max(x1_trnsf[1], x2_trnsf[1]))
        x_min = int(min(x1_trnsf[0], x2_trnsf[0]))
        x_max = int(max(x1_trnsf[0], x2_trnsf[0]))

        warpImage = np.zeros(( output_shape[0], output_shape[1],3))
        for i in range(x_min, x_max): 
            for j in range(y_min, y_max): 
                M = invA.dot(np.array([i,j,1]).T)
                M = M/M[2]
                p, q = M[0], M[1]
                rp = int(round(p))
                rq = int(round(q))

                try:
                    if rq>=0 and rp>=0 and j+self.xOffset>=0:
                        warpImage[j +self.xOffset, i +self.yOffset] = im[rq, rp]
                except:                    
                    continue              

        return warpImage