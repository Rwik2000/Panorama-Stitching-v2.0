import numpy as np
import cv2
import os
import imutils
import random

'''
Calculating the homography matrix using points mapping 
eachother in different planes. 

_single_homography() calculates homography matrix between any 
2 given points.

getHomography() finds the best homography transformation from 
the given points using RANSAC.
'''
class homographyRansac():
    def __init__(self, ransacThresh, ransacIter):
        self.ransacThresh = ransacThresh
        self.ransaciter = ransacIter
    def _single_homography(self, matches):
        num_rows = 2*len(matches)
        num_cols = 9
        A = np.zeros((num_rows, num_cols))
        for i in range(len(matches)):
            x_1 = (matches[i][0][0])
            y_1 = (matches[i][0][1])
            z_1 = 1
            x_2 = (matches[i][1][0])
            y_2 = (matches[i][1][1])
            z_2 = 1
            A[2*i,:] = [x_1,y_1,1,0,0,0,-x_2*x_1,-x_2*y_1,-x_2*z_1]
            A[2*i+1,:]=[0,0,0,x_1,y_1,1,-y_2*x_1,-y_2*y_1,-y_2*z_1]
        
        # Using SVD to solve
        U, D, V_t = np.linalg.svd(A)
        h = V_t[-1]
        H = np.zeros((3,3))
        H[0] = h[:3]
        H[1] = h[3:6]
        H[2] = h[6:9]
        H = H/H[2,2]

        return H

    def getHomography(self,ptsA, ptsB):
        final_H = 0
        Bestcount= 0
        for _ in range(self.ransaciter):
            _inliers = []
            # Random sampling
            rand_indices = random.sample(range(1, len(ptsA)),4)
            matches = [[ptsA[i], ptsB[i]] for i in rand_indices]

            H = self._single_homography(matches)
            count = 0

            for iter in range(len(ptsA)):
                curr_loc = ptsA[iter]
                trans_curr_loc = np.array([curr_loc[0], curr_loc[1], 1]).T
                transf_loc = H.dot(trans_curr_loc)
                transf_loc = transf_loc/transf_loc[2]
                actualNewLoc = np.array([ptsB[iter][0], ptsB[iter][1], 1])

                # L2 norm between the given points
                if np.linalg.norm(transf_loc - actualNewLoc) <= self.ransacThresh:
                    count+=1
                    _inliers.append([ptsA[iter],ptsB[iter]])
            # Assesing the best matrix
            if count>Bestcount:
                Bestcount = count
                final_H = self._single_homography(_inliers)
        return final_H