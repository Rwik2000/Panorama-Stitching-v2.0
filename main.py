# Code by Rwik Rana rwik.rana@iitgn.ac.in
'''
This is code to stitch image staken from various perspectives to output a
panorama image. In this code, I have used concepts of inverse homography, 
warping, RANSAC and Laplacian Blending to solve the problem statement.

To use the code, 
1. Add your dataset in the Dataset Directory.
2. In line 147, add your datasets in the Datasets array.
3. The output of the same would be save in the Output directory.

Flow of the Code:
1. SIFT to detect features.
2. Matching features using KNN
3. Finding the the homography matrix i.e. the mapping from one image plane to another.
4. warping the images using the inverse of homography matrix to 
   ensure no distortion/blank spaces remain in the transformed image.
5. Stitching all the warped images together
'''
import numpy as np
import cv2
import os
import shutil
import imutils
import random

from homography import homographyRansac
from Warp import Warp
from blend import Laplacian_blending

class panaroma_stitching():    
    def __init__(self):
        # Class parameters
        self.isLeft = 1
        self.isFinal = 0
        self.LoweRatio = 0.75
        # Blending toggle
        self.blendON = 1
    def stitchTwoImg(self, images):

        (imageB, imageA) = images
        # Finding features and their corresponding keypoints in the given images
        (kpsA, featuresA) = self.findSIFTfeatures(imageA)
        (kpsB, featuresB) = self.findSIFTfeatures(imageB)
        # match features between the two images
        H, invH= self.mapKeyPts(kpsA, kpsB,featuresA, featuresB)
        warpClass = Warp()
        # Setting the offsets
        if self.isLeft:
            warpClass.yOffset = imageA.shape[1]
        else:
            warpClass.yOffset = 200
        warpClass.xOffset = 50
        offsets = [warpClass.xOffset, warpClass.yOffset]

        # Warping the image
        warpedImg = warpClass.InvWarpPerspective(imageA, invH,H,
            (imageB.shape[0] + 100, imageB.shape[1] + imageA.shape[1]))
        result = np.uint8(warpedImg)
        if self.isFinal==0:
            if self.blendON:
                # Mask for laplacian blending
                m = np.ones_like(imageA)
                # Warping the mask
                m = warpClass.InvWarpPerspective(m, invH,H,
                (imageB.shape[0] + 100, imageB.shape[1] + imageA.shape[1]))

                warpedImg = np.uint8(warpedImg)
                tmp_result = np.zeros_like(warpedImg)
                tmp_result[0 + offsets[0]:imageB.shape[0] + offsets[0], 0 +  offsets[1]:imageB.shape[1]+  offsets[1]] = imageB
                # Blending
                result = Laplacian_blending(warpedImg, tmp_result, m, 3)
            else:
                # No blending
                result[0 + offsets[0]:imageB.shape[0] + offsets[0], 0 +  offsets[1]:imageB.shape[1]+  offsets[1]] = imageB
        else:
            result = np.uint8(warpedImg)
            for i in range(0 + offsets[0], imageB.shape[0] + offsets[0]):
                for j in range(0 +  offsets[1], imageB.shape[1]+  offsets[1]):
                    blank = result[i][j]==[0,0,0]
                    if blank.all():
                        result[i][j] = imageB[i - offsets[0]][j - offsets[1]]

        return result

    def findSIFTfeatures(self, image):

        desc = cv2.SIFT_create()
        (kps, features) = desc.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)
    
    def mapKeyPts(self, kpsA, kpsB, featuresA, featuresB):
        desc = cv2.DescriptorMatcher_create("BruteForce")
        _Matches = desc.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in _Matches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.LoweRatio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])      
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # Calling the homography class
            homographyFunc = homographyRansac(4,400)
            # Obtaining the homography matrix
            H= homographyFunc.getHomography(ptsA, ptsB)
            # Obtaining the inverse homography matrix
            invH = np.linalg.inv(H)
            return H, invH
        return None
    
    def MultiStitch(self, images):
        num_imgs = len(images)
        # Separating left and right side of the final image 
        if num_imgs%2 == 0:
            left = images[:num_imgs//2]
            right = images[num_imgs//2:]
        else:
            left = images[:num_imgs//2 + 1]
            right = images[num_imgs//2:]
        print("========> Stitching Left Side ...")
        # Processing left Side
        tempLeftStitch = images[0]
        while len(left) >1:
            temp_imgB = left.pop(0)
            temp_imgA = left.pop(0)
            tempLeftStitch = self.stitchTwoImg([temp_imgA, temp_imgB])
            left.insert(0, tempLeftStitch)
        tempRightStitch = images[-1]
        self.isLeft = 0
        # Processing Right Side
        print("========> Done! \n========> Stitching Right Side ...")
        while len(right)>1:
            temp_imgA = right.pop(0)
            temp_imgB = right.pop(0)
            tempRightStitch = self.stitchTwoImg([temp_imgA, temp_imgB])
            right.insert(0, tempRightStitch)
        print("========> Done! \n========> Stitching Both Sides ...")
        self.isLeft = 0
        self.isFinal = 1
        final = self.stitchTwoImg([tempLeftStitch, tempRightStitch])        
        return final, tempLeftStitch, tempRightStitch
        

# image in the dataset must be in order from left to right
Datasets = ["I5"]
for Dataset in Datasets:
    print("Stitching Dataset : ", Dataset)
    Path = "Dataset/"+Dataset
    images=[]
    for filename in os.listdir(Path):
        if filename.endswith(".JPG") or filename.endswith(".PNG"):
            img_dir = Path+'/'+str(filename)
            images.append(cv2.imread(img_dir))

    for i in range(len(images)):
        images[i] = imutils.resize(images[i], width=500)

    images = images[1:5]
    stitcher = panaroma_stitching()
    result, left, right = stitcher.MultiStitch(images)
    print("========>Done! Final Image Saved in Outputs Dir!")
    if os.path.exists("Outputs/"+Dataset):
        shutil.rmtree("Outputs/"+Dataset)
    os.makedirs("Outputs/"+Dataset, )
    cv2.imwrite("Outputs/"+Dataset+"/"+Dataset+".JPG", result)
    cv2.imwrite("Outputs/"+Dataset+"/"+Dataset+"_left.JPG", left)
    cv2.imwrite("Outputs/"+Dataset + "/"+Dataset+"_right.JPG", right)


