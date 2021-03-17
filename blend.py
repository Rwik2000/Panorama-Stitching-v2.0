import cv2
import numpy as np
'''
The utility code to get the laplacian blending 
between two images. Levels denotes the number of layers to go in the gaussian pyramid
'''
def Laplacian_blending(img1,img2,mask,levels=4):
    
    G1 = img1.copy()
    G2 = img2.copy()
    GM = mask.copy()
    gp1 = [G1]
    gp2 = [G2]
    gpM = [GM]
    # Going down the pyramid
    for i in range(levels):
        GM = cv2.pyrDown(GM)
        G1 = cv2.pyrDown(G1)
        G2 = cv2.pyrDown(G2)
        gp1.append(np.float32(G1))
        gp2.append(np.float32(G2))
        gpM.append(np.float32(GM))

    # Arranging the pyramid formed
    gpMt = [gpM[levels-1]]
    lp1  = [gp1[levels-1]] 
    lp2  = [gp2[levels-1]]

    # Finding the difference of gaussians of scaled up version 
    # current version
    for i in range(levels-1,0,-1):

        gp1_up = cv2.pyrUp(gp1[i])
        gp2_up = cv2.pyrUp(gp2[i])
        k1 = np.subtract(gp1[i-1], gp1_up[:gp1[i-1].shape[0],:gp1[i-1].shape[1]])
        k2 = np.subtract(gp2[i-1], gp2_up[:gp2[i-1].shape[0],:gp2[i-1].shape[1]])
        lp1.append(k1)
        lp2.append(k2)
        gpMt.append(gpM[i-1]) 

    LS = []
    ct = 0
    for l1,l2,gm in zip(lp1,lp2,gpMt):
        ls = l1*gm+ l2*(1-gm)
        LS.append(ls)
        ct+=1

    # Reconstruction
    ls_ = LS[0]
    for i in range(1,levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = np.add(ls_[:LS[i].shape[0],:LS[i].shape[1]], LS[i])

    return np.uint8(ls_)

