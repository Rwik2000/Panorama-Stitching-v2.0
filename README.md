![alt text](https://github.com/Rwik2000/Panorama-Stitching/blob/main/Dataset/sample.jpg)
# Panorama Stitching

Code by Rwik Rana, IITGn.

View the project on https://github.com/Rwik2000/Panorama-Stitching for better viewing 

This is a Code to stitch multiple images to create panoramas. This is robust for stitching **4 images**. This is the tree for the entire repo:\

```
Panorama Stitching
        |
        |___Dataset
        |       |--I1
        |       |--I2 ....
        |
        |___Outputs
        |
        |___blend.py
        |___Homography.py
        |___Main.py
        |___Warp.py

```

In this code, I have used concepts of inverse homography, 
warping, RANSAC and Laplacian Blending to solve the problem statement.

### To use the code, 

##### Libraries needed
* opencv
* numpy
* imutils

##### Procedure:
In the [main.py](https://github.com/Rwik2000/Panorama-Stitching/blob/main/main.py), 
1. Add your dataset in the Dataset Directory.
2. In [line 147](https://github.com/Rwik2000/Panorama-Stitching/blob/main/main.py#L147), add your datasets in the Datasets array.
```python
Datasets = ["I5"] #ADD YOUR DATASET HERE
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
```
3. The output of the same would be save in the Output directory.

### Flow of the Code:
1. SIFT to detect features.
2. Matching features using KNN
3. Finding the the homography matrix i.e. the mapping from one image plane to another.
4. warping the images using the inverse of homography matrix to 
   ensure no distortion/blank spaces remain in the transformed image.
5. Stitching all the warped images together

#### Implementation of Ransac:
Refer to the `homographyRansac()` in [Homography.py](https://github.com/Rwik2000/Panorama-Stitching/blob/main/homography.py). Access the object `getHomography()` of the class to get the desired homography matrix.
```python
        # Set the threshold and number of iterations neede in RANSAC.
        self.ransacThresh = ransacThresh
        self.ransaciter = ransacIter


```
For Warping, use the `InvWarpPerspective()` from [Warp.py](https://github.com/Rwik2000/Panorama-Stitching/blob/main/Warp.py).

**Note** : Avoid using the `warpPersepctive()`, because it gives distorted and incomplete projection of the images.

```python
def InvWarpPerspective(self, im, invA, H,output_shape):

        x1 = [0,0,1]
        x2 = [im.shape[1], im.shape[0],1]
        x1_trnsf = H.dot(np.array(x1).T)
        x1_trnsf = list(x1_trnsf/x1_trnsf[2])
        x2_trnsf = H.dot(np.array(x2).T)
        x2_trnsf = list(x2_trnsf/x2_trnsf[2])

        .
        .
        .
        .       
                
        return warpImage
```

## Results

![alt text](https://github.com/Rwik2000/Panorama-Stitching/blob/main/Final_Outputs/I5.JPG)

![alt text](https://github.com/Rwik2000/Panorama-Stitching/blob/main/Outputs/I1/I1.JPG)

![alt text](https://github.com/Rwik2000/Panorama-Stitching/blob/main/Outputs/I2/I2.JPG)


