![alt text](https://github.com/Rwik2000/Panorama-Stitching-v2.0/blob/main/Dataset/sample.jpg)
# Panorama Stitching

Code by Rwik Rana, IITGn.

View the project on https://github.com/Rwik2000/Panorama-Stitching-v2.0 for better viewing 

This is a Code to stitch multiple images to create panoramas.This is the tree for the entire repo:

```
Panorama Stitching
        |
        |___Dataset
        |       |--I1
        |       |--I2 ....
        |
        |___Outputs
        |
        |___blend_and_stitch.py
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
In the [main.py](https://github.com/Rwik2000/Panorama-Stitching-v2.0/blob/main/main.py), 
1. Add your dataset in the Dataset Directory.
2. In [line 157](https://github.com/Rwik2000/Panorama-Stitching-v2.0/blob/main/main.py#L157), add your datasets in the Datasets array.
```python
Datasets = ["I1","I4","I5"] #ADD YOUR DATASET HERE
for Dataset in Datasets:
    print("Stitching Dataset : ", Dataset)
    Path = "Dataset/"+Dataset
    images=[]
    alpha = []
    for filename in os.listdir(Path):
        if filename.endswith(".JPG") or filename.endswith(".PNG"):
            img_dir = Path+'/'+str(filename)
            images.append(cv2.imread(img_dir))

    for i in range(len(images)):
        images[i] = imutils.resize(images[i], width=300)

    images = images[:]
    stitcher = panaroma_stitching()
    result = stitcher.MultiStitch(images)
    print("========>Done! Final Image Saved in Outputs Dir!\n\n")
    if os.path.exists("Outputs/"+Dataset):
        shutil.rmtree("Outputs/"+Dataset)
    os.makedirs("Outputs/"+Dataset, )
    cv2.imwrite("Outputs/"+Dataset+"/"+Dataset+".JPG", result)

```
3. The output of the same would be save in the Output directory.
4. In the terminal run `python main.py`

### Flow of the Code:
1. SIFT to detect features.
2. Matching features using KNN
3. Finding the the homography matrix i.e. the mapping from one image plane to another.
4. warping the images using the inverse of homography matrix to 
   ensure no distortion/blank spaces remain in the transformed image.
5. Stitching and blending all the warped images together

#### Implementation of Ransac:
Refer to the `homographyRansac()` in [Homography.py](https://github.com/Rwik2000/Panorama-Stitching-v2.0/blob/main/homography.py). Access the object `getHomography()` of the class to get the desired homography matrix.
For my implementation, for each pair of image, RANSAC does 100 iterations and threshold is kept as 4.
```python
        # Set the threshold and number of iterations neede in RANSAC.
        self.ransacThresh = ransacThresh
        self.ransaciter = ransacIter


```
#### Warping
For Warping, use the `InvWarpPerspective()` from [Warp.py](https://github.com/Rwik2000/Panorama-Stitching-v2.0/blob/main/Warp.py).

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
#### Laplacian Blending
Input list of images and their corresponding masks in grayscale. Along with that add the number of layers of laplcian pyramid is to be made for the final blending and stitching. it is to be noted that the input images to the `LaplacianBlending()` function must have a shape such that the number of rows and columns are divisible by 2^n.

The shape of the output image can be changed in [Line 60](https://github.com/Rwik2000/Panorama-Stitching-v2.0/blob/main/main.py#L60) of of `main.py` file. Notice here, I have kept a size of (640, 1600) i.e. height and width being 640 and 1600 respectively.

If you do not want blending, in the main.py `panorama_stitching()` Class, turn the `blendON` to False.
One would get a result like:
![alt text](https://github.com/Rwik2000/Panorama-Stitching-v2.0/blob/main/Outputs/No_blend.png)

## Results

![alt text](https://github.com/Rwik2000/Panorama-Stitching-v2.0/blob/main/Outputs/I1/I1.JPG)

![alt text](https://github.com/Rwik2000/Panorama-Stitching-v2.0/blob/main/Outputs/I4/I4.JPG)

![alt text](https://github.com/Rwik2000/Panorama-Stitching-v2.0/blob/main/Outputs/I5/I5.JPG)


## Files:
* `main.py` to run and compile the entire code.
* `Warp.py` contains `Warp()` clas which helps in image warping
* `homography.py` helps in finding the homography for two images. It also contains the code for RANSAC.
* `blend_and_stitch.py` contains `stitcher()` Class which helps in blending and stitching. You can either use `laplcacianBlending()` for blending and stitching or `onlyStitch()` to only stitch without blending.

