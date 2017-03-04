# Term-1 Assignment-5: CarND-Vehicle-Detection

## Vehicle Detection Project

### Goal
The goal is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car.

### Steps to complete this project are the following:
1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images 
2. Perform color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
3. Normalize the features 
4. Train a classifier to classify vehicles vs. non-vehicles
5. Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
6. Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
7. Estimate a bounding box for vehicles detected.

### Step 1: Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images 

The code for this step is contained in the 2, 5, 6 & 7th code cells of the IPython notebook located in "./vehicle_detection_for_submission.ipynb".

I used the labeled data set <a href=https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip>vehicles</a> and <a href=https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip>non-vehicles</a> to extract the HOG features. 

Here is an example of a car image and an exmaple of a non-car image:

![ScreenShot](images/image1.png)

I tried different color spaces, such as, HLS, HSV, YCrCb and found that YCrCb generated the best result when i tried using the trained model to detect vehicles in the test images. Also, I tested 0,1,2 and 'All' HOG channels and found that 0 performed the worst. 1 & 2 performed well with the validation set with accuracy > 0.98 but bad when detecting vehicles in the test images.  "All" performed good with the validation set with accuracy = 0.9575 but very well when detecting vehicles in the test images. 

Here are the HOG parameters to generate the HOG features:

```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
block orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
```

The extracted hog features are:

![ScreenShot](images/image2.png)

