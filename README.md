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

The code for this step is contained in the 2nd code cell of the IPython notebook located in "./vehicle_detection_for_submission.ipynb".
