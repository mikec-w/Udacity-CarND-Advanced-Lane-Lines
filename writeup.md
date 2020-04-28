## Advanced Lane Finding Project Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration2.jpg "Original"
[image2]: ./output_images/calibration2_undistorted.jpg "Undistorted"
[image3]: ./examples/undistort_output.png "Undistorted"
[image4]: ./test_images/test1.jpg "Road Transformed"
[image5]: ./straight2_undistorted.jpg "Straight Lanes - Undistorted"
[image6]: ./straight2_undistorted_polygon.jpg "Straight Lanes - with road polygon"
[image7]: ./straight2_undistorted_transformed_polygon.jpg "Straight Lanes - Bird's eye view"

[image8]: ./examples/binary_combo_example.jpg "Binary Example"
[image9]: ./examples/warped_straight_lines.jpg "Warp Example"
[image10]: ./examples/color_fit_lines.jpg "Fit Visual"
[image11]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Writeup / README

This write up describes the details of the implementation along with issues experienced and potential improvements that could be made for each of the required points.


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Camera calibration was performed in the seperate `CameraCalibration.ipynb` notebook and was based on the example [code](https://github.com/udacity/CarND-Camera-Calibration). It was kept seperate as the camera calibration is essentially a pre-processing step that only has to be performed once. Once complete the calibration information was saved to a pickle file that is loaded into the main pipeline processing notebook.

The calculation of the camera matrix follows the description already provide below

>I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

>I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

The calibration created attempted to use all of the camera calibration images provided. Initially it failed on most images as the 8x6 grid could not be found. Changing this to 9x6 and updating the objp array size resolved this issue with only 3 images now failing. It is likely that by combining different grid sizes for the algorithm to find, these images could be included in the calibration but a satisfactory result was obtained without that being necessary.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

This first stage of the image processing pipeline used the OpenCV function to apply the previously calculated camera calibration to the image. 

```python
img = cv2.undistort(img, mtx, dist, None, mtx)
```

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Stage 3 of the pipeline took the road in front of the car and performed a perspective transform to give a bird's eye view. The road surface itself was identified manually from a sample image where the car appears to be on a straight section of road. 

The 4 points obtained to give a suitable polygon are shown as source in the table below. The destination points are then the four corners of the output image.

| Source        | Destination      | 
|:-------------:|:----------------:| 
| 607, 440      | 200, 0           | 
| 673, 440      | xsize-200, 0     |
| 265, 675      | 200, ysize       |
| 1052, 675     | xsize-200, ysize |

Having manually defined these points, a sensible next step would be to normalize their coordinates to the frame size. This has already been done for the destination frame. 

Note, the manual comparison needs to be performed with the undistorted images rather than the sample images as this will affect the result.

Below is an example showing one of the transforms applied. In theory it should look perfectly parallel in the processed image although there is a slight deviation. This has come about because the manual technique was applied to the two images and a slight difference was witnessed - hence this transform is an average of the two as there is no way to discern which of the examples is more representative. More examples could be further analysed to improve the accuracy of the transform if it causes issues further down the line.

![alt text][image5]
![alt text][image6]
![alt text][image7]


The code for the transform can be found in the pipeline step cells of the IPython notebook. It calculated the transform, as well as the inverse transform that will be used at the end of the pipeline, outside for efficiency. These are then wrapped up in the `pipeline_perspectivetransform` and `pipeline_inverttransform` functions.



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
