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

[image5]: ./WriteUpImages/straight2_undistorted.jpg "Straight Lanes - Undistorted"
[image6]: ./WriteUpImages/straight2_undistorted_polygon.jpg "Straight Lanes - with road polygon"
[image7]: ./WriteUpImages/straight2_undistorted_transformed_polygon.jpg "Straight Lanes - Bird's eye view"

[image8]: ./WriteUpImages/test2_slidingwindows.jpg "Sliding Windows Example"
[image9]: ./examples/warped_straight_lines.jpg "Warp Example"

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

The thresholded binary images was created by a combination of colour transforms, selecting only the Saturation channel after an RGB->HLS conversion, combined with a gradient thresholding using the sobel method.

In order to tune the thresholds, a seperate notebook (ImageProcessingTesting.ipnyb) was created to scan through the thresholding parameters using the test images. An example of the output of this process is shown below.

![alt text][image4]


While this method proved reasonably effective, the final video shows some areas where the chosen method struggles a little with shadows of trees that are cast across the road. This has the effect of causing a few jitters in the lane tracking algorithm but nothing too severe. It would be unlikely to use the output of this particular pipeline would be used directly for car control though without some suitable filtering on the curvature (see later section).

As an extention to this work, the next steps would be to consider CLAHE (Contrast Limited Adaptive Histogram Equalization) https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html which may improve the ability to filter out the shadows.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Stage 3 of the pipeline took the road in front of the car and performed a perspective transform to give a bird's eye view. The road surface itself was identified manually from a sample image (taken post the undistort step) where the car appears to be on a straight section of road. 

The 4 points obtained to give a suitable polygon are shown as source in the table below. The destination points are then the four corners of the output image. As the points refer to the actual lane line on the left and right, a 200 pixel margin outside the lane has been included.

| Source        | Destination      | 
|:-------------:|:----------------:| 
| 598, 445      | 200, 0           | 
| 683, 445      | xsize-200, 0     |
| 272, 675      | 200, ysize       |
| 1052, 675     | xsize-200, ysize |

Having manually defined these points, a sensible next step would be to normalize their coordinates to the frame size rather than assume the image will always be 1280 x 720. This has already been done for the destination frame. 

Below is an example showing one of the transforms applied. In theory it should look perfectly parallel in the processed image although there is a slight deviation. This has come about because the manual technique was applied to the two images and a slight difference was witnessed - hence this transform is an average of the two as there is no way to discern which of the examples is more representative. More examples could be further analysed to improve the accuracy of the transform if it causes issues further down the line.

![alt text][image5]
![alt text][image6]
![alt text][image7]


The code for the transform can be found in the pipeline step cells of the IPython notebook. It calculates the transform, as well as the inverse transform that will be used at the end of the pipeline, outside for efficiency. These are then wrapped up in the `pipeline_perspectivetransform` and `pipeline_inverttransform` functions.


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Stage 4 of the pipeline (defined in the 6th cell in the notebook) takes the bird's eye view and uses a sliding window algorithm to work up the image and identify the lane line before fitting a 2nd order polynomal to the results.

The algorithm works by using the previous frames (or a suitable initial guess) polynomial fits as a starting point before applying a window around this line. For each window, the centre of the line is calculated and these points are used to perform a new polyinomal fit. 

This line then describes the lane in the Bird's eye frame.

The code itself consists of two functions `fit_poly` uses the numpy library to fit the polynomial to a series of points that represent the centre of the line. `search_around_poly` takes the binary image, extracts any points that are activated and then fits the sliding window around the original polynomial and throws away any other points. Applying the `fit_poly` to these remaining points cunningly pulls the line towards dense clusters of activated pixels elegantly skipping the step of trying to iterate through each row of the image and calculating the position of the lane at each point. 

The polynomial coefficients for both the left and right lane are stored as global variables that are updated each step, accordingly everytime the polynomial is updated the initial guess is also updated.

The final step of the pipeline step is to colour in the window and plot the resulting fit line. An example is shown below.

![alt text][image8]

Note this algorithm can be tuned by defining the size of the window around the existing line. In this case a value of 125 pixels was chosen. By reducing the window size the algorithm is less sensitive to spurious pixels and the polynomial should not change as much from frame-to-frame. However, this leaves the possibility of not tracking higher curvatures. the value chosen in this case seems appropriate for the highway scenario although could be further wrapped up with either a low-pass filter or a slew-rate limiting filter that only permits a certain degree of variation between consecutive frames. This would seem sensible as it is highly unlikely that the lane lines will change dramatically within the time step of the frame. These filters are suggested and recommended but have not been investigated in this submission.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Having calculated approprate polynomial fits to the lane lines, calculating the curvature is a matter of differentiation. That is achieved with the following formula:

```
curverad = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
```
where y_eval is the point at which we are evaluating the curvature. In this case it is the bottom of the image, or the current location of the car.

It should be noted that this calculates the curvature of the line in pixels and therefore needs converting to actual distances. As no measures are provided, these need to be estimated for this purpose. Given that the lane on the highway in the USA is approximately 3.7m and in the image it is 880 pixels, the conversion can be defined as xm_per_pixel = 3.7 / 880. 

For the y axis, it is a little more difficult but an approximation would suggest it is about 30 metres being used as the region of interest. This will change with inclination and camera pitch but will suffice for these purposes. According ym_per_pixel = 30 / 720.

In addition to the curvature calculation, this code cell includes the function to calculate offset. Here the lane fits are used to determine where the centre of the lane is relative to the centre of the car (assuming the camera is in the centre of the car) and calculated the offset between the two in metres.



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Having determined the sliding windows for lane identification, polynomial fits, radius of curvature and lane offset a set of plots are generated at each stage to demonstrate these outputs. In the final stage of the pipeline, these images are warped back from the bird's eye view to the aspect of the original image (post undistort step). A caption is then added with the radius of curvature and lane offset calculation. An example is shown below.


![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_processed.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

During the development of the pipeline one of the most difficult aspects seemed to be getting a reasonably robust lane detection stage. This was primarily due to the process of adjusting the various parameters of the colour threshold and gradient thresholds to give the clearest line delineation from the background noise. The shadows cast across the road in the final video show this as being a particularly difficult scenario to deal with.

To try and help with this process, a seperate notebook was created that iterated through various parameters for thresholds and allowed a side-by-side comparison of the images. Using this sensible values were honed in on quite reasonably.

Once this stage was achieved with a suitable level of accuracy, the remaining stages of the pipeline fell into place. Some debugging was necessary to ensure that the radius of curvature, particularly the conversion from pixels to metres and the lane offset, was correct and some iteration was required on the polyfit of the lane lines. Here, as it is a recursive process, the initial guess had to be tweaked for each sample image to ensure the first fit was appropriate. When moving to the video this was less of an issue as the previous frame was available. 

Looking at the final result the most likely area for the pipeline to fall over would be due to the issue already noted with the shadows cast over the road surface confusing the lane fitting algorithm. Further work could be performed to try further refine the thresholding to better cope with these shadows. One method is suggested previously in the text. In addition various other techniques could be employed to impose more realistic limits on the lane fitting as a function of time - for example a low pass filter as we know that the lanes do not change dramatically, particularly on a highway.

Finally, there are certain refactoring elements that could be applied to the resulting code to tidy it up. The use of global variables could be tidied up and there are some inefficiencies in the calculation that have come about as each stage is developed in isolation. For production this would be cleaned up but has been left as is to demonstrate each step independently as well as being formally parameteristed to make a pipeline robust against different image sizes, camera positions and installations. 


