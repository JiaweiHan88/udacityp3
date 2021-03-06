# **Traffic Sign Recognition** 



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeupimage/roadwork_rgb.png "roadwork_rgb.png"
[image2]: ./writeupimage/roadwork_y.png "roadwork_y.png"
[image3]: ./writeupimage/class_dist.png "class_dist.png"
[image4]: ./writeupimage/augmented.png "augmented.png"
[image5]: ./writeupimage/augm_dist.png "augm_dist.png"
[image6]: ./testimages/attention.png "Traffic Sign 1"
[image7]: ./testimages/child.png "Traffic Sign 2"
[image8]: ./testimages/roadwork.png "Traffic Sign 3"
[image9]: ./testimages/stop.png "Traffic Sign 4"
[image10]: ./testimages/vorfahrt.png "Traffic Sign 5"
[image11]: ./writeupimage/results.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy methods to calculate the following information of the data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. An example image from the database is shown with the provided sign name found in the csv according to label:

![alt text][image1]

The distribution of the training, validation and test dataset are displayed:

As can be seen, the distribution is very uneven between classes, but the class distribution is similar in the three datasets.
 
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to YUV color space and only use the Y channel to reduce the dimensionality without losing too much information. This was also one of the approaches taken in the research paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks"

Here is an example of a traffic sign image before and after conversion.

![alt text][image1]
![alt text][image2]

As a next step, I normalized the image data to increase numerical stability, using very big or small values can add up to a large error. Instead we want our data to have zero mean.
For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data. I used another variation where i normalized the data around its mean value.

The mean value of the training dataset before normalization was 131.34 and afterwards 0.03

I decided to generate additional data because i see a big difference in training data samples for the different classes. I was afraid that this could lead to a bias of the neural network to the classes with high sample count.
Another reason was to increase robustness of the network and reduce overfitting.

To add more data to the the data set, I used the geometric transformation of images such as  translation, rotation and affine transformation as described in (https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html)

In addition i added a random offset to the Y channel, which represents a change in brightness.

These images could represent the same traffic sign from different camera angles/parameters under different lighting condition.

Here is an example of an original image and an augmented image:

![alt text][image4]

Initially augmentated images were only generated for classes with low sample count in the training data, i increased the minimum sample count to 1000. Which increased the training data set to 51690 samples with the following class distribution:

![alt text][image5]

I compared the results with original and augmentated data set and they were virtually the same, i was not sure why the additional data did not lead to higher accuracy.
Even after increasing the sample count to 5000, i could not see any significant improvements in accuracy.
Through testing with different transformation parameters i recognized that the coefficients i used to translate and rotate the images were too big. Reducing them led to a much better training progress and accuracy.
For the final model, i used a training set of 215000 with 5000 sample each class.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Y Channel image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 (*A) 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x800    									|
| RELU					|							(*B) 					|
| Flatten               | (*A) 5x5x32 = 800 concat with (*B)  1x1x800 = 800	outputs 1600		|
| Dropout Layer	      	| 			|
| Fully connected		| 1600 in 500 out	        									|
| Dropout Layer	      	| 			|
| Fully connected		| 500 in 200 out	        									|
| Dropout Layer	      	| 			|
| Fully connected		| 200 in 43 out	        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer to minimize the softmax cross entropy loss function. It was used in the class and said to be suitable for this task.
The model was trained in 30 epochs using a batch size of 32. (Although the accuracy was already almost at max after 15 epoch, just to make sure it wont increase further) I used an exponential decaying learning rate starting from 0.0005 to receive the best results for my network.
For the dropout i set a keep probability of 0.5 to reduce overfitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Initially I used the original LeNet architecture from the class project, because it was providing good results for image classification.
Using this architecture i played a bit around with RGB / Grayscal / Y channel as input information and different normalization methods.
Using the hyperparameters from the previous examples, the accuracy was only around 90%. I did not spend to much time on the LeNet parameters after reading the research paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks", because the proposed architecture seems to provide better results.
The proposed architecture has two main differences, firstly it has a third convolution layer compared to LeNet and secondly it uses both the output of the last stage and the output of the second conv layer to feed into a classifier.
For the classifier I compared a 1-layer vs 2-layer vs 3-layer implementation where the 3-layer provided better results.
With the new architecture and normalized Y channel inputs, i could get a validation accuracy of around 94% for the original dataset. To further improve the model the following steps were taken:

- increase augmentated data to 5000 each class (with reduced transformation parameters)
- testing different normalization methods
- adding a histogram equalizer to adjust contrast for better visibility (in preprocessing)
- lowering learning rate and using decaying learning rate
- increase number of epochs (could be decreased after using a much larger training dataset)
- adding dropout layers before each fully connected layer
- increase output dimension of the convolution layers and number of features fed to the classifier

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.987 
* test set accuracy of 0.967


 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The pictures I found from the web have better quality then most of the training data set. I did not expect difficulties for the classification.
I could imagine that a classification might be difficult/incorrect in case of heavily warped/distorted images, bad lighting/shadows, partly covered or dirty signs or very low resolution.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution      		| General caution   									| 
| Children crossing    			| Children crossing								|
| Road work	      		| Road work					 				|
| Stop		| Stop     							|
| Yield					| Yield											|



The model was able to correctly classify 5 of the 5 traffic signs, which gives an accuracy of 100%. Because of the low test samples, the high accuracy is not trustworthy, but indicates that the classifier works well on real world data (data outside of the dataset)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image11]

Softmax prob for the five images above (highest 5) and their respective labels

    [1.0000000e+00, 1.0290601e-20, 6.1712697e-22, 5.9756872e-33, 4.1600515e-34] --> [18, 26, 27, 11, 22]

    [1.0000000e+00, 2.7169181e-19, 2.1926519e-20, 7.1907966e-21, 3.1725194e-22] --> [28, 29, 20, 27, 30]

    [1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00] --> [25,  0,  1,  2,  3]
       
    [1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00] --> [17,  0,  1,  2,  3]
       
    [1.0000000e+00, 4.3071352e-24, 1.5087655e-25, 5.5242635e-27, 4.4911257e-27] --> [13, 12, 15,  3, 14]


The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

For all the images the softmax probability was at 100 or near 100%. For image 3, 4, the softmax prob was flat zero except for the correctly predicted class.
For the other images, we have a near zero probability for the 2nd and following class, which represents classes with similarities compared to the ground truth class as can be seen in the Ipython notebook.
As all the probabilities are ~100%, i did not include a bar chart and also left the 5th class out (since the prob for the 2nd is already ~0).

This means that our model is pretty confident in the prediction of my images, it would be nice to find more traffic signs and find examples where our model fail or have multiple classes as candidates.
We could try to find images for the speedlimits or beware of ice/snow, which were revealed in the confusion matrix to be less accurate.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

A visual output is shown in the Ipython notebook.