# **Traffic Sign Recognition Using Convolutional Neural Networks**
![Cover][image11]
---



[//]: # (Image References)

[image1]: ./report_pics/ori_datagrid.png "Original data grid"
[image2]: ./report_pics/ori_data.png "Original data chart"
[image3]: ./report_pics/aug_datagrid.png "Augmented Data Grid"
[image4]: ./report_pics/aug_data.png " Augmented data chart"
[image5]: ./report_pics/aug.png "Augmentations"
[image6]: ./report_pics/preprocess.png "Preprocess"
[image7]: ./report_pics/web_imgs.png "New imgs"
[image8]: ./report_pics/test_wrong.png "Wrong Predictions"
[image9]: ./report_pics/results.png "Results"
[image10]: ./report_pics/test_datagrid.png "Test"
[image11]: ./report_pics/cover.png "Cover"



### Data Set Summary & Exploration

#### 1. Basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3) = 3 Channel RGB Image of size 32 by 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

To get a better understanding of the type images in the dataset, I plotted out images from a few different categories. The visualizations would provide me the insights into which type of preprocessing techniques I should employ. I also plotted the distribution of the number of images per category as a horizontal bar chart.

![Image visualizations][image1]
![Image Distribution][image2]


---
# __Preprocessing__
#### Observations from data visualization and how they determined the type of preprocessing chosen.

#### Crop
Ideally you only want your model to learn the features that will help it classify traffic signs correctly. However in most of the pictures the traffic sign is centered in the middle and is surrounded by unnecessary visual information that may lead to learning feature maps corresponding to the surrounding areas.So in order to prevent that, I implemented a crop function that zooms into the region of interest and resizes the image to its original size .This resulted in a considerable improvement in the accuracy. This is somewhat of a hardcoded [Spatial Transformer by Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu](https://arxiv.org/abs/1506.02025). I will be implementing this into my model in the future as the current model does not handle images that are very small and far away(as the crop factor is fixed). As seen from the trend in the wrongly predicted images.


#### CLAHE
The training images vary greatly in illumination and contrast. To alleviate this I began by following the pipeline documented by Yan Lecunn in his paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). Although converting to the YUV space and processing the Y channel
with global and local contrast normalization did help, I saw greater accuracy results by implementing CLAHE (Contrast Limited Adaptive Histogram Equalization). Besides improving the contrast and normalizing the illumination CLAHE also helped increase the definition of edges in blurry images. CLAHE is an advancement on AHE. AHE has a drawback of overamplifying noise. CLAHE limits the amplification by clipping the histogram at a predefined value (called clip limit) before computing the CDF.



#### Normalization
Lastly I normalized the image to the bounds of .1 and .9. It helps to lead to faster convergence training Deep Neural Networks and remove any feature specific bias from the equation.


![alt text][image6]


I decided to generate additional data because I wanted to balance out the imbalanced dataset and also ensure the model is robust against perturbations and environmental factors.  

To add more data to the the data set, I used the following techniques because they represent environmental factors that can affect the image and changes in the camera position

 * __Translate__ - To achieve translational invariance.
 * __Rotate__ - To achieve rotational invariance.
 * __Add Noise__ - To represent snow and noisy images. Hopefully by blocking parts of the picture the model becomes more robust to occlusions of the sign. Achieved by replacing random pixels with the median of the surrounding pixels.
 * __Transform Augmentation__ - To represent different camera perspectives. Makes an affine transform using random shear and rotation factors within a normal distribution using μ = 0 and σ = 0.1, This will return an rotated and/or skewed image.

Here is an example of an original image and an augmented image:

![alt text][image5]

The augmented dataset has now 3000 images per class. The augmentations were applied at random from the list of augmentations till they reached the desired amount. Each time an augmentation is applied it's parameters are randomized, ensuring a diverse amount of augmented signs.

![Image visualizations][image3]
![Image Distribution][image4]

#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x16	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Flatten				| outputs 800									|
| Fully connected		| outputs 200     								|
| RELU					|												|
| Dropout				|												|
| Fully connected		| outputs 84     								|
| RELU					|												|
| Dropout				|												|
| Fully connected		| outputs 43     								|
| Softmax				|        								         |





#### 3. Training Description

|Parameter 	| Value 	|
|:-------------:|:-------------:|
| μ (mu)	| 0		|
| σ (sigma)	| 0.1		|
| Epochs	| 30		|
| Batch size	| 64		|
| learning rate	| 0.001		|
| dropout 	| 0.5		|

Utilized the AdamOptimizer with the above mentioned parameters. Training was performed on the augmented dataset. Training took about 30 mins on a Macbook Pro utilizing only the CPU.  

Was not too sure how batch sizes would affect the training accuracy so tried various sizes and found that the size 64 seem to perform best for this model.  [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836).




#### 4. Model Architecture Discussion

My final model results were:
* validation set accuracy of 95.9%
* test set accuracy of 94.5%


My personal objective for this project was to learn what type of data augmentations and data preprocesses affected the test accuracy of the model. So to allow for a quick iteration cycle I started off with the computationally inexpensive LeNet to test my different hypothesis.

Once I was sure of which types of augmentations, preprocesses and its parameters to use I proceeded to change the model. The LeNet model was under-fitting the data as it could not go pass the threshold of 92% for validation accuracy. So I increased the depth of the network to introduce more parameters. This increased the validation accuracy considerably, however signs of overfitting were starting to show as training accuracy would be higher than the validation accuracy. To prevent over fitting I implemented 2 dropout layers on both the fully connected layers. This gave a desirable validation accuracy that consistently hovered around 95%. The dropout layer also helped ensure that the model does not over-rely on certain features and increases it's robustness, especially in cases of occlusions.

In hopes of improving the model I tried implementing the multi-scale model, by concatenating the first layer feature activations to the fully connected layers as documented in the paper by Yan Lecunn. But the difference in accuracy was not noticeable.

After reading a few research papers I decided on trying a deeper architecture, that would be able to better fit the data. I implemented an architecture similar to  [VGG Net](https://www.google.com.sg/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwj6-_HGzajTAhWDFpQKHWXoBaUQFgghMAA&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1409.1556&usg=AFQjCNGCj1kt2G50dIxnPbwC-QmXnL7Mcg&sig2=LVsaRHgcR8jBp7cwigKX_A&cad=rja). However as it was way more computationally expensive, I decided to go with the modified LeNet for the time being as I was able to test out different parameters and find out what worked quickly. Will be trying out the VGG Net on AWS soon.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image7]

I choose theses 6 images because I wanted to see if the data augmentations have actually helped make the model more robust against perturbations.

* The first image - perspective invariance
* The second image - occlusion invariance
* The third image - perspective invariance
* The fourth image - presence of other sign edges
* The fifth image - part of sign is obscured, as half the sign is covered in snow  
* The sixth image - rotational invariance



#### Model Accuracy


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.5%. As these images are relatively easy to classify it is not the most accurate representation of the model. For a more accurate representation of the model we should inspect its shortcoming on a larger test set. Below are some cases from the test set in which the model does not perform as well.

### Images that the model fails to predict
![alt text][image8]

#### Comparing to test set and how to improve accuracy
When comparing the images that failed with the rest of the test set as seen below we can see that the instances that fail on the model seem to be the signs that are small(further from the camera). In order to fix this we could utilizing a spatial transformer to automatically crop the images instead.

Other instances which the model failed to classify
* Blurry Images, could possibly introduce gaussian blur augmentation to the pipeline

* Motion Blur/Double-vision, could possibly try to learn through augmentation.


![alt text][image10]

#### 3. Softmax Probabilities and Classification Confidence

For all images the model was very confident on the result as the probabilities are all 1, with no other probabilities for the other classes.


![alt text][image9]
