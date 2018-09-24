[//]: # (Image References)

[compute graph]: ./graph.png "Compute Graph"
[conv2d]: ./Conv2D.png "Convolutional 2D operation"
[Validation accuracy graph]: ./accuracy_graph.PNG "Validation Accuracy"
[predictions]: ./predictions.PNG "Predictions"
[conv visualization]: ./layer_visualization.PNG "Conv2D visualization"

## Project: Build a Traffic Sign Recognition Program
Overview
---
In this project, I have used what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I've trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I tried out my model on images of German traffic signs that I found on the web.

To meet specifications, the project consists of below three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

All files are written to meet the [rubric points](https://review.udacity.com/#!/rubrics/481/view)

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualize the convolutional neural network layer
* Summarize the results with a written report

### Python Packages Dependencies
* Pickle : This is for loading the existing dataset pickled
* Numpy : For converting RGB images to gray
* Matplotlib : For visualization purpose
* Sklearn : For shuffling the input data
* Tensorflow : Neural network creation, training

The Implementation
---
### Loading and Visualization of the data set
The [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) is already pickled for us. It comes predivided into three sections.
* train.p : Training dataset consisting of *34799* samples
* valid.p : Validation dataset consisting of *4410* samples
* test.p : Testing dataset consisting of *12630* samples

1. I loaded these datasets in respective variables.
2. I chose randomly which image and its respective label to visualize. This made me know how our image of traffic sign actually looks like
3. All the images present in the train, test, validation set are `32x32x3` size.

### Preprocessing of training set
1. The images/samples under train.p are in `RGB` format. Thus, I changed that to `grayscale` with a very good method found on web. This method is explained [here](https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python).
2. After converting it to `grayscale`, I normalized it with below implementation.
```
a = .1
b = .9
min_data = 0 # Min value in grayscale is 0
max_data = 255 # Max value in grayscale is 255
norm = a + ((rgb - min_data)*(b - a)/(max_data - min_data))
```
3. This preprocessing helped reduce the total size of dataset. This is because, if we use `RGB` color space, to save one image, we need to save data of 3 channels. But to save image in `grayscale`, we need to save data for only one channel. Thus size of our images data set reduces.
4. Once all the training set images are converted into grayscale, we are ready to design a neural network for image classification.

### Network definition
The network is defined in the `model()` function.
The model consists of 
* <b>3</b> `Conv2D` layers
* Every Conv2D layer has a `maxpool`ing layer
* <b>4</b> `Fully Connected` layers

The below graph will describe how the network looks.

![alt text][compute graph]

This graph is generated using [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)

### Optimizers and other supporting functionalities
- For optimization, we're using `AdamOptimizer` already built in tensorflow with the default hardcoded `learning_rate = 0.001`.
- I have written a function `evaluate()` for calculation of our accuracy after every epoch.
- This function returns the accuracy over the passed `X_data` and `y_data` to the `evaluate()` function.

### Training
- Once we are done with the preprocessing and setting up other functionalities, we're ready to run a session for training purpose
- For session, we've set below two hyperparameters for
  - `batch_size = 128`
  - `num_epochs = 50`
- For every epoch, we're finding the validation accuracy and storing it under `validation_accuracy_list` which will later help us plot the validation accuracy graph to see how well we are doing. See below print of epoch #31 for more clarification

```
Training: 31/50
EPOCH 31 ...
Validation Accuracy = 0.957
```
- Once all the epochs are complete we are ready to plot the validation accuracy graph.
- The validation accuracy increases with the number of iterations.
- It increases rapidly in the beginning, but later stays almost constant.

![Validation Accuracy vs Number of Epochs][Validation accuracy graph]

### Saving the trained model
- Our model has run for predefined (50) number of epochs. It has lerned from the training set and it's time to save that learning in some checkpoint so that we can load that later for inferencing part.
- Here while training, if the validation accuracy goes above `0.95`, we're saving the model to `trafficSignClassifier`
- For this, we've used tensorflow's `Saver()` method.

### Testing model on test set
- We had already loaded the test set from `German Traffic Sign Dataset`
- Now, we will load the saved model and run the inferencing on it
- For this purpose, we'll be using the `evaluate()` function to evaluate the accuracy on test set
- For our model, it comes out to be, `0.939`

```
INFO:tensorflow:Restoring parameters from ./trafficSignClassifier
Test accuracy = 0.939
```

Testing model on new test set
---
This is to check the accuracy on the images downloaded from internet.
I've downloaded some images from internet. But these images are of different shapes. So, initially to feed these to our network, I'll have to modify them to a `32x32` format with `grayscale` color.
These images are

<p>
 <img src="./test_images/keep_right.jpg" height="50" width="50">Keep Right
 <img src="./test_images/no_entry.jpg" height="50" width="50">No Entry
 <img src="./test_images/priority_road.jpg" height="50" width="50">Priority Road
 <img src="./test_images/speed_limit_50.jpg" height="50" width="50">Speed Limit 50km/h
 <img src="./test_images/stop.jpg" height="50" width="50">Stop
<br>
</p>

- We predicted these images and the accuracy came out to be <b>100%</b> since all 5/5 images were predicted correctly.

![Predictions][predictions]

Analyzing the model
---
- In the further steps, we're analyzing the model.
- Here, we have used the tensorflow's `top_k()` function to find the top 'k' number of probabilities for inference
- As we're asked to find the top 5 probabilities, it's found and printed in the next cells along with the index prediction

```
Label: keep_right 
 Output: [[1. 0. 0. 0. 0.]] 
 Predicted Label Index: [[38.  0.  1.  2.  3.]]
Label: no_entry 
 Output: [[1. 0. 0. 0. 0.]] 
 Predicted Label Index: [[17.  0.  1.  2.  3.]]
Label: speed_limit_50 
 Output: [[1. 0. 0. 0. 0.]] 
 Predicted Label Index: [[2. 0. 1. 3. 4.]]
Label: priority_road 
 Output: [[1. 0. 0. 0. 0.]] 
 Predicted Label Index: [[12.  0.  1.  2.  3.]]
Label: stop 
 Output: [[1. 0. 0. 0. 0.]] 
 Predicted Label Index: [[14.  0.  1.  2.  3.]]
```

Visualizing the layers (Optional)
---
This is the optional topic to cover in this project.
- For this visualizations, we're already given a function named `outputFeatureMap()` which takes several arguments including the activation stage and  the input image.
- We had to load the saved model in order to start inferencing for one particular layer.
- From the session variable, we used the `get_tensor_by_name()` function which takes in the tensor name which we gave to particular layer while writing the network model and returns its object.
- We sent the image and the tensor we got from previous step to the `outputFeatureMap()` function and it plotted the visualization of layer.
- Here, we've visualized second conv2d layer from our model

![Conv2D Visualization][conv visualization]
