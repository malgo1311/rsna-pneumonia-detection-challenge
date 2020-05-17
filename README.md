# rsna-pneumonia-detection-challenge

Below is my solution to this kaggle challenge - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview

I am trying this 2 years after the challenge was conducted, so all the numbers I specify below are Stage 2 related only.

#### Stage 2 Public Score - 0.21354 (Not great but in top 25 of the LB :| )

#### All the training and prediction was done on Google Colab using GPU. 

Main objective here was to train an object detection model to identify lung area infected with pneumonia. In an image - the blacker the lungs, the healthier they are. Black area represents air, whereas greyish/whitish area represents fluids and tissues. Let's dive deeper into each step. 

## 1. Data Preparation
There were 26684 images provided for training, and only 30% of these images were pneumonia positive. They had another set of 3000 images for evaluation. The only resource where I could train these many images was Google Colab. So I decided to go ahead with a smaller set of images, so I split my train-test set as follows -

Train - 4000 positive images + 4000 negative images

Test - 400 positive images + 400 negative images

Even though this is an object detection problem, I wanted the model to learn the negative images as well because that reduces the False Positives as essentially the model knows what normal lungs look like. 

The images provided were DICOM images. And they all were gray scale with fixed size of 1024 x 1024, even the test set. As far as I know the OD checkpoints work with 3-channel images, so I decided to first convert these images to jpgs and then stacked the image matrix thrice, the end image size was 1024 x 1024 x 3.

I randomly created the train and test set, and generated the tfrecords using the script 'generate_tfrecords.py'. It also covers including the negative images inside the tfrecords.

## 2. Pretrained Checkpoints
I was doing this exercise as a part of a job interview so I wanted to get back with test result in just a couple of days, so I decided to try just the following two architectures which I had worked on extensively in the past -

1. Faster R-CNN (faster_rcnn_resnet101_kitti_2018_01_28)
2. SSD (ssd_resnet50_v1_fpn)

Finally, Faster R-CNN showed better performance on my test set, so have included the config file that was used.
  
## 3. Training
Nothing much here, since it was my first run, most of the hyperparameters were set to default apart from -

1. Batch Size - 2
2. Image Size - 512 x 512
3. Epoch - 7
4. Preprocessing - None
5. Postprocessing - None

While training from the SSD checkpoint, I realised that we cannot use hard_negative_miner with focal loss, so trained SSD arch without hard_negative_miner. I got much better results on my test set with faster r-cnn, so here chose the bestcheckpoint for final evaluation.

Note: hard_negative_miner config is important when using hard negative examples in training your object detection model.

## 4. Evalutation on Kaggle Test Set
I have directly used the predictions made on the test set without any post-processing. I tried filtering the boxes with score > 0.5 but I got the best result with score > 0.3.

## 5. Thoughts

1. It needs more training as the confidence levels are low
2. Trying out more architectures
3. Using some post-processing - NMS etcs
4. Hyperparameter tuning
5. Preprocessing techniques - cropping the lungs first and then training OD model
6. Augmentation
7. Ensemble Techniques
