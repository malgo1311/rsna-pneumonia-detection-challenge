# rsna-pneumonia-detection-challenge

My solution to this kaggle challenge - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview
Public Score - 0.21354

All the training and prediction was done on Google Colab using GPU. Have included the training and the prediction script. Let's dive deeper into each step. 

## 1. Data Preparation

## 2. Pretrained Checkpoints

Decided to intially try two architectures -
i. Faster R-CNN (without hard negative miner)
ii. SSD (without hard negative miner)

Finally, Faster R-CNN showed better performance on my test set, so have included the config file that was used.
  
## 3. Training

## 4. Evalutation on Kaggle Test Set

