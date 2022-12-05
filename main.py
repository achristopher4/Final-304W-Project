############################################################
############################################################

## Project: Surface Flaw Detection via Image Detection
## Group 5: Alexander Christopher, Werner Hager
## Date: 

############################################################
############################################################

import SteelDataset as sd
import shutil
import Classification as c
import os


train_num = 300
num_segmented = 0


print('#'*60)
print('\nCollecting and Segementing Training Data\n')
training_data = sd.STEELDataAcquisition('TRAIN', train_num, num_segmented)
print('Done\n')
print('#'*60, end = '\n\n\n\n')



train_pos_des = './STEEL_TRAINING/POSITIVE'
train_neg_des = './STEEL_TRAINING/NEGATIVE'



## Create positve samples train folder
print('#'*60)
print('\nCollecting Training Positive Samples\n')
for pos_sample in training_data.training['pos']:
    ## Copy and place into POSITIVE folder
    src = pos_sample[4]
    shutil.copy(src, train_pos_des)
print('Done\n')

## Create negative samples train folder
print('\nCollecting Training Negative Samples\n')
for neg_sample in training_data.training['neg']:
    ## Copy and place into NEGATIVE folder
    src = neg_sample[4]
    shutil.copy(src, train_neg_des)
print('Done\n')
print('#'*60, end = '\n\n\n\n')



## Create positive samples test folder
print('#'*60)
print('\nCollecting Testing Positive Samples\n')
num_pos_training = set(os.listdir('./STEEL_TRAINING/POSITIVE'))
print('Done\n')

## Create negative samples test folder
print('\nCollecting Testing Negative Samples\n')
num_neg_training = set(os.listdir('./STEEL_TRAINING/NEGATIVE'))
print('Done\n')
print('#'*60, end = '\n\n\n\n')



train_pos_path = './STEEL_TRAINING/POSITIVE/'
test_pos_des = './STEEL_TESTING/POSITIVE/'
train_neg_path = './STEEL_TRAINING/NEGATIVE/'
test_neg_des = './STEEL_TESTING/NEGATIVE/'
testing_combo = './STEEL_TESTING/COMBO'

## Create positive samples test folder
print('#'*60)
print('\nCollecting Testing Positive Samples\n')
num_pos_training = list(os.listdir('./STEEL_TRAINING/POSITIVE'))
testing_pos = num_pos_training[-300:]
for img in testing_pos:
    current_path = train_pos_path + img
    shutil.copy(current_path, testing_combo)
    shutil.move(current_path, test_pos_des)
print('Done\n')

## Create negative samples test folder
print('\nCollecting Testing Negative Samples\n')
num_neg_training = list(os.listdir('./STEEL_TRAINING/NEGATIVE'))
testing_neg = num_neg_training[-300:]
for img in testing_neg:
    current_path = train_neg_path + img
    shutil.copy(current_path, testing_combo)
    shutil.move(current_path, test_neg_des)
print('Done\n')
print('#'*60, end = '\n\n\n\n')


# Generate and collect autoencoders outputs
print('#'*60)
print('\nRunning Autoencoder\n')

import Autoencoder_sample as a

print('Done\n')
print('#'*60, end = '\n\n\n\n')



fold = 10
#model = c.Classification(train_num, num_segmented, fold, data)
print('#'*60)
print('\nRunning Classification Model\n')
data = sd.STEELDataAcquisition('TRAIN', train_num, num_segmented)
model = c.Classification(train_num, num_segmented, fold, data)

print('Done\n')
print('#'*60, end = '\n\n\n\n')