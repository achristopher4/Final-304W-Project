import Classification as c
import SteelDataset as sd

import os
import shutil

fold = 10
train_num = 3000
num_segmented = 0

data = sd.STEELDataAcquisition('TRAIN', train_num, num_segmented)
model = c.Classification(train_num, num_segmented, fold, data)
