# Final-304W-Project

The code can be run by directly running main.py

The following files are defined as follows:
- Autoencoder - steel.py contains the autoencoder architecture used for this project. This file is not ran directly when running main.py, and is only contained to show how the autoencoder is made.
- Autoencoder_sample.py is an example of loading a saved autoencoder model from a file and then using it to reconstruct new data.
- Classification.py contains the functions used by the Classification class which trains and runs the model.
- SteelDataset.py contains the functions used by the STEELDataAcquisition class which acquires data using given splits from the full dataset.
- data_acquisition.py calls STEELDataAcquisition to split the data into positive and negative folders.
- testing.py calls both Classification and STEELDataAcquisition to test the model.
- train_model.py runs Classification to train the model.
- main.py is the central function used to run the overall code and acquire results from the trained model after extracting the needed data.
