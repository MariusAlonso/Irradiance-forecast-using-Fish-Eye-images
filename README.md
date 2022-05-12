# Irradiance-forecast-using-Fish-Eye-images

## Introduction

This repository is part of a 3-month research project - DATA SOPHIA - carried out under the supervision of the research laboratories of the Ecole Mines Paris PSL in Sophia Antipolis. The code was developed as a particular approach to the problem of solar irradiance prediction from hemispheric sky images. This approach is the subject of a scientific article, available at the following link XXX.

## Presentation of the repository

The objective of the code is thus to make a prediction of the solar irradiance in the short term (5 - 20 minutes), from hemispheric images of the sky taken minute by minute.

### Inputs

The various inputs are stored in the /data file.

- The images for prediction are meant to be stored in a subfile related to the specific camera used. Their format is JPEG and their size is 1280x960.
- A csv file, whose rows are timely ordered, minute per minute, and correspond to images in the database. The columns are the following : the true mesured solar irradiance (for at least one captor) ; other various computations (GHI : global horizontal irradiance, Altitude (of the sun), Azimuth (of the sun)). Other columns are not relevant.
- A file named cnn.pth, corresponding to the pre-trained pytorch estimator model (more details below).
- An image mask (format PNG), used to remove background pixels from various incomming computations.
- A generic sun (format PNG), used notably when the sun, hidden up until now behind clouds, is predicted to appear.

### Main files

- main.py is the predictor itself. Its execution leads to a test prediction (8 min of observation - 30 min of prediction) on the small dataset loaded in the repository
- pyramid_of.py computes optical flow between two projected consecutive images, using the pyramidal technique (more robust with respect to large displacements)
- segmentation.py segments the images into clouds, sky, and sun. The difficulty of the task lies in the halo that surrounds the sun, and gets confused with clouds.
- estimator.py is the structure of the estimator neural network (written in pytorch). It is used in combination with the pre-trained stored weights in the file cnn.pth

### Utilities

- open-sans : a police used for outputs
- image_flower.py : a simple shifting and overlapping algorithm used to displace the cloud mask once the CMV field is computed
- projections.py : allows to compute various image projections, including the planar projection, which the final predictor uses

### Outputs

