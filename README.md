# Irradiance-forecast-using-Fish-Eye-images

## Introduction

This repository is part of a 3-month research project - DATA SOPHIA - carried out under the supervision of the research laboratories of the Ecole Mines Paris PSL in Sophia Antipolis. The code was developed as a particular approach to the problem of solar irradiance prediction from hemispheric sky images. This approach is the subject of a scientific article, available at the following link XXX.

## General purpose of the code

The objective of the code is to make a short term (5 to 30 minutes) prediction of solar irradiance, based on hemispherical sky images taken minute by minute.

In order to do so, a predictor algorithm is constructed. It works in two steps :
- First, the current sky image is propagated : a prediction of what the sky will look like in one minute is performed. For that, the clouds are segmented from the sky image, a cloud motion vector field is computed, and eventually the segmented clouds are displaced according to this field.
- Then, an estimator trained to compute solar irradiance to corresponding sky image is put to use. The estimator, in the form of a Multi-Layer Perceptron, outputs the desired solar irradiance.

This process is repeated minute by minute.

## File description

### Inputs

The various inputs are stored in the /data file.

- The images for prediction are meant to be stored in a subfile related to the specific camera used. Their format is JPEG and their size is 1280x960.
- A csv file, whose rows are timely ordered, minute per minute, and correspond to images in the database. The columns are the following : the true measured solar irradiance (for at least one captor) ; other various computations (GHI : global horizontal irradiance, Altitude of the sun, Azimuth of the sun). Other columns are not relevant.
- A file named cnn.pth, corresponding to the pre-trained pytorch estimator model.
- An image mask (format PNG), used to remove background pixels for various incomming computations.
- A generic sun (format PNG), used notably when the sun, hidden up until now behind clouds, is predicted to appear.

### Main files

- main.py is the predictor itself. Its execution leads to a test prediction (8 min of observation and 30 min of prediction) on the small dataset loaded in the repository. An observation phase of about 10 minutes / 10 images is recommended during the execution of the predictor. Each image is segmented, and a clear sky model can thus be progressively built up (provided the blue sky appears during the obervation time ...).
- pyramid_of.py computes optical flow between two projected consecutive images, using the pyramidal technique, which is more robust than standard optical flow when it comes to large displacements.
- segmentation.py segments the images into clouds, sky, and sun. The difficulty of the task lies in the halo that surrounds the sun, and gets confused with clouds.
- estimator.py is the structure of the estimator neural network (written in pytorch). It is used in combination with the pre-trained stored weights in the file cnn.pth.

### Utilities

- open-sans : a police used for outputs
- image_flower.py : a simple shifting and overlapping algorithm used to displace the cloud mask once the CMV field is computed
- projections.py : allows to compute various image projections, including the planar projection, which the final predictor uses

### Outputs

- results/prediction/ : results of the prediction, displayed as minute by minute images depicting the actual observed sky to the left and the predicted sky to the right (with the CMV field overlaid), and incorporating a graph of the solar irradiance predicted and real values.
- results/optical_flow/ : results of the minute by minute CMV computation.
- results/cloudmask_propagated/ : results of the minute by minute dispacement of the cloud mask.
- results/bluesky_model/ : results of the progressive built up of the blue sky background during observation phase.

## Known issues

Due to lack of time, the code remained rather messy, not very robust and not very portable. The predictor obtains remarkable results on some situations, but fails in many others (too cloudy, veiled sky, advection phenomena, ...). Moreover, the code is probably still riddled with bugs.

## Acknowledgements

I would like to thank Youri Tchouboukoff, who provided me with a segmenter code he developed back during winter 2020-2021. It has proved well on the dataset I used, and served for me as a starting point and guideline for my own segmenter (a much less robust version, simpler and tailored to the goals of my predictor).

I would also like to thank SOLAÏS, which provided me with the images and associated irradiance data over several months of measurements, essential for training the estimator and building the predictor. The small dataset uploaded in the directory is a sample of the complete dataset provided by SOLAÏS. All rights reserved to them.