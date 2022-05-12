# Irradiance-forecast-using-Fish-Eye-images

## Introduction

This repository is part of a 3-month research project - DATA SOPHIA - carried out under the supervision of the research laboratories of the Ecole Mines Paris PSL in Sophia Antipolis. The code was developed as a particular approach to the problem of solar irradiance prediction from hemispheric sky images. This approach is the subject of a scientific article, available at the following link XXX.

## Presentation of the repository

The objective of the code is thus to make a prediction of the solar irradiance in the short term (5 - 20 minutes), from hemispheric images of the sky taken minute by minute.

### Inputs

The various inputs are stored in the /data file.

- The images for prediction are meant to be stored in a subfile related to the specific camera used. Their format is JPEG and their size is 1280x960.
- A csv file is also stored in data (one row per minute / image in the database). It contains necessarly the true mesured solar irradiance (for at least one captor) and various computations (GHI : global horizontal irradiance, Altitude (of the sun), Azimuth (of the sun))