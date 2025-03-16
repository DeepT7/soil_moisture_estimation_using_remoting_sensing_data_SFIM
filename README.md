# Soil Moisture Estimation Using Satellite Images
## Overview
This repository contains code implementations for soil moisture estimation based on the research paper titled **"[Integration of Sentinel-1A Radar and SMAP Radiometer for Soil Moisture Retrieval over Vegetated Areas](https://www.researchgate.net/publication/379454081_Integration_of_Sentinel-1A_Radar_and_SMAP_Radiometer_for_Soil_Moisture_Retrieval_over_Vegetated_Areas)"** by Saeed Arab, Greg Easson, and Zahra Ghaffari. The paper discusses the integration of Sentinel-1A radar data with SMAP radiometer data to improve soil moisture retrieval in vegetated areas.

## Features
- Integration of Sentinel-1A radar data with SMAP radiometer data.
- Implementation of the Smoothing Filter-based Intensity Modulation (SFIM) method for downscaling.
- Use of Artificial Neural Networks (ANN) for improved soil moisture predictions.

## Data preparation

### In-situ soil moisture
Ground soil moisture measurements are available at https://ismn.geo.tuwien.ac.at/en/.

The outputs include: 
- Site-specific files named as network-station including the available daily averaged soil moisutre at 0 - 5 cm of each station;
- A csv file containing the details of each station.

### SMAP data
The SMAP data is available at https://nsidc.org/data/SPL3SMP. A pyhton script can be generated automatically for batch download

### Remote sensing data from Google Earth Engine (GEE)
#### Setup
An google developer account is required to access the GEE

The https://github.com/giswqs/geemap is suggested for the setup of GEE
