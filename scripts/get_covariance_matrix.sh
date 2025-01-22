#!/bin/bash

config_file="/home/joeadamo/Research/SPHEREx/covapt_mt/config/get_covariance.yaml"

# calculate the window function
python make_window_function.py $config_file

# calculate the (Gaussian) covariance matrix
python make_gaussian_covariance.py $config_file