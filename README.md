# tfpose-fit
Exercise Applications for OpenPose

## Updates
  * rev0 (19.2.2020): Fall detection and plank analysis

## Background
Aside from using human judgement, physiotherapists or coaches can now use deep learning technologies to complement their judgement in sports performance or rehabilitation progress.

This repository contains Python code that applies the OpenPose algorithm, that can be modified easily to assess different exercises.

## Install required packages
Please follow the guide [here](https://github.com/ildoonet/tf-pose-estimation) on how to install and set up the required packages.

## Usage
*As I have been facing problems with the code not being able to locate the `tensorrt` package, this code does not allow you to view the analysis in real time. Instead, it creates another video file (named `processed_video.mp4` by default, which you can specify in the parser arguments stated below) with the OpenPose analysis shown at the original framerate.*

The main script lies in `run_pose_estimation_#.py`, and should be run in the terminal with the specification of the following arguments:
  * `--video`: directory of the video file containing the exercise demonstration
  * `--model`: default is `mobilenet-thin`, but you may specify `mobilenet_v2_large` / `mobilenet_v2_small`
  * `--tensorrt`: boolean flag to indicate if tensorrt should be used or not. Default is `False`
  * `--resize`: if provided, resize images before they are processed. Default is '*656x368*' but you may choose otherwise. Recommended to be *432x368* or *656x368* or *1312x736*
  * `--video_out`: name of the directory of the OpenPose-processed video. Default is *processed_video.mp4*

For example, in the terminal, run:

`python run_pose_estimation_0.py --video plank.mp4 --model mobilenet_v2_small --tensorrt False --resize 640x480 --video_out plank_processed.mp4`

Follow the instructions in the terminal to select the mode. As of `rev0`, there are 2 modes to choose from; *Fall detection* and *Plank*.


## Acknowledgements
[Kim Ildoo](https://github.com/ildoonet) for the main tf-pose algorithm.
