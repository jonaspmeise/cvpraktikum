
# Table of Contents
- [General](#general)
  * [Important Information](#important-information)
- [Setup](#setup)
  * [Folder Structure](#folder-structure)
  * [Pipeline and Model Setup](#pipeline-and-model-setup)
  * [Test: Evaluation of models on video data](#how-are-the-models-evaluated--how-do-you-get-the-performance-values-)
    + [Single Video](#single-video)
    + [Multiple videos](#multiple-videos)

# General
This github repository contains the programming project which accompanies the computer vision project *Mara & Eisenbär* in the winter semester 2020/2021 at the WWU Münster.

## Important Information

The project requires Python and the following libraries in order to be executed:
``

Numpy

CV2

``

# Setup
The setup for this project is very straightforward. In order to run it, you require \*.avi videos with the format 1920x1080. They should be part of the video corpus acquired from the Zoo Münster, although running the project on other source material is possible, too.

In addition, annotation labels are required in order to assert the preicison of the algorithm. Those labels should be in the Pascal-VOC/XML-Format, which can be created with the usage of [labelImg](https://github.com/tzutalin/labelImg) and applied through all singular frames of a video.



The main Pipeline is included in [TrackingPipeline.py](https://github.com/jonaspmeise/cvpraktikum/blob/main/TrackingPipeline.py) 
