
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
This github repository contains the programming project which accompanies the computer vision project *Mara & Ameisenbär* in the winter semester 2020/2021 at the WWU Münster.

## Important Information

The project requires Python and the following libraries in order to be executed:
```
Numpy
CV2
```

# Setup
The setup for this project is very straightforward. In order to run it, you require \*.avi videos with the format 1920x1080. They should be part of the video corpus acquired from the Zoo Münster, although running the project on other source material is possible, too.

In addition, annotation labels are required in order to assert the preicison of the algorithm. Those labels should be in the Pascal-VOC/XML-Format, which can be created with the usage of [labelImg](https://github.com/tzutalin/labelImg) and applied through all singular frames of a video. 

## Folder Structure
It is best practice to save the labels in a subfolder with the same name as the video where the frames originate from. Those subfolders should be located in the same folder and on the same hierarchy level as their corresponding videos. As long as the names are identical and the subfolders contains valid ``*.xml`` annotation files, there should be no execution problem. 


## Pipeline and Model Setup
The main Pipeline is included in [TrackingPipeline.py](https://github.com/jonaspmeise/cvpraktikum/blob/main/TrackingPipeline.py). This is a general framework for fetching videos and their annotation labels, executing the localisation problem and printing the results. [TrackingPipeline.py](https://github.com/jonaspmeise/cvpraktikum/blob/main/TrackingPipeline.py) should not be edited by itself. 

If you want to apply a certain algorithm method to the pipeline, create a subclass of [TrackingPipeline.py](https://github.com/jonaspmeise/cvpraktikum/blob/main/TrackingPipeline.py), like shown in [BackgroundSubtractorPipeline.py](https://github.com/jonaspmeise/cvpraktikum/blob/main/BackgroundSubtractorPipeline.py).
The only necessary things to implement are the following methods:

- ``__init__(self, parameters = {})``, which should always call ``super().__init__(parameters = defaultParameters)``. It is generally a good idea to first create the default parameters for the current model and then pass them through to the parent class with this call.
- ``extractFrame(self, frame)``, the main method of the algorithm. Here, from the perspective of the pipeline, the algorithm is considered a blackbox: It receives a frame from the pipeline and should return a dictionary with bounding box information. The dictionary format should be the following ``'ANIMAL_NAME'->[BoundingBox_x_min, BoundingBox_x_max, BoundingBox_y_min, BoundingBox_y_max]``
- eventually ``createDefaultParmeters(self)``, depending on whether your model requires external hyperparameters to perform or not. If so, then make this method return a dictionary with entries consisting out of ``'STRING_PARAM_NAME'->DEFAULT_PARAM_VALUE``.

# Test: Evaluation of models on video data
In order to get the performance values of a model on a given set of videos, first either implement a model or use one of the example models available in the github. The final output is generated as a ``*.txt`` file, which will be located in the same folder as  ``PARAM_folderPathAnnotation`` with the video name and the time of creation as its name.

## Single Video
If you want to analyse a single video, set ``PARAM_filePathVideo`` to the absolute path of the video and ``PARAM_folderPathAnnotation`` to the absolute path of the folder containing the ``.xml`` annotation files, then call the method ``runPipeline()`` after initialising the model with the given parameters. 

Here is an example call:

```
myParameters = {
    'PARAM_filePathVideo': 'D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7.avi',
    'PARAM_folderPathAnnotation': 'D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7',
}

myTestPipeline = testPipeline(parameters = myParameters)
myTestPipeline.runPipeline()
```

## Multiple Videos
If you want to analyse multiple videos at once, some changes will need to be made. Instead of a frame-by-frame log-file, only a single log-file containing the grouped information for all videos instead will be generated. For multi-video analysis, change the parameter ``PARAM_massVideoAnalysis`` to ``TRUE``. ``PARAM_filePathVideo`` should no longer point to a single video, but instead to an absolute folder path, where the desired videos are located. Finally, ``PARAM_folderPathAnnotation`` still points to a folder. But that folder should not contain the annotation labels for only a single video, but instead subfolders with identical names as the videos, in which the respective labels are located. Non-matching folders and videos will be skipped in the analysis. Please check the console log to mak sure that the correct video-folder correspondence is found.

Here is an example call:
```
myParameters = {
    'PARAM_filePathVideo': 'D:\cv_praktikum\inside\',
    'PARAM_folderPathAnnotation': 'D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7',
    'PARAM_massVideoAnalysis': True,
}

myTestPipeline = testPipeline(parameters = myParameters)
myTestPipeline.runPipeline()
```
