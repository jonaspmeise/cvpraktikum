from TrackingPipeline import TrackingPipeline
import cv2
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

class testPipeline(TrackingPipeline):
    def __init__(self, parameters = {}):
    #def __init__(self, filePathVideo, folderPathAnnotation, image_scaling, filter_strength = 10, max_bounding_boxes = 1, history = 100):
        #create standard parameters and update based on given parameters, then propagate it to the parent class
        defaultParameters = self.createDefaultParameters()
        
        for otherParameter in parameters.keys():
            value = parameters.get(otherParameter)
            defaultParameters[otherParameter] = value
             
        super().__init__(parameters = defaultParameters)
        self.framesWithoutChange = 0
        
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        #This feature is bugged...
        self.fgbg.setHistory(self.parameters.get('PARAM_BGSHistory'))
        
        self.previousBoundingBox = None

    def scaleBack(self, number):
        return int(100/self.parameters.get('PARAM_scalePercentage')) * number
        
    def createDefaultParameters(self):
        defaultSettings = {
            #PARAM_maxNumberOfBoundingBoxes = the highest amount of bounding boxes our algorithm should find in a single frame
            'PARAM_maxNumberOfBoundingBoxes': 1,
            #PARAM_scalePercentage = the initial scaling percentage of incoming frames
            'PARAM_scalePercentage': 20,
            #PARAM_lingerFrameCount = the amount of frames after which we dismiss a bounding box if no change appeared throughout this time span 
            'PARAM_lingerFrameCount': 150,
            #PARAM_filterStrength = the amount of iterations of dilations done after we found a foreground mask and before finding the bounding boxes
            'PARAM_filterStrength': 0,
            #PARAM_BGSHistory = the amount of unchanging frames needed before we accept something as background
            #this feature is initially bugged in python/cv2, changing this value does not result in changes
            'PARAM_BGSHistory': 10000,
            #PARAM_initialGaussBlurKernelSize = the kernel size of an initial gaussian blur that we apply on a scaled down input image
            'PARAM_initialGaussBlurKernelSize': 9,
            #PARAM_smoothedBoxAcceptanceFactor = the higher this value, the easier the algorithm will accept new values. 1 should be the lowest value
            'PARAM_smoothedBoxAcceptanceFactor': 5,
        }
        
        return defaultSettings
        
    #previousBoundingBox/currentBoundingBox = [x_min,x_max,y_min,y_max]
    #acceptanceFactor                       = the higher this value, the easier the algorithm will accept new values. 1 should be the lowest value
    #we want to punish "extreme" bounding box jumps, so that we only gradually "move" the bounding box towards the current box
    @staticmethod
    def calculateSmoothBoundingBox(previousBoundingBox, currentBoundingBox, acceptanceFactor = 5): 
        smoothBox = []
        
        for previousValue, currentValue in zip(previousBoundingBox, currentBoundingBox):
            #calculate a smooth box based on previous<->current bounding boxes that adapts to small distances well and huge distances reluctantely
            difference = currentValue-previousValue
            smoothBox.append(previousValue + int(acceptanceFactor/(abs(difference)+acceptanceFactor)*difference))
        
        #print('Mapped the found BB ' + str(currentBoundingBox) + ' -> smoothened box ' + str(smoothBox))
            
        return smoothBox
    
    def getObjectMoments(self, mask):
        currentCentroidCount = -1
        #iterations we did so far when trying to find a maximum amount of boxes: 
        iterations = 0
        currentCentroids = []
        
        #remove initial noise
        editedMask = cv2.erode(mask,np.ones((3,3),np.uint8), iterations = self.parameters.get('PARAM_filterStrength'))
        
        while(currentCentroidCount < 0 or currentCentroidCount > self.parameters.get('PARAM_maxNumberOfBoundingBoxes')):
            #remove noise by dilating
            editedMask = cv2.dilate(editedMask,np.ones((3,3),np.uint8), iterations = iterations)
            
            cv2.imshow('Filtered Centroids', editedMask)
           
            currentCentroids, _ = cv2.findContours(editedMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            currentCentroidCount = len(currentCentroids)
            
            iterations += 1                    
            
        #we pass both the filter strength and the iterations to adapt our resulting bounding boxes respective to the prior filter operations done
        return [self.parameters.get('PARAM_filterStrength'), iterations, currentCentroids]
     
    @staticmethod
    #moments = contours found with cv2 inside a binary mask image
    #max_boxes = maximum amount of boxes that should be found inside a single frame
    #filter = if true, applies max_boxes and only returns the biggest bounding boxes found
    def getBoundingBox(foundMoments):
        offset, iterations_needed, moments = foundMoments
    
        #iterations_needed *= 1.5

        boundingBoxes = []
        for moment in moments:
            x,y,w,h = cv2.boundingRect(moment)
            boundingBoxes.append([x+iterations_needed-offset,y+iterations_needed-offset,w-2*iterations_needed+2*offset,h-2*iterations_needed+2*offset,w*h])  
        #if filter:
        #   return sorted(boundingBoxes, key=itemgetter(4), reverse=True)[:max_boxes]
        #else:
        return boundingBoxes
             
     
    def extractFrame(self, frame):
        frame = self.scale_frame(frame, self.parameters.get('PARAM_scalePercentage'))
        
        #TODO: asspull magic numbers...?
        blurFrame = cv2.blur(frame,(self.parameters.get('PARAM_initialGaussBlurKernelSize'), self.parameters.get('PARAM_initialGaussBlurKernelSize')))
        cv2.imshow("Blurred Frame", blurFrame)
        
        fgMask = self.fgbg.apply(blurFrame)

        cv2.imshow("BackgroundSubtractionMask",fgMask)        
        
        foundBoundingBoxes = self.getBoundingBox(self.getObjectMoments(fgMask))
        
        #we automatically assume that our "base prediction" for the current bounding box is just the previous one
        if self.previousBoundingBox is not None:
            currentBoundingBox = self.previousBoundingBox
        else:
            #very initial value for first frame
            currentBoundingBox = [0,0,0,0]
            
        #initial values
        maxContourLength = 0
        #we check if the bounding box gets changed, if there is no change for too long then we "lost" the object
        boundingBoxChanged = False
            
        for x,y,w,h,_ in foundBoundingBoxes:
            #TODO: right now we only accept a single box -> make it so that more boxes can be found
            if(w+h > maxContourLength):
                maxContourLength = w+h
                currentBoundingBox = [x,y,w,h]
                boundingBoxChanged = True
                
        if boundingBoxChanged:
            self.framesWithoutChange = 0
        else:
            self.framesWithoutChange += 1
        
        #if previousBoundingBox has not been initialzed yet, set it equal to the first found one
        if self.previousBoundingBox is None:
            self.previousBoundingBox = currentBoundingBox   
        else:
            if self.framesWithoutChange > self.parameters.get('PARAM_lingerFrameCount'):
                previousBoundingBox = [0,0,0,0]
                currentBoundingBox = [0,0,0,0]
                self.framesWithoutChange = 0
                
                print('Lost tracked item... :(')
            else:
                currentBoundingBox = self.calculateSmoothBoundingBox(self.previousBoundingBox, currentBoundingBox, self.parameters.get('PARAM_smoothedBoxAcceptanceFactor'))
        
        labelDictionary = {}
        
        #TODO: Put classification here in order to find out which bounding box shows which animal                   
        returnBox = ([currentBoundingBox[0],
            (currentBoundingBox[0]+currentBoundingBox[2]),
            currentBoundingBox[1],
            (currentBoundingBox[1]+currentBoundingBox[3])])
                
        #scale it back to the original frame size
        scaledBack_currentBoundingBox = list(map(self.scaleBack, returnBox))
        labelDictionary = {'baer': scaledBack_currentBoundingBox}        
        
        #set old previous bounding box to current one
        self.previousBoundingBox = currentBoundingBox
        
        return labelDictionary
        
myParameters = {
    #'PARAM_filePathVideo': 'D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7.avi',
    #'PARAM_folderPathAnnotation': 'D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7',
    'PARAM_filePathVideo': 'D:\cv_praktikum\inside',
    'PARAM_folderPathAnnotation': 'D:\cv_praktikum\inside',
    'PARAM_massVideoAnalysis': True,
    #best parameter!
    'PARAM_initialGaussBlurKernelSize': 7,
    #best parameter!
    'PARAM_BGSHistory': 200,
    #best parameter!
    'PARAM_filterStrength': 0,
    #best parameter!
    'PARAM_smoothedBoxAcceptanceFactor': 10
}

myTestPipeline = testPipeline(parameters = myParameters)
myTestPipeline.runPipeline()