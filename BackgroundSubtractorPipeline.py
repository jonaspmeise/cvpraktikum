from TrackingPipeline import TrackingPipeline
import cv2
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

class testPipeline(TrackingPipeline):
    def __init__(self, filePathVideo, folderPathAnnotation, image_scaling, filter_strength = 10, max_bounding_boxes = 1, history = 100):
        super().__init__(filePathVideo, folderPathAnnotation)
        self.percentage = 20
        self.image_scaling = image_scaling
        self.max_bounding_boxes = max_bounding_boxes
        self.filter_strength = filter_strength
        
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.fgbg.setHistory(history)
        
        self.previousBoundingBox = None

    def scaleBack(self, number):
        return int(100/self.percentage) * number
        
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
        
        print('Mapped the found BB ' + str(currentBoundingBox) + ' -> smoothened box ' + str(smoothBox))
            
        return smoothBox
     
    @staticmethod
    def getObjectMoments(mask, noise_cancelling_iterations = 1, max_bounding_boxes = 1):
        currentCentroidCount = -1
        #iterations we did: 
        iterations = 0
        currentCentroids = []
        
        #remove initial noise
        editedMask = cv2.erode(mask,np.ones((3,3),np.uint8),iterations = noise_cancelling_iterations)
        
        while(currentCentroidCount < 0 or currentCentroidCount > max_bounding_boxes):
            #remove noise by dilating
            editedMask = cv2.dilate(editedMask,np.ones((3,3),np.uint8),iterations = iterations)
            
            cv2.imshow('Filtered Centroids', editedMask)
           
            currentCentroids, _ = cv2.findContours(editedMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            currentCentroidCount = len(currentCentroids)
            
            iterations += 1
            
        #iterations-1 because we count the "erode-step" as a -1 and the "dilate-step" as +1
        return iterations-1, currentCentroids
     
    @staticmethod
    #moments = contours found with cv2 inside a binary mask image
    #max_boxes = maximum amount of boxes that should be found inside a single frame
    #filter = if true, applies max_boxes and only returns the biggest bounding boxes found
    def getBoundingBox(foundMoments, max_boxes, filter=True):
        iterations_needed, moments = foundMoments
    
        boundingBoxes = []
        for moment in moments:
            x,y,w,h = cv2.boundingRect(moment)
            boundingBoxes.append([x+iterations_needed,y+iterations_needed,w-2*iterations_needed,h-2*iterations_needed,w*h])  
        if filter:
            return sorted(boundingBoxes, key=itemgetter(4), reverse=True)[:max_boxes]
        else:
            return boundingBoxes
             
     
    def extractFrame(self, frame):
        frame = self.scale_frame(frame, self.percentage)
        
        fgMask = self.fgbg.apply(cv2.blur(frame,(3,3)))

        cv2.imshow("BackgroundSubtractionMask",fgMask)
        
        foundBoundingBoxes = self.getBoundingBox(self.getObjectMoments(fgMask, self.filter_strength, self.max_bounding_boxes), self.max_bounding_boxes, filter=True)
        
        labelDictionary = {}
        
        #initial values
        maxContourLength = 0
        
        #we automatically assume that our "base prediction" for the current bounding box is just the previous one
        if self.previousBoundingBox is not None:
            currentBoundingBox = self.previousBoundingBox
        else:
            #very initial value for first frame
            currentBoundingBox = [0,0,0,0]
            
        #draw our own found bounding boxes
        for x,y,w,h,_ in foundBoundingBoxes:
            #TODO: right now we only accept a single box -> make it so that more boxes can be found
            if(w+h > maxContourLength):
                maxContourLength = w+h
                currentBoundingBox = [x,y,w,h]
                  
        #if previousBoundingBox has not been initialzed yet, set it equal to the first found one
        if self.previousBoundingBox is None:
            self.previousBoundingBox = currentBoundingBox   
        else:
            currentBoundingBox = self.calculateSmoothBoundingBox(self.previousBoundingBox, currentBoundingBox)
                        
        self.previousBoundingBox = currentBoundingBox
        #TODO: Put classification here in order to find out which bounding box shows which animal                   
        returnBox = ([currentBoundingBox[0],
            (currentBoundingBox[0]+currentBoundingBox[2]),
            currentBoundingBox[1],
            (currentBoundingBox[1]+currentBoundingBox[3])])
                
        #scale it back to the original frame size
        scaledBack_currentBoundingBox = list(map(self.scaleBack, returnBox))
        labelDictionary = {'baer': scaledBack_currentBoundingBox}        
        
        return labelDictionary
        
myTestPipeline = testPipeline('D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7.avi','D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7', 20, filter_strength = 0, max_bounding_boxes=1, history=500)
myTestPipeline.runPipeline()