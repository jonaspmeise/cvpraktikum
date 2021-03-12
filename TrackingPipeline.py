from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import re

'''
---Pipeline für Mara/Ameisenbär-Tracking als abstrakte Klasse---
!!! TODO: Die Methode extractFrame(self, frame) muss implementiert werden !!!

Pipeline:
    - load video
    - fetch frame
    - start loop of:
        - fetching new frame
        - fetch annotations for current frame (if they exist)
        - !!! TODO: extractFrame to calculate the bounding boxes based on an implemented model !!!
        
        - draw bounding boxes from both the annotation and from the found ones by the implemented model
        - compare the found bounding boxes with the ones from the annotation
        - calculate the difference of bounding boxes through an error function
    - return statistics overview and dump the results into a .txt
'''

class TrackingPipeline(ABC):  
    #boundingBoxes: dictionary with key=frame -> value=annotation as XML-Tree
    boundingBoxes = {}
    frameCounter = 0
    
    #frameErrorDic: dictionary with key=frame -> value=[calculated error between generated/annotated bounding boxes, listOfPerformanceValues]
    #                   with listOfPerformanceValues = [amountOfTruePositives, amountOfFalsePositives, amountOfFalseNegatives]
    frameErrorDic = {}
    
    
    def __init__(self, filePathVideo, folderPathAnnotation):
        super().__init__()
        
        #create video stream
        if os.path.exists(filePathVideo):
            self.filePathVideo = filePathVideo 
            self.cap = cv2.VideoCapture(filePathVideo)
            
            print('Loading video from ' + filePathVideo + ' ...')
        else:
            raise OSError('Error while trying to open the video file: ' + filePathVideo)
        
        #read and fetch annotations
        if os.path.isdir(folderPathAnnotation):
            print('Loading annotations from folder ' + folderPathAnnotation + ' ...')
        
            self.loadBoundingBoxes(folderPathAnnotation)
        else:
            raise OSError('Error while trying to open the folder: ' + folderPathAnnotation)
        
    #load bounding boxes from annotations
    def loadBoundingBoxes(self, annotationFolder):
        if annotationFolder is not None:
            for filename in os.listdir(annotationFolder):
                if filename.endswith('.xml'):
                    fullname = os.path.join(annotationFolder, filename)
                    tree = ET.parse(fullname)
                    
                    #some regex to extract the frameNumber
                    frameNumber = int(re.findall(r'\d+', tree.find('filename').text)[0])
                    self.boundingBoxes[frameNumber] = tree
                    
                    print('Found annotation for frame #' + str(frameNumber))
    
    #returns the video capture object
    def getCap(self):
        return self.cap

    #fetch next frame
    def fetchFrame(self):
        success, frame = self.getCap().read()
        if success:
            self.frameCounter += 1
            return frame
        else:
            print('Error while fetching!')
    
    #returns dictionary with animal names as keys ('mara','mara1','mara2','baer') for a given annotation (of a single frame)
    #values in the dictionary are bounding box informations for given annotated frame: x_min, x_max, y_min, y_max
    @staticmethod
    def annotationToData(singleFrameAnnotationXML):
        returnDict = {}
        
        #find all objects inside the XML
        for object in singleFrameAnnotationXML.getroot().findall('object'):
            #get coordinates save them in the dictionary under the name of the animal/annotation label
            x_min = int(object[4][0].text)
            y_min = int(object[4][1].text)
            x_max = int(object[4][2].text)
            y_max = int(object[4][3].text)
            
            returnDict[object[0].text] = [x_min, x_max, y_min, y_max]
            
        return returnDict
    
    #draw bounding boxes for animals based on a dictionary into a frame:
    #dictionary: each animal has bounding box values ('x_min', 'x_max', 'y_min', 'y_max') with its name ('mara','mara1','mara2','baer') as the key in the dictionary
    @staticmethod
    def drawBoundingBoxesForAnimals(dictionary, frameToDraw, userGeneratedData = False):
        mara = dictionary.get('mara')
        mara1 = dictionary.get('mara1')
        mara2 = dictionary.get('mara2')
        baer = dictionary.get('baer')
        
        #TODO: Set better values for the colors of the bounding boxes
        if mara is not None:
            cv2.rectangle(frameToDraw,(mara[0],mara[2]),(mara[1],mara[3]),(int(userGeneratedData)*125,int(not userGeneratedData)*125,0),2)    
        if mara1 is not None:
            cv2.rectangle(frameToDraw,(mara1[0],mara1[2]),(mara1[1],mara1[3]),(int(userGeneratedData)*255,int(not userGeneratedData)*255,0),2)
        if mara2 is not None:
            cv2.rectangle(frameToDraw,(mara2[0],mara2[2]),(mara2[1],mara2[3]),(int(userGeneratedData)*255,125,0),2)
        if baer is not None:
            cv2.rectangle(frameToDraw,(baer[0],baer[2]),(baer[1],baer[3]),(int(userGeneratedData)*255,int(not userGeneratedData)*255,255),2)
            
        return frameToDraw
    
    #the main tracking process
    def runPipeline(self):
        while(1):
            frame = self.fetchFrame()    
            
            if frame is not None:                   
                #get annotation (XML-Tree) for current frame as an dictionary object or None if there is no annotation
                annotationForCurrentFrame = self.boundingBoxes.get(self.frameCounter)
                
                #generate annotation with the user-implemented model
                userGeneratedAnnotation = self.extractFrame(frame)
                
                #if there are bounding boxes in the frame annotations
                if annotationForCurrentFrame is not None:
                    #parse XML-Tree to Dictionary-Object
                    annotationForCurrentFrame = self.annotationToData(annotationForCurrentFrame)
                
                    #draw the bounding box from the annotation into the frame
                    frame = self.drawBoundingBoxesForAnimals(annotationForCurrentFrame, frame, False)
                
                #if there are bounding boxes that the user generated
                if userGeneratedAnnotation is not None:
                    #draw the bounding box from the annotation into the frame
                    frame = self.drawBoundingBoxesForAnimals(userGeneratedAnnotation, frame, True)
                    
                if annotationForCurrentFrame is not None:
                    currentError = self.calculateError(annotationForCurrentFrame, userGeneratedAnnotation)
                    self.frameErrorDic[self.frameCounter] = currentError
                    
                    print('Found error of ' + str(currentError) + ' for frame #' + str(self.frameCounter) + '.');
                
                cv2.imshow("Annotated Frame", frame)
                    
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
        
        #print all the error into a log file
        logFilePath = self.filePathVideo + '_' + datetime.now().strftime("%d%m%Y_%H%M%S") + '_log.txt'
        self.writeLog(logFilePath)
        
        cv2.destroyAllWindows()
    
    def writeLog(self, logFilePath):
        logFileContent = ''
        averageError = 0
        tps = 0
        fps = 0
        fns = 0
        
        for errorFrame in self.frameErrorDic.keys(): 
            errorSummary = self.frameErrorDic.get(errorFrame)
            
            tps += errorSummary[1][0]
            fps += errorSummary[1][1]
            fns += errorSummary[1][2]
            
            logFileContent += ('#' + str(errorFrame) 
                + ' TP=' + str(errorSummary[1][0]) 
                + ' FP=' + str(errorSummary[1][1]) 
                + ' FN=' + str(errorSummary[1][2]) 
                + ' TotalError=' + str(errorSummary[0]) + '\n')
                
            averageError += errorSummary[0]
        
        averageError = averageError / len(self.frameErrorDic.keys())
        
        with open(logFilePath, 'w') as logFile:
            logFile.write('AverageError=' + str(averageError) + '\n' 
                + 'TruePositives=' + str(tps) + '\n'  
                + 'FalsePositives=' + str(fps) + '\n'
                + 'FalseNegatives=' + str(fns) + '\n'
                + 'F-Score=' + str(2*(tps / (tps + 0.5*(fps+fns)))) + '\n\n'
                + logFileContent)
            
        print('Log file was written to \"' + logFilePath + '\"')   
        
    
    #calculates the distance between found bounding box corners between the true annotations and the user-generated data
    #returns:   [calculated error between generated/annotated bounding boxes, listOfPerformanceValues]
    #               with listOfPerformanceValues = [amountOfTruePositives, amountOfFalsePositives, amountOfFalseNegatives]
    #
    #parameter: annotationBoxes = the bounding box information of the (true) annotated labels
    #           generatedBoxes = the bounding box information of the user-generated labels
    #           TODO:   enforceDifferenceBetweenMaras = If true, mara1 and mara2 will be differentiated and responsively get an error score
    #                                           For false, mara1 and mara2 will not cause an mismatch
    #                                           Mara is always matched to both mara1 and mara2
    def calculateError(self, annotationBoxes, generatedBoxes, enforceDifferenceBetweenMaras = True):
        error = 0
        
        #Create empty dictionaries in case they are 'None', otherwise we can't use operations on it
        if annotationBoxes is None:
            annotationBoxes = {}
            
        if generatedBoxes is None:
            generatedBoxes = {}
        
        truePositives = set.intersection(set(annotationBoxes.keys()), set(generatedBoxes.keys()))
        falseNegatives = len(annotationBoxes) - len(truePositives)
        falsePositives = len(generatedBoxes) - len(truePositives)
        
        for animalKey in truePositives:
            bbAnnotation = annotationBoxes.get(animalKey)
            bbGenerated = generatedBoxes.get(animalKey)
            
            for value_annotation, value_generated in zip(bbAnnotation, bbGenerated):
                error += self.dist_lp(value_annotation, value_generated)
                
        return [error, [len(truePositives), falsePositives, falseNegatives]]
    
    #calculates distance as absolute difference
    @staticmethod
    def dist_lp(p1, p2):
        return abs(p1 - p2)        
        
    # !!! TODO: This method should implement a model which returns a dictionary of bounding boxes (refer 'to annotationToData()') for all found labels 
    #           or None, if no labels were found for the given frame
    # !!!
    # For implementing a method, create a sub-class of TrackingPipeline and properly implement this method, including the return value
    #
    #returns dictionary with keys=animal names ('mara','mara1','mara2','baer') -> values=boundingbox information [x_min, x_max, y_min, y_max]
    @abstractmethod
    def extractFrame(self, frame):
        pass