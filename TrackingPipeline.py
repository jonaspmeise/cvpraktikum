from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import re
from timeit import default_timer as timer

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
    #parameters for all kind of functions
    parameters = {}
    
    #list of videos that we are going to track and their according annotation folders
    videoAnnotationPathList = []
    
    #boundingBoxes: dictionary with key=frame -> value=annotation as XML-Tree
    boundingBoxes = {}
    frameCounter = 0
    
    #frameErrorDic: dictionary with key=frame -> value=[calculated error between generated/annotated bounding boxes, listOfPerformanceValues]
    #                   with listOfPerformanceValues = [amountOfTruePositives, amountOfFalsePositives, amountOfFalseNegatives]
    frameErrorDic = {}
    frameErrorDicList = []
      
    def __init__(self, parameters = {}):
        super().__init__()
        
        #load default parameters and then update them based on given parameter settings
        self.parameters = self.createDefaultParameters()
        for otherParameter in parameters.keys():
            value = parameters.get(otherParameter)
            print('Parameters: ' + str(otherParameter) + ' -> ' + str(value))
            self.parameters[otherParameter] = value
        
        if not self.parameters.get('PARAM_massVideoAnalysis'):
            #create video stream
            if os.path.exists(self.parameters.get('PARAM_filePathVideo')):
                self.videoAnnotationPathList.append([self.parameters.get('PARAM_filePathVideo'), self.parameters.get('PARAM_folderPathAnnotation')])
            else:
                raise OSError('Error while trying to open the video file: ' + self.parameters.get('PARAM_filePathVideo'))
        else:
            #fetch all *.avi files in the given folder
            for filename in os.listdir(self.parameters.get('PARAM_folderPathAnnotation')):
                if filename.endswith('.avi'):
                    fullname = os.path.join(self.parameters.get('PARAM_filePathVideo'), filename)
                    
                    #try to fetch the annotation folder corresponding to the video name from the 'PARAM_folderPathAnnotation'
                    nameWithoutExtension = os.path.splitext(fullname)[0]
                    
                    folderPath = os.path.join(self.parameters.get('PARAM_folderPathAnnotation'), nameWithoutExtension)
                    if os.path.isdir(folderPath):
                        self.videoAnnotationPathList.append([fullname, folderPath])
                    else:
                        print('Found video ' + filename + ', but ' + str(folderPath) + ' is not a viable folder...')
            
        print('----- Found ' + str(len(self.videoAnnotationPathList)) + ' videos and annotation folders:\n' + str(self.videoAnnotationPathList))
        
    #creates the default settings for all parameters
    def createDefaultParameters(self):
        defaultSettings = {
            #PARAM_enforceDifferenceBetweenMaras = If true, mara1 and mara2 will be differentiated and responsively get an error score
            #   For false, mara1 and mara2 will not cause an mismatch and their error score will be calculated (if there are mara1&mara2 bb in both pictures, match them so that they have the lowest error)
            'PARAM_enforceDifferenceBetweenMaras1And2': False,
            #PARAM_massVideoAnalysis = If true, it will analyse all videos in a given folder 'PARAM_filePathVideo' and its subfolders while trying to find fitting annotations for each video in ''PARAM_folderPathAnnotation' and its subfolders for each video
            #   If false, please set 'PARAM_filePathVideo' and 'PARAM_folderPathAnnotation' correctly.
            'PARAM_massVideoAnalysis': False,
            #PARAM_filePathVideo should point to a folder where the videos are located if 'PARAM_massVideoAnalysis' is true
            'PARAM_filePathVideo': 'video.avi',
            #PARAM_folderPathAnnotation points to the folder where in its subfolders the *.xml annotations are located if 'PARAM_massVideoAnalysis' is true
            'PARAM_folderPathAnnotation': '.\annotation'
        }
        
        return defaultSettings
        
    #load bounding boxes from annotations
    def loadBoundingBoxes(self, annotationFolder):
        boundingBoxDic = {}
    
        if annotationFolder is not None:
            for filename in os.listdir(annotationFolder):
                if filename.endswith('.xml'):
                    fullname = os.path.join(annotationFolder, filename)
                    tree = ET.parse(fullname)
                    
                    #some regex to extract the frameNumber
                    frameNumber = int(re.findall(r'\d+', tree.find('filename').text.split("frame")[1])[0])
                    boundingBoxDic[frameNumber] = tree
                    
                    print('Found annotation for frame #' + str(frameNumber))
    
        return boundingBoxDic
        
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
            return None
    
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
                
        start_pipeline = timer()
                
        #cycle through all videos and feed them into the cap
        for videoPath, annotationFolderPath in self.videoAnnotationPathList:
            self.cap = cv2.VideoCapture(videoPath)
                        
            print('----- Loading video from ' + videoPath + ' ...')
          
            print('----- ... Starting to fetch annotations from folder ' + annotationFolderPath + ' ...')
            
            boundingBoxDic = self.loadBoundingBoxes(annotationFolderPath)
        
            #generate an empty dictionary with error-frame descriptions:
            frameErrorDic = {}
            
            #we start to count from frame 0 every time
            currentFrameCounter = 0
        
            while(1):          
                frame = self.fetchFrame()    
                
                if frame is not None:         

                    currentFrameCounter += 1
                    #get annotation (XML-Tree) for current frame as an dictionary object or None if there is no annotation
                    annotationForCurrentFrame = boundingBoxDic.get(currentFrameCounter)
                    
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
                        print('Generated annotation ' + str(userGeneratedAnnotation) + ' in frame ' + str(currentFrameCounter))
                        frame = self.drawBoundingBoxesForAnimals(userGeneratedAnnotation, frame, True)
                        
                    if annotationForCurrentFrame is not None:
                        currentError = self.calculateError(annotationForCurrentFrame, userGeneratedAnnotation)
                        frameErrorDic[currentFrameCounter] = currentError
                        
                        print('Found error of ' + str(currentError) + ' for frame #' + str(currentFrameCounter) + '.');
                    
                    cv2.imshow("Annotated Frame", frame)  
                    
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break
                else:
                    break
            
            #add the frameErrorDic to our list of dictionaries for later analysis:
            self.frameErrorDicList.append([videoPath, frameErrorDic])
                
            cv2.destroyAllWindows()
                
        end_pipeline = timer()
        
        logFilePath = self.parameters.get('PARAM_folderPathAnnotation') + '\\' + datetime.now().strftime("%d%m%Y_%H%M%S") + '_log.txt'
        #print all the error into a log file
        self.writeLog(logFilePath, end_pipeline-start_pipeline)
    
    def writeLog(self, logFilePath, elapsedTime):
        #for calculating stats througout all videos
        logFileContent = ''
        total_averageError = 0
        total_tps = 0
        total_fps = 0
        total_fns = 0
        total_error_frames = 0
            
        #cycle through all single videos
        for videoPath, frameErrorDic in self.frameErrorDicList:
            averageError = 0
            tps = 0
            fps = 0
            fns = 0
            
            for errorFrame in frameErrorDic.keys(): 
                errorSummary = frameErrorDic.get(errorFrame)
                
                tps += errorSummary[1][0]
                fps += errorSummary[1][1]
                fns += errorSummary[1][2]
                
                #ignore the frame-by-frame analysis for mass video analysis
                if not self.parameters.get('PARAM_massVideoAnalysis'):
                    logFileContent += ('#' + str(errorFrame) 
                        + ' TP=' + str(errorSummary[1][0]) 
                        + ' FP=' + str(errorSummary[1][1]) 
                        + ' FN=' + str(errorSummary[1][2]) 
                        + ' TotalError=' + str(errorSummary[0]) + '\n')
                    
                averageError += errorSummary[0]
            
            numberOfErrorEntries = len(frameErrorDic.keys())
            
            #we may have a division by zero: the +1 does not matter since the error is only used as a relative measure
            averageErrorResult = 'not defined'
            if numberOfErrorEntries > 0:
                #remember them for calculating the total score for ALL videos
                total_averageError += averageError
                total_error_frames += numberOfErrorEntries
                
                averageErrorResult = averageError / numberOfErrorEntries
            
            #dodge another division by zero mistake, which can appear if we have 0 TPs
            fScore = 'not defined'
            if tps > 0:
                fScore = tps / (tps + 0.5*(fps+fns))
        
            total_tps += tps
            total_fps += fps
            total_fns += fns
        
            #logFileContent for single video analysis
            logFileContent = (logFileContent + '\n'
                + '--------------------------------\n'
                + '----- VIDEO=' + videoPath + '\n'
                + 'AverageError=' + str(averageErrorResult) + '\n' 
                + 'TruePositives=' + str(tps) + '\n'  
                + 'FalsePositives=' + str(fps) + '\n'
                + 'FalseNegatives=' + str(fns) + '\n'
                + 'F-Score=' + str(fScore) + '\n')
        
        #paste the culmination of all video analysis into a single analysis, add all the single video analysis and dump the entire thing
        with open(logFilePath, 'w') as logFile:
            #dodge division by zero
            if total_error_frames > 0:
                total_averageError = total_averageError / total_error_frames
            else:
                total_averageError = 'not defined'
            
            #same here
            fScore = 'not defined'
            if tps > 0:
                fScore = total_tps / (total_tps + 0.5*(total_fps+total_fns))
            
            #ignore cross-video analysis section if we only have a single video
            #if self.parameters.get('PARAM_massVideoAnalysis')):
            logFileContent = ('TotalAverageError=' + str(total_averageError) + '\n' 
                + 'TotalTruePositives=' + str(total_tps) + '\n'  
                + 'TotalFalsePositives=' + str(total_fps) + '\n'
                + 'TotalFalseNegatives=' + str(total_fns) + '\n'
                + 'TotalF-Score=' + str(fScore) + '\n'
                + 'FramesPerSecond=' + str(self.frameCounter/elapsedTime) + '\n\n'
                + 'Parameters=' + str(self.parameters) + '\n\n\n'
                + logFileContent)
        
            logFile.write(logFileContent)
            
        self.parameters = {}
        self.videoAnnotationPathList = []
        self.boundingBoxes = {}
        self.frameCounter = 0      
        self.frameErrorDic = {}
        self.frameErrorDicList = []
            
        print('Log file was written to \"' + logFilePath + '\"')   
        
    
    #calculates the distance between found bounding box corners between the true annotations and the user-generated data
    #returns:   [calculated error between generated/annotated bounding boxes, listOfPerformanceValues]
    #               with listOfPerformanceValues = [amountOfTruePositives, amountOfFalsePositives, amountOfFalseNegatives]
    #
    #parameter: annotationBoxes = the bounding box information of the (true) annotated labels
    #           generatedBoxes = the bounding box information of the user-generated labels
    #                                           "mara" tag should only appear as a substitute to "mara1" or "mara2"; never in combination with one of them
    #                                           >   If "mara" appears in both pictures, calculate the respective error
    #                                           >   If "mara" appears in one picture and "mara1"/"mara2" in the other, match to that
    #                                           >   If "mara" appears in one picture and "mara1","mara2" in the other, only calculate the lower error
    def calculateError(self, annotationBoxes, generatedBoxes):
        error = 0
        
        #if we match stuff which is not included in the pure intersection of both dictionary keys
        additionalTruePositives = 0
        
        #Create empty dictionaries in case they are 'None', otherwise we can't use operations on it       
        if generatedBoxes is None:
            generatedBoxes = {}
        
        truePositives = set.intersection(set(annotationBoxes.keys()), set(generatedBoxes.keys()))
        falseNegatives = len(annotationBoxes) - len(truePositives)
        falsePositives = len(generatedBoxes) - len(truePositives)
        
        print('generatedBoxes : ' + str(generatedBoxes.keys()))
        print('true annotationBoxes : ' + str(annotationBoxes.keys()))
        print('True Positives => ' + str(truePositives))
        
        #calculate the mara-mara1-mara2 mismatches manually, only if XOR appearance as keys in both bounding boxes
        #TODO: comment: very ugly code, but I don't think there is an easy way to solve this
        if ('mara' in annotationBoxes.keys()) != ('mara' in generatedBoxes.keys()):
            print('Matching Mara -> Mara1/Mara2')
            #the keys that were not matched 
            nonMatchedKeysAnnotation = set(annotationBoxes.keys())-truePositives
            nonMatchedKeysGenerated = set(annotationBoxes.keys())-truePositives
            
            #TODO: Fix this horrible coding mess
            for current, other in ([[[nonMatchedKeysAnnotation, annotationBoxes],
                [nonMatchedKeysGenerated, generatedBoxes]],
                [[nonMatchedKeysGenerated, generatedBoxes],
                [nonMatchedKeysAnnotation, annotationBoxes]]]):
                
                #see which set belongs to which dic to later make sure which dictionary we access
                currentSet = current[0]
                currentDic = current[1]
                otherSet = other[0]
                otherDic = other[1]
                
                #suche nach matching von 'mara'(in current set) -> 'mara1/2'(in other set)
                if 'mara' in currentSet:
                    #no idea how to fix this "elegantly" right now
                    initialValue = 10000000000
                    mara1error = initialValue
                    mara2error = initialValue
                    
                    #see if there is mara1 or mara2 in the the other set
                    if 'mara1' in otherDic.keys():
                        #TODO: fix this math
                        mara1error = self.calculateErrorBetweenBoundingBoxes(currentDic.get('mara'), otherDic.get('mara1'))
                        print('mara1error: ' + str(mara1error))
                    if 'mara2' in otherDic.keys(): 
                        mara2error = self.calculateErrorBetweenBoundingBoxes(currentDic.get('mara'), otherDic.get('mara2'))
                        print('mara2error: ' + str(mara2error))
                     
                    #sanity check: did we find a real bounding box error?
                    if mara1error is not initialValue or mara2error is not initialValue:
                        if mara1error <= mara2error:
                            error += mara1error
                        else:
                            error += mara2error
                        
                        #since we managed to find a matching which would otherwise not be matched
                        falseNegatives -= 1
                        falsePositives -= 1
                        additionalTruePositives += 1
            
        
        for animalKey in truePositives:
            bbAnnotation = annotationBoxes.get(animalKey)
            bbGenerated = generatedBoxes.get(animalKey)
            
            error += self.calculateErrorBetweenBoundingBoxes(bbAnnotation, bbGenerated)
            
            #for value_annotation, value_generated in zip(bbAnnotation, bbGenerated):
            #    error += self.dist_lp(value_annotation, value_generated)
                
        return [error, [len(truePositives)+additionalTruePositives, falsePositives, falseNegatives]]
    
    #calculates distance as absolute difference
    @staticmethod
    def dist_lp(p1, p2):
        return abs(p1 - p2)        
        
    #first erode, then dilate a frame given a kernel size and iteration amount in order to remove noise
    @staticmethod
    def dilateErodeFrame(frame, kernelSize, iterations):
        kernel = np.ones((kernelSize, kernelSize),np.uint8)
        return cv2.dilate(cv2.erode(frame,kernel,iterations=iterations), kernel, iterations=iterations)
        
    @staticmethod
    def scale_frame(frame, percentage):
        width = int(frame.shape[1] * percentage/ 100)
        height = int(frame.shape[0] * percentage/ 100)
        dim = (width, height)
        
        return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
    #calculates the error between 2 bounding boxes based on their descriptive attributes (x_min,x_max,y_min,y_max)
    def calculateErrorBetweenBoundingBoxes(self, boundingBox1, boundingBox2):
        error = 0
            
        for value_annotation, value_generated in zip(boundingBox1, boundingBox2):
            error += self.dist_lp(value_annotation, value_generated)
            
        return error
        
    # !!! TODO: This method should implement a model which returns a dictionary of bounding boxes (refer 'to annotationToData()') for all found labels 
    #           or None, if no labels were found for the given frame
    # !!!
    # For implementing a method, create a sub-class of TrackingPipeline and properly implement this method, including the return value
    #
    #returns dictionary with keys=animal names ('mara','mara1','mara2','baer') -> values=boundingbox information [x_min, x_max, y_min, y_max]
    @abstractmethod
    def extractFrame(self, frame):
        pass