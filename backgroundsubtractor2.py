import numpy as np
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from operator import itemgetter
import os
import re
from detector import Detector

'''
---Ansatz für das Ameisenbär/Mara-Tracking mithilfe Backgroundsubtraction---

Pipeline:
    Frame 
--> Backgroundsubtraction 
--> Noise-Removal 
--> Grouping of adherent movements with Dilation + Contour
--> Bounding Boxes

TODO:
-   how to differ between animals?

'''

class Tracking:
    lastFrame = None
    image_scaling = 100
    
    #parameters for bounding box generation
    #TODO: Heuristic?
    max_bounding_boxes = 1
    filter_strength = 10
    
    #boundingBoxes: #frame -> annotation
    boundingBoxes = {}
    frameCounter = 0
    
    #for tracking marked areas for the ignore mask
    ignoreMask = None
    refPt = []
    
    def __init__(self, filePath, image_scaling, filter_strength = 10, max_bounding_boxes = 1, annotationFolder=None, history = 100):
        #video stream
        self.cap = cv2.VideoCapture(filePath)
        self.image_scaling = image_scaling
        self.max_bounding_boxes = max_bounding_boxes
        self.filter_strength = filter_strength
        
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        #TODO: Set "good" parameters
        #self.fgbg.setBackgroundRatio(0.5)
        self.fgbg.setHistory(history)
        #self.fgbg.setNMixtures(10)
        #self.fgbg.setNoiseSigma(0.1)
        
        self.loadBoundingBoxes(annotationFolder)
        
    #load bounding boxes from annotations    
    def loadBoundingBoxes(self, annotationFolder):
        if annotationFolder is not None:
            for filename in os.listdir(annotationFolder):
                if filename.endswith('.xml'):
                    fullname = os.path.join(annotationFolder, filename)
                    tree = ET.parse(fullname)
                    
                    frameNumber = int(re.findall(r'\d+', tree.find('filename').text)[0])
                    self.boundingBoxes[frameNumber] = tree
                    
                    print('Found annotation for frame #' + str(frameNumber))
    
    #returns the video capture object
    def getCap(self):
        return self.cap

    #fetch next frame
    def fetchFrame(self):
        success, frame = self.cap.read()
        if success:
            self.frameCounter += 1
            rescaledFrame, _, _ = self.rescale_image(frame, self.image_scaling)
            return rescaledFrame
        else:
            print('Error while fetching')
            
    #apply foreground mask model to given image
    def getForegroundMask(self, image):
        return self.fgbg.apply(image)
    
    #returns dictionary with animal names ('mara','mara1','mara2','baer') as keys
    #values in the dictionary are bounding box informations for given annotated frame: x_min, x_max, y_min, y_max
    def annotationToData(self, singleFrameAnnotationXML):
        returnDict = {}
        
        #find all objects inside the XML
        for object in singleFrameAnnotationXML.getroot().findall('object'):
            #get coordinates and scale them according to our current image scaling
            
            x_min = int(int(object[4][0].text)*self.image_scaling)
            y_min = int(int(object[4][1].text)*self.image_scaling)
            x_max = int(int(object[4][2].text)*self.image_scaling)
            y_max = int(int(object[4][3].text)*self.image_scaling)
            
            returnDict[object[0].text] = [x_min, x_max, y_min, y_max]
            
        return returnDict
    
    #draw bounding boxes for animals based on a dictionary into a frame:
    #dictionary: each animal has bounding box values ('x_min', 'x_max', 'y_min', 'y_max') with its name ('mara','mara1','mara2','baer') as the key in the dictionary
    @staticmethod
    def drawBoundingBoxesForAnimals(dictionary, frameToDraw):
        mara = dictionary.get('mara')
        mara1 = dictionary.get('mara1')
        mara2 = dictionary.get('mara2')
        baer = dictionary.get('baer')
        
        if mara is not None:
            cv2.rectangle(frameToDraw,(mara[0],mara[2]),(mara[1],mara[3]),(125,0,0),2)    
        if mara1 is not None:
            cv2.rectangle(frameToDraw,(mara1[0],mara1[2]),(mara1[1],mara1[3]),(255,0,0),2)
        if mara2 is not None:
            cv2.rectangle(frameToDraw,(mara2[0],mara2[2]),(mara2[1],mara2[3]),(255,125,0),2)
        if baer is not None:
            cv2.rectangle(frameToDraw,(baer[0],baer[2]),(baer[1],baer[3]),(255,0,255),2)
            
        return frameToDraw
    
    #the main tracking process
    def startTracking(self):
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.click_and_crop)
    
        while(1):
            #get single frame
            frame = self.fetchFrame()    
            cv2.imshow("Original Frame", frame)
            
            if frame is not None:            
                #update ignore mask with cropped box
                if self.ignoreMask is not None:
                    if len(self.refPt) == 2:
                        #calculate the area to be blackened (ignored)
                        x1 = self.refPt[0][0]
                        x2 = self.refPt[1][0]
                        y1 = self.refPt[0][1]
                        y2 = self.refPt[1][1]
                        self.ignoreMask[min(y1,y2):max(y1,y2),min(x1,x2):max(x1,x2)] = 0
                        #cv2.imshow("ignoremask", self.ignoreMask)
                else:
                    print("X" + str(frame.shape[0]))
                    print("Y" + str(frame.shape[1]))
                    self.ignoreMask = 255 * np.ones((frame.shape[0],frame.shape[1]), np.uint8)              
            
                #normalize frame
                if self.lastFrame is not None:
                    frame = cv2.normalize(frame, self.lastFrame, 0, 255, norm_type=cv2.NORM_MINMAX)
            
                #combat noise with small gauss blur and apply to background subtractor model
                fgMask = self.getForegroundMask(cv2.blur(frame,(5,5)))
                #fgMask = self.getForegroundMask(frame)
                                
                #apply mask to ignore areas where no movement should be recorded
                if self.ignoreMask is not None:
                    fgMask = cv2.bitwise_and(fgMask, self.ignoreMask)
                                
                #if there is an bounding box in our annotation for the same frame:
                annotationForCurrentFrame = self.boundingBoxes.get(self.frameCounter)
                
                if annotationForCurrentFrame is not None:
                    #draw the bounding box from the annotation into the frame
                    frame = self.drawBoundingBoxesForAnimals(self.annotationToData(annotationForCurrentFrame), frame)
                    #cv2.imshow('Last annotated frame',frame)
                
                #calculate our possible bounding boxes found inside our mask
                boundingBoxes = self.getBoundingBox(self.getObjectMoments(fgMask, self.filter_strength, self.max_bounding_boxes), self.max_bounding_boxes, filter=False)
                
                #draw our own found bounding boxes
                for x,y,w,h,_ in boundingBoxes:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    #TODO: Put classification here in order to find out which bounding box shows which animal
                
                cv2.imshow('background removal mask', fgMask)
                #AND the ignoremask into our original frame so it's visible which parts of the images are ignored
                cv2.imshow('frame',cv2.bitwise_and(frame, cv2.cvtColor(self.ignoreMask,cv2.COLOR_GRAY2RGB)))
                
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

            self.lastFrame = frame
        self.cap.release()
        cv2.destroyAllWindows()
                     
    #returns an array of moments for a binary mask (white on black) in order to calculate according bounding boxes for each moment
    @staticmethod
    def getObjectMoments(mask, noise_cancelling_iterations = 5, max_bounding_boxes = 1):
        currentCentroidCount = -1
        #iterations we did: 
        iterations = 0
        currentCentroids = []
        
        #remove initial noise
        editedMask = cv2.erode(mask,np.ones((3,3),np.uint8),iterations = noise_cancelling_iterations)
        
        while(currentCentroidCount < 0 or currentCentroidCount > max_bounding_boxes):
            #remove noise by dilating
            editedMask = cv2.dilate(editedMask,np.ones((3,3),np.uint8),iterations = iterations)
            
            cv2.imshow('centroid mask', editedMask)
           
            currentCentroids, _ = cv2.findContours(editedMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            currentCentroidCount = len(currentCentroids)
            
            iterations += 1
            
        print("Iteration " + str(iterations) + ": " + str(currentCentroidCount) + " bounding boxes")
        #iterations-1 because we count the "erode-step" as a -1 and the "dilate-step" as +1
        return iterations-1, currentCentroids
        
    #rescales an image by a percentage value
    @staticmethod
    def rescale_image(image, scale_value=0.5):
        width = int(image.shape[1] * scale_value)
        height = int(image.shape[0] * scale_value)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation = cv2.INTER_AREA), width, height
        
    #calculates minkowski distance with parameter p for points p1 and p2
    @staticmethod
    def dist_lp(p, p1, p2):
        return ((p1[0]-p2[0])**p + (p1[1]-p2[1])**p)**(1/float(p))

    @staticmethod
    #moments = contours found with cv2 inside a binary mask image
    #max_boxes = maximum amount of boxes that should be found inside a single frame
    #filter = if true, applies max_boxes and only returns the biggest bounding boxes found
    def getBoundingBox(foundMoments, max_boxes, filter=True):
        iterations_needed, moments = foundMoments
    
        boundingBoxes = []
        for moment in moments:
            x,y,w,h = cv2.boundingRect(moment)
            #reduce the size of the bounding boxes because the boxes themselves are calculated on the dilated image,
            #and we want to scale it back to an estimation of the original size
            boundingBoxes.append([x+iterations_needed,y+iterations_needed,w-2*iterations_needed,h-2*iterations_needed,w*h])  
        if filter:
            return sorted(boundingBoxes, key=itemgetter(4), reverse=True)[:max_boxes]
        else:
            return boundingBoxes
            
    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.refPt.append((x, y))           
        
program_indoor = Tracking('D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7.avi', 0.5, 1, 1, None, 100)
program_indoor.startTracking()

program_outdoor = Tracking('D:\cv_praktikum\outside\ch04_20200908120907_analysed_part_18.avi', 0.5, 1, 4, 'D:\cv_praktikum\outside\ch04_20200908120907_analysed_part_18', 1000)
program_outdoor.startTracking()