from TrackingPipeline import TrackingPipeline
import cv2
import numpy as np
import matplotlib.pyplot as plt


class testPipeline(TrackingPipeline):
    def __init__(self, filePathVideo, folderPathAnnotation):
        super().__init__(filePathVideo, folderPathAnnotation)
        self.prev_gray = None
        self.hsv_mask = None
        self.percentage = 20
        self.MAXIMUM_TRACKABLE_OBJECTS = 1

    def scaleBack(self, number):
        return int(100/self.percentage) * number
        
    #foundBoundingBox = [x_min,x_max,y_min,y_max]
    def calculateStability(thresholedFrame, foundBoundingBox):
        
        return None
     
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
        
        #each frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        if self.prev_gray is None:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          
            
        if self.hsv_mask is None:
            self.hsv_mask = np.zeros_like(frame)
            self.hsv_mask[..., 1] = 255

        #calculates optical flow/2D flow vector
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray,
                                           None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)

        #compute magnitude + angle of vector
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        #set image hue according to optical flow direction (angle)
        self.hsv_mask[..., 0] = angle * 180 / np.pi / 2

        #set image value according to optical flow magnitude (normalized)
        self.hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        #convert to rgb
        rgb = cv2.cvtColor(self.hsv_mask, cv2.COLOR_HSV2BGR)

        #grayscale image
        h, s, v1 = cv2.split(rgb)
        rgb = v1
        img = rgb
        
        _, thresh = cv2.threshold(img,3,255,cv2.THRESH_BINARY)
        #thresh = self.dilateErodeFrame(thresh, 10, 1)
        
        # Detect the contours in the image
        currentCentroids, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # max_kernel_size = 15
        # result_array = np.zeros(shape=(max_kernel_size,max_kernel_size))
        
        # for x in range(max_kernel_size):
            # for y in range(max_kernel_size):      
                # morphed_thresh = cv2.erode(thresh, np.ones((x,x),np.uint8), 1)
                # morphed_thresh = cv2.erode(morphed_thresh, np.ones((y,y),np.uint8), 1)
        
                # currentCentroids, _ = cv2.findContours(morphed_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                # amountOfCentroids = len(currentCentroids)
                
                # result_array[x,y] = amountOfCentroids
                
                # if(amountOfCentroids <= self.MAXIMUM_TRACKABLE_OBJECTS and amountOfCentroids > 0):
                    # for contour in currentCentroids:
                        # img_x,img_y,w,h = cv2.boundingRect(contour)
                        # cv2.rectangle(thresh, (img_x,img_y),(img_x+w,img_y+h), (255,255,0), 1)
        
        # print('contours found: ' + str(len(currentCentroids)))
        # print(result_array)

        # Draw all the contours
        img = cv2.drawContours(img, currentCentroids, -1, (0,255,0), 1)

        labelDictionary = {}
        
        cv2.imshow("Farneback Optical Flow", thresh)
        
        maxContourLength = 0
        # Iterate through all the contours
        for contour in currentCentroids:
            if cv2.contourArea(contour) > maxContourLength:
                maxContourLength = cv2.contourArea(contour)
            
                # Find bounding rectangles
                x,y,w,h = cv2.boundingRect(contour)
                # Draw the rectangle
                print('found contour!')
                
                currentBoundingBox = [x, (x+w), y, (y+h)]
                
                #scale it back to the original frame size
                scaledBack_currentBoundingBox = list(map(self.scaleBack, currentBoundingBox))
                
                labelDictionary = {'baer': scaledBack_currentBoundingBox}
                     
        cv2.imshow("image", img)
        
        self.prev_gray = gray
        
        return labelDictionary
        
myTestPipeline = testPipeline('D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7.avi','D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7')
myTestPipeline.runPipeline()