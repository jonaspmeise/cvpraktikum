from TrackingPipeline import TrackingPipeline

class testPipeline(TrackingPipeline):
    def __init__(self, filePathVideo, folderPathAnnotation):
        super().__init__(filePathVideo, folderPathAnnotation)

    def extractFrame(self, frame):
        baerOnlyDictionary = {
            'baer': [500, 600, 500, 600]
        }
        
        return baerOnlyDictionary
        
myTestPipeline = testPipeline('D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7.avi','D:\cv_praktikum\inside\ch01_20200909115056_analysed_part_7')
myTestPipeline.runPipeline()