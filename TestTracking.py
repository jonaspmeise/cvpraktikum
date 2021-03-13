from TrackingPipeline import TrackingPipeline

class testPipeline(TrackingPipeline):
    def __init__(self, filePathVideo, folderPathAnnotation):
        super().__init__(filePathVideo, folderPathAnnotation)

    def extractFrame(self, frame):
        baerOnlyDictionary = {
            'baer': [500, 600, 500, 600],
            'mara1': [400, 500, 400, 500]
        }
        
        return baerOnlyDictionary
        
myTestPipeline = testPipeline('D:\cv_praktikum\outside\ch04_20200908120907_analysed_part_18.avi','D:\cv_praktikum\outside\ch04_20200908120907_analysed_part_18')
myTestPipeline.runPipeline()