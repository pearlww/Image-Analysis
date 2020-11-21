import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import glob


class signFinder(object):

    def __init__(self, path):
        self.path = path
        self.shape = (3024, 4032, 3)

    def GetImageAndLabel(self, landmark):
        label = np.zeros(shape = (self.shape[0],self.shape[1]), dtype='int8')
        nSquares = len(landmark)//4
        for n in range(nSquares):
            idx = n*4
            cv2.fillConvexPoly(label, np.array(landmark[idx:idx+4], 'int32'), n+1)
        return label

    def FindSigns(self, img):
        # extract red color in HSV space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_hsv = np.array([0, 43, 46])
        upper_hsv = np.array([10, 255, 255])
        mask1 = cv2.inRange(img_hsv, lowerb=lower_hsv, upperb=upper_hsv)

        lower_hsv = np.array([170, 43, 46])
        upper_hsv = np.array([180, 255, 255])
        mask2 = cv2.inRange(img_hsv, lowerb=lower_hsv, upperb=upper_hsv)

        mask = mask1+mask2

        # morphological operation
        kernel = np.ones((3,3),np.uint8)
        mask_erode =cv2.erode(mask, kernel, iterations = 5)
        mask_dilate =cv2.dilate(mask_erode, kernel, iterations = 5)

        cnts=cv2.findContours(mask_dilate.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #CHAIN_APPROX_NONE
        cnts = imutils.grab_contours(cnts)

        # filter by area and shape
        signs = []
        blobs = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > 10000:
                blobs.append(cnt)
                       
        # for blob in blobs:
        #     epsilon = 0.1*cv2.arcLength(cnt,True)
        #     approx = cv2.approxPolyD
        # P(blob, epsilon,True)
        #     if len(approx)==4:
        #         signs.append(blob)
        #         #output = cv2.drawContours(img.copy(),[cnt],0,(0,255,0),-1)
        
        myLabel = np.zeros(shape = (self.shape[0],self.shape[1]),dtype='int8')

        for i,s in enumerate(blobs):
            cv2.fillConvexPoly(myLabel, s, i+1)
            
        return myLabel

    def DiceScore(self, output, label):
        AOutput = np.sum(output)
        ALabel = np.sum(label)
        
        I = output & label

        ACom = np.sum(I)
        #resT = 2*abs(output-I) + abs(label - I)
        
        DSC = 2*ACom/(AOutput +ALabel)
        
        return DSC

    def CombinedDiceScore(self, myLabel, label):

        output = myLabel.copy()
        nLabel = np.amax(label)
        nOutput = np.amax(output)
        SumScore = 0
        nScores = 0
        
        # run through all label
        for i in range(1, nLabel+1):
            maxScore = 0
            maxIdx = -1
            labelMask = (label ==i)
            
            # find the output with maximum dice score compare to label
            for j in range(1, nOutput+1):
                outputMask = (output == j)
                score = self.DiceScore(labelMask, outputMask)
                if score>maxScore:
                    maxScore = score
                    maxIdx = j
                    
            # remove the found label that already been used
            if maxIdx>0:
                output[np.where(output==maxIdx)] = 0 # find python functions
                
            SumScore += maxScore
            nScores = nScores +1
        
        CDSC = 0
        if nScores>0:
            CDSC = SumScore/nScores
            
        return CDSC
    
    def TestOneImage(self, nSign):

        img = cv2.imread( self.path + "DTUSigns{0:03d}.jpg".format(nSign))
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])        
        landmark = np.loadtxt( self.path + "DTUSigns{0:03d}.txt".format(nSign))

        label = self.GetImageAndLabel(landmark)
        myLabel = self.FindSigns(img)
        CDSC = self.CombinedDiceScore(myLabel, label)
        
        cv2.imshow('image', img)
        cv2.imshow('ground truth', label)
        cv2.imshow('my ground truth', myLabel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Score:", CDSC)



    def TestValidationSet(self):
        CDSCScores = []

        imgs_path = sorted(glob.glob(self.path+'DTUSigns*.jpg'))
        lms_path = sorted(glob.glob(self.path+'DTUSigns*.txt'))
        
        assert imgs_path
        assert lms_path

        for img_path, lm_path in zip(imgs_path, lms_path):
            img = cv2.imread(img_path)
            b,g,r = cv2.split(img)
            img = cv2.merge([r,g,b])

            landmark = np.loadtxt(lm_path)
            
            label = self.GetImageAndLabel(landmark)
            myLabel = self.FindSigns(img)

            CDSC = self.CombinedDiceScore(myLabel, label)
            print(CDSC)
            CDSCScores.append(CDSC)
        
        print("Validation Score: ", np.mean(CDSCScores))


if __name__ == "__main__":
    path = 'DTUSignPhotos/'

    sf = signFinder(path)
    #sf.TestOneImage()
    sf.TestValidationSet()