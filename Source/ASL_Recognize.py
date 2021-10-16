import cv2
import numpy as np

import re
import imutils 
import argparse

from collections import deque
from numpy.linalg import norm
from cv2 import ml

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  
    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.6):
        self.model = cv2.ml.SVM_create()

        self.model.setGamma(0.6)
        self.model.setC(1)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):

        self.model.train(samples,cv2.ml.ROW_SAMPLE, responses) # inbuilt training function 

    def save(self,file_name):

        self.model.save(file_name) 
    def Load(self,file_name):

        self.model= cv2.ml.SVM_load(file_name)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:100,:100], bin[100:,:100], bin[:100,100:], bin[100:,100:]
        mag_cells = mag[:100,:100], mag[100:,:100], mag[:100,100:], mag[100:,100:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)



def hog_single(img):
	samples=[]
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	bin_n = 16
	bin = np.int32(bin_n*ang/(2*np.pi))
	bin_cells = bin[:100,:100], bin[100:,:100], bin[:100,100:], bin[100:,100:]
	mag_cells = mag[:100,:100], mag[100:,:100], mag[:100,100:], mag[100:,100:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)

	# transform to Hellinger kernel
	eps = 1e-7
	hist /= hist.sum() + eps
	hist = np.sqrt(hist)
	hist /= norm(hist) + eps

	samples.append(hist)
	return np.float32(samples)
def trainSVM(num):
	imgs=[]
	h=int(input("How many images do you want to train for each letter? "))
	for i in range(65,num+65):

		for j in range(1,h+1):
			print ('Class '+chr(i)+' is being loaded ')
			imgs.append(cv2.imread('TrainData/'+chr(i)+'_'+str(j)+'.jpg',0))  # all images saved in a list
	labels = np.repeat(np.arange(num), h)[:,np.newaxis] # label for each corresponding image saved above
	samples=preprocess_hog(imgs)                # images sent for pre processeing using hog which returns features for the images 
	print('SVM is building wait some time ...')
	print (len(labels))
	print (len(samples))
	model = SVM(C=1, gamma=0.5)
	model.train(samples, labels)  # features trained against the labels using svm
	model.save("1.xml") 
	return model
def Load(fn):
	
	model = SVM(C=1, gamma=0.5)
	
	model.Load(fn) 
	return model

def predict(model,img):
	samples=hog_single(img)
	resp=model.predict(samples)
	return resp

def getMaxContour(contours,minArea=100):
    maxC=None
    maxArea=minArea
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if(area>maxArea):
            maxArea=area
            maxC=cnt
    return maxC

    
#Get Gesture Image by prediction
def getGestureImg(cnt,img,th1,model):
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    imgT=img[y:y+h,x:x+w]
    imgT=cv2.bitwise_and(imgT,imgT,mask=th1[y:y+h,x:x+w])
    imgT=cv2.resize(imgT,(200,200))
    imgTG=cv2.cvtColor(imgT,cv2.COLOR_BGR2GRAY)
    
    resp=predict(model,imgTG)
    # img=cv2.imread('TrainData/'+chr(int(resp[0])+65)+'_2.jpg')
    return chr(int(resp[0])+65)
    #cv2.imshow('imgTG',imgTG)
def onChange(val): #callback when the user change the detection threshold
        value = val
def nothing(x) :
    pass


ans=True
ans1=True
while ans:
    print ("""
----------------------------------------
|    1.Train data                      |
|    2.Create dataset                  |
|    3.Load data                       |
|    4.Set skin value                  |
|    0.Exit                            |
----------------------------------------
    """)
    an=input("What would you like to do? ") 
    if an=="1": 
        model=trainSVM(25)
        ans1=True
        break
    elif an=="2":
        cap=cv2.VideoCapture(0)
        k=input("What Letter do you want to create data (capital letter)? ")
        i=int(ord(list(k)[0])-64)
        j=int(input(" First Serial of the images ? "))
        l=int(input(" Last Serial of the images ? "))
		
        name=""
       
        cv2.namedWindow('trackbar')
        cv2.createTrackbar('Y_min','trackbar',0,255,nothing)
        cv2.createTrackbar('Y_max','trackbar',0,255,nothing)
        cv2.createTrackbar('Cr_min','trackbar',0,255,nothing)
        cv2.createTrackbar('Cr_max','trackbar',0,255,nothing)
        cv2.createTrackbar('Cb_min','trackbar',0,255,nothing)
        cv2.createTrackbar('Cb_max','trackbar',0,255,nothing)
        #cv2.setTrackbarPos('Y_min','trackbar',7)
        #cv2.setTrackbarPos('Y_max','trackbar',161)
        #cv2.setTrackbarPos('Cr_min','trackbar',86)
        #cv2.setTrackbarPos('Cr_max','trackbar',211)
        #cv2.setTrackbarPos('Cb_min','trackbar',61)
        #cv2.setTrackbarPos('Cb_max','trackbar',146)
        cv2.setTrackbarPos('Y_min','trackbar',np.load('Skin.npy')[0])
        cv2.setTrackbarPos('Y_max','trackbar',np.load('Skin.npy')[1])
        cv2.setTrackbarPos('Cr_min','trackbar',np.load('Skin.npy')[2])
        cv2.setTrackbarPos('Cr_max','trackbar',np.load('Skin.npy')[3])
        cv2.setTrackbarPos('Cb_min','trackbar',np.load('Skin.npy')[4])
        cv2.setTrackbarPos('Cb_max','trackbar',np.load('Skin.npy')[5])
        while(cap.isOpened()):
            Y_min = cv2.getTrackbarPos('Y_min','trackbar')
            Y_max = cv2.getTrackbarPos('Y_max','trackbar')
            Cr_min = cv2.getTrackbarPos('Cr_min','trackbar')
            Cr_max = cv2.getTrackbarPos('Cr_max','trackbar')
            Cb_min = cv2.getTrackbarPos('Cb_min','trackbar')
            Cb_max = cv2.getTrackbarPos('Cb_max','trackbar')
          
            _,img=cap.read()
            img=cv2.flip (img,1)
            cv2.putText(img,"Press z to capture ",(50,450), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.rectangle(img,(350,128),(600,400),(255,0,0),3) # bounding box which captures ASL sign to be detected by the system
            img1=img[128:400,350:600]
            img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
            blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)
            skin_ycrcb_min = np.array((Y_min,Cr_min,Cb_min))
            skin_ycrcb_max = np.array((Y_max,Cr_max,Cb_max))

 
            
            mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)
	        #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	        #ret,mask = cv2.threshold(gray.copy(),20,255,cv2.THRESH_BINARY)
            im2,contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, 2) 
            cnt=getMaxContour(contours,4000)
            if np.any(cnt)!=None:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
                imgT=img1[y:y+h,x:x+w]
                imgT=cv2.bitwise_and(imgT,imgT,mask=mask[y:y+h,x:x+w])
                imgT=cv2.resize(imgT,(200,200))
                cv2.imshow('Trainer',imgT)
            cv2.imshow('Frame',img)
            cv2.imshow('Thresh',mask)
            k = 0xFF & cv2.waitKey(10)
            if k == 27:
                break
            if k == 122:
                name=str(chr(i+64))+"_"+str(j)+".jpg"
                cv2.imwrite('TrainData/'+name,imgT)
                if(j<l):
                    j+=1
                else:
                    break
            if k == 113:
                np.save ("Skin1",y)
                
        cap.release()        
        cv2.destroyAllWindows()
        ans1 = False
        
    elif an=="3":
        model=Load('1.xml')
        
        break
    elif an=="4":
        b=int(input(" Y_min: "))
        d=int(input(" Y_max: "))
        e=int(input(" Cr_min: "))
        f=int(input(" Cr_max: "))
        g=int(input(" Cb_min: "))
        h=int(input(" Cb_max: "))
        m = [b,d,e,f,g,h]
        np.save('Skin.npy', m)
        
    elif an=="0":
        ans1 = False
        
        break
    


cap=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
fil_val1=np.array([255,255,255,255,255,255,255], np.uint8)




text= " "
x = []
temp=0
temp1=0
temp2 = 0
temp3=0
tempI = 0
tempJ = 0
switch1 = True
switch2 = False
swtich3= False
switch4=False
switch5 = True
switch6 = False
switch7= False
t=""
a=""
t1=""
a1=""
previouslabel=None
previousText=" "
label = None
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])
pts1 = deque(maxlen=args["buffer"])
counter = 0
counter1=0
(dX, dY) = (0, 0)
direction = ""
direction1 = ""
direction2 =""
predirection=""
prepredirection=""
predirection1=""
prepredirection1=""

cv2.namedWindow('trackbar')
cv2.createTrackbar('Y_min','trackbar',0,255,nothing)
cv2.createTrackbar('Y_max','trackbar',0,255,nothing)
cv2.createTrackbar('Cr_min','trackbar',0,255,nothing)
cv2.createTrackbar('Cr_max','trackbar',0,255,nothing)
cv2.createTrackbar('Cb_min','trackbar',0,255,nothing)
cv2.createTrackbar('Cb_max','trackbar',0,255,nothing)

cv2.setTrackbarPos('Y_min','trackbar',np.load('Skin.npy')[0])
cv2.setTrackbarPos('Y_max','trackbar',np.load('Skin.npy')[1])
cv2.setTrackbarPos('Cr_min','trackbar',np.load('Skin.npy')[2])
cv2.setTrackbarPos('Cr_max','trackbar',np.load('Skin.npy')[3])
cv2.setTrackbarPos('Cb_min','trackbar',np.load('Skin.npy')[4])
cv2.setTrackbarPos('Cb_max','trackbar',np.load('Skin.npy')[5])

#cv2.setTrackbarPos('Y_min','trackbar',7)
#cv2.setTrackbarPos('Y_max','trackbar',161)
#cv2.setTrackbarPos('Cr_min','trackbar',86)
#cv2.setTrackbarPos('Cr_max','trackbar',211)
#cv2.setTrackbarPos('Cb_min','trackbar',61)
#cv2.setTrackbarPos('Cb_max','trackbar',146)

while(cap.isOpened() and ans1==True):
	Y_min = cv2.getTrackbarPos('Y_min','trackbar')
	Y_max = cv2.getTrackbarPos('Y_max','trackbar')
	Cr_min = cv2.getTrackbarPos('Cr_min','trackbar')
	Cr_max = cv2.getTrackbarPos('Cr_max','trackbar')
	Cb_min = cv2.getTrackbarPos('Cb_min','trackbar')
	Cb_max = cv2.getTrackbarPos('Cb_max','trackbar')
	
	
	_, img=cap.read()
	img=cv2.flip (img,1)
	cv2.rectangle(img,(350,128),(600,400),(255,0,0),3) # bounding box which captures ASL sign to be detected by the system
	img1=img[128:400,350:600]
	img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
	if img_ycrcb is not None:
		blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)
	
	skin_ycrcb_min = np.array((Y_min, Cr_min,Cb_min))
	skin_ycrcb_max = np.array((Y_max, Cr_max, Cb_max))
	mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection
	
	contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, 2)
	cnt=getMaxContour(contours,400)						  # using contours to capture the skin filtered image of the hand
	center=None
	
	if np.any(cnt)!=None:
		((x, y), radius) = cv2.minEnclosingCircle(cnt)
		M = cv2.moments(cnt)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
		extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
		extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
		extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
# draw the outline of the object, then draw each of the
# extreme points, where the left-most is red, right-most
# is green, top-most is blue, and bottom-most is teal
		cv2.drawContours(img1, [cnt], -1, (0, 255, 255), 2)
		cv2.circle(img1, extLeft, 8, (0, 0, 255), -1)
		cv2.circle(img1, extRight, 8, (0, 255, 0), -1)
		cv2.circle(img1, extTop, 8, (255, 0, 0), -1)
		cv2.circle(img1, extBot, 8, (255, 255, 0), -1)
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			
			cv2.circle(img1, center, 5, (0, 0, 255), -1)
			pts.appendleft(extTop)
			pts1.appendleft(center)
		label=getGestureImg(cnt,img1,mask,model)   # passing the trained model for prediction and fetching the result
		
		if(label!=None and switch2 == False):
			if(temp==0):
				previouslabel=label
			if previouslabel==label :
				previouslabel=label
				temp+=1
			else :
			   	temp=0
			if(temp==40) and (label != "J") and label != "Z":
				previousText = text
				text= text + label
				
				print (text)
		
		
# 		cv2.imshow('Predict',gesture)
	if label != "J" and label != "Z":
		cv2.putText(img,label,(50,150), font,6,(0,125,155),2)  # displaying the predicted letter on the main screen
	cv2.putText(img,text,(50,450), font,1,(0,0,255),2)
    # loop over the set of tracked points
	for i in np.arange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
 
		# check to see if enough points have been accumulated in
		# the buffer
		if counter >= 10 and i == 10 and pts[i-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
			dX = pts[i-10][0] - pts[i][0]
			dY = pts[i-10][1] - pts[i][1]
			(dirX, dirY) = ("", "")
 
			# ensure there is significant movement in the
			# x-direction
			if np.abs(dX) > 25:
				dirX = "East" if np.sign(dX) == 1 else "West"
 
			# ensure there is significant movement in the
			# y-direction
			if np.abs(dY) > 25:
				dirY = "" if np.sign(dY) == 1 else "North"
 
			# handle when both directions are non-empty
			if dirX != "" and dirY != "":
				direction = "{}-{}".format(dirY, dirX)
 
			# otherwise, only one direction is non-empty
			else:
				direction = dirX if dirX != "" else dirY
			
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			cv2.line(img1, pts[i - 1], pts[i], (0, 0, 255), thickness)
 	# show the movement deltas and the direction of movement on
	
	
	if temp ==15 :
		direction=" "
		t=" "
		a=" "
		prepredirection = " "
		predirection = " "

	if label == "D" and temp >=18:
		
		switch4 = True
	if label == "D" and temp >=18 and temp < 30:
		cv2.putText(img,"Z Ready",(50,300), font,1,(0,125,155),2) 
	if temp >=40 :
		switch4 = False
	if label == "D" and temp >=40 :
		cv2.putText(img,"Z Off",(50,300), font,1,(0,125,155),2) 
	if switch4 == True:
		if a == "East"  and (t == "West") and (direction == "East" or direction == "South-East"):
			label = "Z"
			switch4 = False
		if label == "Z":
			text= text + label
			cv2.putText(img,label,(50,150), font,8,(0,125,155),2)  # displaying the predicted letter on the main screen
			cv2.putText(img,text,(50,450), font,1,(0,0,255),2)
		if direction != "" and direction !=t and direction != "South" :
			prepredirection=t
			predirection=direction
		
		a=prepredirection		
		t=predirection
		
	for j in np.arange(1, len(pts1)):
		# if either of the tracked points are None, ignore
		# them
		if pts1[j - 1] is None or pts1[j] is None:
			continue
 
		# check to see if enough points have been accumulated in
		# the buffer
		if counter1 >= 10 and j == 10 and pts1[j-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
			dX = pts1[j-10][0] - pts1[j][0]
			dY = pts1[j-10][1] - pts1[j][1]
			(dirX1, dirY1) = ("", "")
 
			# ensure there is significant movement in the
			# x-direction
			if np.abs(dX) > 10:
				dirX1 = "" if np.sign(dX) == 1 else ""
 
			# ensure there is significant movement in the
			# y-direction
			if np.abs(dY) > 10:
				dirY1 = "South" if np.sign(dY) == 1 else "North"
 
			# handle when both directions are non-empty
			if dirX1 != "" and dirY1 != "":
				direction2 = "{}-{}".format(dirY1, dirX1)
 
			# otherwise, only one direction is non-empty
			else:
				direction1 = dirX1 if dirX1 != "" else dirY1
			
			thickness = int(np.sqrt(args["buffer"] / float(j + 1)) * 2.5)
			cv2.line(img1, pts1[j - 1], pts1[j], (0, 255, 0), thickness)
	if temp ==15 :
		direction1=" "
		t1=" "
		
	
	if label == "I" and temp >=18:
		switch7 = True
		switch6 = True
	if label == "I" and temp >=18 and temp < 30:
		cv2.putText(img,"J Ready",(50,300), font,1,(0,125,155),2) 
		
	if temp >=40 :
		switch6 = False
		switch7 = False
	if label == "I" and temp >=40 :
		cv2.putText(img,"J Off",(50,300), font,1,(0,125,155),2) 
		switch7 = False
	if switch7 == True:
		if direction1 == "South":
			t1 = direction1
		
	if switch6 == True:
		if t1 == "South" and label == "J":
			label = "J"
			
			if label == "J" and temp == 5:
				text= text + label
				cv2.putText(img,label,(50,150), font,8,(0,125,155),2)  # displaying the predicted letter on the main screen
				cv2.putText(img,text,(50,450), font,1,(0,0,255),2)
				switch6 = False
				switch7 = False
	if np.any(cnt)== None:	
		switch6 = False
		switch7 = False
		
				
		
	
	cv2.putText(img, "dx: {}, dy: {}".format(dX, dY),(10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.35, (0, 0, 255), 1)
	cv2.imshow('Frame',img)
	cv2.imshow('Mask',mask)
	
	counter += 1
	counter1 +=1
	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break
	if k == 32:
	    text= text + " "
	if k == 8:
		text=""
	
	

cap.release()        
cv2.destroyAllWindows()
