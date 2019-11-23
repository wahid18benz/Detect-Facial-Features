# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:04:17 2019

@author: tsu
"""

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]
    
def smile(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    L = (A+B+C)/3
    if (L<12):
        return "closed"
    else:
        return "opened"
    
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 

	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

def face_alignement(right_eyebrow,left_eyebrow):
    thresh=15
    if( (right_eyebrow[1][1] < (left_eyebrow[1][1]+thresh )) and (right_eyebrow[1][1] > (left_eyebrow[1][1]-thresh )) ):
        return True
    else:
        return False

def center_face(nose):
    x_thresh=17
    y_thresh=17
    if( ((nose[0][0] > (375-x_thresh) )and( nose[0][0]< (375+x_thresh) )) ):
        return True       

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 3

mouth_COUNTER = 0
mouth_TOTAL = 0

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
eye_case = "ferme"

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(args["shape_predictor"])
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream().start()
fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)



while True:
	
	frame = vs.read()
	frame = imutils.resize(frame,height=750 , width=750)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
    
	faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
		#frameClone = frame.copy()
	if len(faces) > 0:
			faces = sorted(faces, reverse=True,
			key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
			(fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
			roi = gray[fY:fY + fH, fX:fX + fW]
			roi = cv2.resize(roi, (64, 64))
			roi = roi.astype("float") / 255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)
        
			preds = emotion_classifier.predict(roi)[0]
        #emotion_probability = np.max(preds)
			label = EMOTIONS[preds.argmax()] 
			cv2.putText(frame, "Emotion: {}".format(label), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
		     
            
# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		right_eyebrow =shape[reStart:reEnd]
		left_eyebrow =shape[leStart:leEnd]
		mouth= shape[mStart:mEnd]
		nose = shape[nStart:nEnd]
		
		face_case = face_alignement(right_eyebrow,left_eyebrow)        
		center_case= center_face(nose)   
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0    
		mouth_case= smile(mouth)   
    
# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		#cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		#cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
      
		mouthHull = cv2.convexHull(mouth)
		#cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)  
        # check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
       
		if center_case == True:
				cv2.putText(frame, "Visage au centre", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
		  
        
		if ear < EYE_AR_THRESH:
			#COUNTER += 1
			eye_case = "ferme"   
 
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
 
			# reset the eye frame counter
			COUNTER = 0
			eye_case = "ouvert"
            
		cv2.putText(frame, "yeux: {}".format(eye_case), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		####mouth state###
		if mouth_case == "opened" :
			mouth_COUNTER += 1
            #print("hello ")
		else:
			if mouth_COUNTER >= 15:
                #print("world")
				mouth_TOTAL += 1
				#frame = vs.read()
				#time.sleep(0.3)
				#img_name = "opencv_frame_{}.png".format(mouth_TOTAL)
				#cv2.imwrite(img_name, frame)
				#print("{} written!".format(img_name))
				#cv2.destroyWindow("test") 
				mouth_COUNTER = 0
		cv2.putText(frame, "Bouche: {}".format(mouth_case), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
		
        
		if face_case == True:
				cv2.putText(frame, "Face Alignement: {}".format(face_case), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
		
        
        
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()