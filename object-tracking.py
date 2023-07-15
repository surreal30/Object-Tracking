import cv2
import mediapipe as mp 
import numpy as np

# Getting modules for hands and poses from mediapipe.
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

vid = cv2.VideoCapture(0)
vid.set(3, 1800)
vid.set(4, 600)

points = []

while True:
	ret, frame = vid.read()
	img = cv2.flip(frame, 1)
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = pose.process(imgRGB)
	if results.pose_landmarks:
		mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
		
		# Get the landmark ID and show all the points on the body
		for pId, lm in enumerate(results.pose_landmarks.landmark):
			h, w, c = img.shape
			cx, cy = int(lm.x*w), int(lm.y*h) #Converts the points given by landmark into coordinates

			if pId == 16 or pId == 15: 
				cv2.circle(img, (cx, cy), 15, (0, 0, 0), cv2.FILLED) 
				points.append([cx, cy])
				for point in points:
					cv2.circle(img, (point[0], point[1]), 8, (127, 255, 0), cv2.FILLED)
			if pId == 25 or pId == 26:
				cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED) 

	cv2.imshow('Object Tracker', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

vid.release()
cv2.destroyAllWindows()