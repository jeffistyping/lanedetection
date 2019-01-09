import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def make_coor(image, params):
	m, b = params
	y1 = image.shape[0]-6
	y2 = int(y1*(0.84))
	x1 = int((y1 - b) / m)
	x2 = int((y2 - b) / m)
	return np.array([x1,y1,x2,y2]) 

def avg_yb(image, lines):
	left = []
	right = []
	for line in lines: 
		x1, y1, x2, y2 = line.reshape(4)
		params = np.polyfit((x1,x2),(y1,y2),1)
		m = params[0]
		b = params[1]
		if m < 0:
			left.append((m,b))
		else:
			right.append((m,b))
	left_avg = np.average(left,axis=0)
	right_avg = np.average(right,axis=0)
	ll = make_coor(image,left_avg)
	rl = make_coor(image,right_avg)
	return np.array([ll,rl])

def region_of_interest(image):
	h = image.shape[0] - 20 #avoiding the car hood lol
	leftcorner = (200,h)
	rightcorner = (1030,h)
	topleft  = (540, 540)
	topright = (660, 540)
	polygons = np.array([[leftcorner,rightcorner,topright,topleft]])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask,polygons,255)
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image

def canny_ed(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 100,150)	
	return canny

def display_lines(image,lines):
	line_img = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			print(line)
			cv2.line(line_img,(x1,y1),(x2,y2),(255,0,0),10)
	return line_img

def show_region(image,region):
	region_img = np.zeros_like(image)
	pass


if __name__ == "__main__":
	# image = cv2.imread('lane_still.jpg')
	# lane_image = np.copy(image)
	# canny = canny(lane_image)
	# crop = region_of_interest(canny)
	# lines = cv2.HoughLinesP(crop, rho=1, theta=np.pi/180, threshold=10, maxLineGap=40, minLineLength=3)
	# average_lines = avg_yb(lane_image, lines)
	# line_img = display_lines(lane_image, average_lines)
	# combined = cv2.addWeighted(lane_image, 0.8, line_img, 1, 1)
	# cv2.imshow("Result", combined)
	# cv2.waitKey(0)

	cap = cv2.VideoCapture("lane.mp4")
	while (cap.isOpened()):
		_, frame = cap.read()
		canny_img = canny_ed(frame)
		crop = region_of_interest(canny_img)
		lines = cv2.HoughLinesP(crop, rho=2, theta=np.pi/180, threshold=10, maxLineGap=40, minLineLength=20)
		# average_lines = avg_yb(frame, lines)
		line_img = display_lines(frame, lines)
		# region_img = show_region(frame, )
		combined = cv2.addWeighted(frame, 0.7, line_img, 1, 1)
		cv2.imshow("Result", combined)
		if cv2.waitKey(1) == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

