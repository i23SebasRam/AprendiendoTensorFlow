#!/usr/bin/env python3


########################################################################################################
################################# Sección de importaciones #############################################
########################################################################################################

import tensorflow
import numpy as np
import cv2
import rospy 
import roslib
from sensor_msgs.msg import Image 


########################################################################################################
#################################### Clase Camera ######################################################
########################################################################################################

class Camera(object):

	def __init__(self):
		self.cv_image = np.zeros((128,128,3)) # cv_image corresponde a la imagen captada por la cámara en simulación.
		

	def main(self):
		rospy.init_node('senecabot_number_detector', anonymous=True)
		rospy.Subscriber("/visionSensorData/image_raw",Image ,self.callback)
		rospy.spin()


	def callback(self,data):
		self.cv_image = np.fromstring(data.data, np.uint8).reshape((128,128,3))
		self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

		
		cv2.imshow("Camera Vision", self.cv_image)
		cv2.waitKey(3)





########################################################################################################
######################################### Main #########################################################
########################################################################################################

if __name__ == '__main__':
	Camera_vision = Camera()
	Camera_vision.main()
