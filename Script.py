from hmac import trans_36
import cv2  as cv
import Transformations as tf
img_f= cv.imread("logo_fondo.png") 
a= tf.Circle(img_f)
cv.imshow("logo_fondo.png", a)
cv.waitKey(0)