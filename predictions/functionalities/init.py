
import cv2

def init():
    age_net =cv2.dnn.readNetFromCaffe(
    'age_deploy.prototxt','age_net.caffemodel')
    gender_net=cv2.dnn.readNetFromCaffe(
    'gender_deploy.prototxt','gender_net.caffemodel')

    return (age_net,gender_net)
