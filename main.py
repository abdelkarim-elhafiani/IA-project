
from predictions.detection_robot import Detection_robot
from predictions.functionalities.init import init


if __name__ == '__main__':

    detection_robot = Detection_robot()

    detection_robot.start()

    detection_robot.read_from_youtube("https://www.youtube.com/watch?v=C6ohRM48oAc&list=RDC6ohRM48oAc&start_radio=1")
    #detection_robot.read_from_image("images/sample1.jpg")
    #detection_robot.read_from_camera()
