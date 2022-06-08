from predictions.functionalities.intialize_models import init

class Detection_robot:

    def __init__(self):
        self.frame = []
        self.ret = False
        self.gender_net = None
        self.age_net = None

    def start(self):
        self.setup_robot()

    def setup_robot(self):
        (self.age_net,self.gender_net) = init()
        print(self.gender_net)

    def read_from_camera(self):
        pass

    def read_from_image(image):
        pass

    def read_from_youtube(video_url):
        pass