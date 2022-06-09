
from predictions.camera import initialize_caffe_models
from predictions.camera import read_from_camera
from predictions.image import read_from_image
from predictions.youtube import read_from_youtube


def print_hi(name):

    print(f'Hi, {name}')



if __name__ == '__main__':

    age_net,gender_net=initialize_caffe_models()
    read_from_youtube(age_net,gender_net,
    "https://www.youtube.com/watch?v=2ylVw5xBNEk&list=RD2ylVw5xBNEk&start_radio=1")


