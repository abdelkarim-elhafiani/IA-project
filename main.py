
from predictions.camera import initialize_caffe_models
from predictions.camera import read_from_camera


def print_hi(name):

    print(f'Hi, {name}')



if __name__ == '__main__':

    age_net,gender_net=initialize_caffe_models()
    read_from_camera(age_net,gender_net)


