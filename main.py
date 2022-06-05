
from predictions.camera import initialize_caffe_models
from predictions.camera import read_from_camera
from predictions.image import read_from_image


def print_hi(name):

    print(f'Hi, {name}')



if __name__ == '__main__':
    print_hi('PyCharm')
    age_net,gender_net=initialize_caffe_models()
    read_from_image(age_net,gender_net)


