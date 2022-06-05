  import cv2 
  import pafy 

  #le lien du vedeo : static 

  url = 'https://www.youtube.com/*'
  vPafy = pafy.new(url)
  play = vPafy.getbest(preftype="mp4")
  cap = cv2.VideoCapture(play.url)

  cap.set(3,480)#la largeur du frame
  cap.set(4,640)#la hauteur du frame

  age_list=['(0,2)','(4,6)','(8,12)','(15,20)','(25,32)',
            '(38,43)','(48,53)','(60,100)',]
  gender_list=['Home','Femme']

  def load_caffe_models():
      age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt','age_net.caffemodel')
      gender_net = cv2.dnn,readNetFromCaffe('deploy_gender.prototxt','gender_net.caffemodel')
      return (age_net,gender_net)

  def video_detector(age_net, gender_net):
      font = cv2.FONT_HERSHEY_SIMPLEX
      while True:
          ret, image = cap.read()
          face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          faces = face_cascade.detectMultiScale(gray, 1.1, 5)
          if(len(faces)>0):
              print("found {} faces".format(str(len(faces))))
          for(x, y, w, h)in faces:
              cv2.rectangle(image, (x,y), (x+w, y+h), (255, 255, 0), 2)
              #Capture Face
              face_img = image[y:y+h, h:h+w].copy()
              blob = cv2.dnn.blobFromImage(face_img, 1,(277, 277), MODEL_MEAN_VALUES, swapRB=False)

              #Predict Gender
              gender_net.setInput(blob)
              gender_preds = gender_net.forward()
              gender = gender_list[gender_preds[0].argmax()]
              print("le genre :" + gender)

              #Predict Afe
              age_net.setInput(blob)
              age_preds = age_net.forward()
              age = age_list[age_preds[0].argmax()]
              print("Tranche d'âge: " + age)
              overlay_text = "%s %s " % (gender, age)
              cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

      cv2.imshow('cadre', image)
      #0xFF is a hecadecimal constant which is 11111111 in binary
      if cv2.waitKey(1) & 0xFF == ord('q')

              break  
    
if __name__ == "__main__":
    age_net, gender_net = load_caffe_model()
    video_detector(age_net, gender_net)


