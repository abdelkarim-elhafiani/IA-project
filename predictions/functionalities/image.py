import cv2
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

def read_from_image(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.imread("./images/sample1.jpg")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,5)
    if(len(faces)>0):
        print("Found {} faces".format(str(len(faces))))
    for(x,y,w,h) in faces:
        cv2.rectangle(image, (x,y),(x+w,y+h),(255,255,0),2)
        # Get Face
        face_img = image[y:y+h,h:h+w].copy()
        blob = cv2.dnn.blobFromImage(face_img,1,(227,227),MODEL_MEAN_VALUES,swapRB=False)
        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender: " + gender)
        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age: " + age)
        overlay_text = "%s %s" % (gender, age)
        cv2.putText(image, overlay_text, (x,y-5), font, 0.5,(100,255,50),1, cv2.LINE_AA)
    cv2.imshow("",image)
    cv2.waitKey(0)



