import cv2
import face_recognition
import numpy as np
import os
import pyttsx3

path = "image_faces"
images = [] # -----> array for each image
classNames = [] # -----> name of eacg image
myList = os.listdir(path)

for cl in myList:
    person_image = cv2.imread(f'{path}/{cl}')
    images.append(person_image)     # accessing to main image
    classNames.append(os.path.splitext(cl) [0])  #the text written on each image

def find_encoding(images):

    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list

encode_list_known = find_encoding(images)
print("encoding complete")

#-----------------------------------------------------------------------------

cap = cv2.VideoCapture(0)

history = list(range(4))

while True:
    success, img = cap.read()

    img = cv2.resize(img, (0, 0), None, 0.37, 0.37)

    face_loc_frame = face_recognition.face_locations(img)

    encode_frame = face_recognition.face_encodings(img, face_loc_frame)

    for encodeFace, faceLoc in zip(encode_frame, face_loc_frame):

        matches = face_recognition.compare_faces(encode_list_known, encodeFace) #[t ,f, f, f]
        faceDis = face_recognition.face_distance(encode_list_known, encodeFace)#[0.1, 0,7, 0,8 ,0.9]

        matchIndex = np.argmin(faceDis)
        name = ""

#-------------------------------------------------------------------------------------
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc  # (top ,right, bottom, left)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-10), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name, (x1+6, y2-6), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 2)

        history.append(name)

        cv2.imshow("video", img)
        cv2.waitKey(1)

#----------------------------------------------------------------------------------------------

        if (history[-1] != history[-2]) and (history[-1] != history[-3] and history[-2] != history[-4]):
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            voice = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
            engine.setProperty("voice", voice)
            if name != "":
                engine.say(f'hello {name}')
            engine.runAndWait()

        if len(history) == 10 :
            history = history[5:]
