import face_recognition
import cv2

picture = face_recognition.load_image_file(r"E:\Ammar work\Photos\me.jpg")
my_face_encoding = face_recognition.face_encodings(picture)[0]

unknown = face_recognition.load_image_file("E:\Ammar work\Photos\collage.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown)[0]

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
faceCascade = cv2.CascadeClassifier("file.xml")
img = cv2.imread(r"E:\Ammar work\Photos\collage.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=10)

font = cv2.FONT_HERSHEY_SIMPLEX
LeftCornerOfText = (0, 20)
fontScale = 0.5
fontColor = (0, 0, 0)
lineType = 2
if results[0] == True:
    cv2.putText(img, 'It is a picture of me... Number of Faces found :'+str(len(faces)), LeftCornerOfText, font, fontScale, fontColor, lineType)
else:
    cv2.putText(img, "It is not my picture", LeftCornerOfText, font, fontScale, fontColor, lineType)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow("Recognizing window", img)
cv2.waitKey(0)
