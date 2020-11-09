import cv2

eye_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 21)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + w), (255, 250, 0), 2)
        img_gray_face = img_gray[y:y+h, x:x+w]
        eyes = eye_cascade_db.detectMultiScale(img_gray_face, 1.1, 19)
        for (ex, ey, ew, eh) in eyes:
            er = ex - ex + int(ew/2.5)
            cv2.circle(img, (ex+x+int(ew/2), ey + y + int(eh/2)), er, (5, 0, 250), 2)
    cv2.imshow('result', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
