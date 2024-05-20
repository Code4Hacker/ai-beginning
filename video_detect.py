import cv2 # type: ignore

trained_modal = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    rec, frame = cap.read()
    if rec == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect = trained_modal.detectMultiScale(gray)
        for (x, y, w, h) in face_detect:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)
        cv2.imshow("image Detector", frame)
        
       
        key = cv2.waitKey(1)
        if key==81 or key==113:
            break
    else:
        break