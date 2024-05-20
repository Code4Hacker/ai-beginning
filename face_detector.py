import cv2 # type: ignore

trained_modal = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread("test2.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_detect = trained_modal.detectMultiScale(gray)

print(face_detect)
for (x, y, w, h) in face_detect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (25, 232,186), 5)

cv2.imshow("image Detector", img)
key = cv2.waitKey()

if key==81 or key==113:
    cv2.destoryAllWindows()
