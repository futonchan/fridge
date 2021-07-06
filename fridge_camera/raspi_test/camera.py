import cv2

# VideoCapture オブジェクトを取得します
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(ret)
cv2.imwrite("test.jpg", frame)

cap.release()
