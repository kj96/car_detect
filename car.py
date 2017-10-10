import cv2
import time

frame = cv2.imread('D:/Online-compete/interview/test_images/test6.jpg')


car_cascade = cv2.CascadeClassifier('cars.xml')


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.1, 4)

x=0

for (x,y,w,h) in cars:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    carf = frame[x:x+w,y:y+h]
    #hist = cv2.calcHist([carf],[0],None,[256],[0,256])
    fil = 'car'+str(x)+'.png'
    #cv2.imwrite(fil,carf)
    x=x+1


cv2.imshow('Cars', frame)

cv2.imwrite('decar.png',frame)
cv2.destroyAllWindows()
