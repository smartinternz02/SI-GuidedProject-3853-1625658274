import cv2

car_classifier=cv2.CascadeClassifier("cars.xml")
video=cv2.VideoCapture(1)

while True:
    check,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Video',gray)
    
    car=car_classifier.detectMultiScale(gray,1.3,5)

    print(car)
    
    for(x,y,w,h) in car:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (127,0,255), 2)
        cv2.imshow('Face detection', frame)
        picname=datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        cv2.imwrite(picname+".jpg",frame)
    Key=cv2.waitKey(1)
    if Key==ord('q'):
        video.release()
        cv2.destroyAllWindows()
        break

