import cv2
import numpy as np
import demo

backSub = cv2.createBackgroundSubtractorMOG2()

capture = cv2.VideoCapture('in.avi')

while True:
    _, frame = capture.read()
    if not _:
        break

    fgMask = backSub.apply(frame)
    fgMask = cv2.cvtColor(fgMask, 0)

    kernel = np.ones((5, 5), np.uint8)
    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)
    fgMask = cv2.GaussianBlur(fgMask, (3, 3), 0)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    _, fgMask = cv2.threshold(fgMask, 130, 255, cv2.THRESH_BINARY)

    fgMask = cv2.Canny(fgMask, 20, 200)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        area = cv2.contourArea(contours[i])

        if area > 200:
            cv2.drawContours(fgMask, contours[i], 0, (0, 0, 255), 6)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            object = frame_gray[y:y+h, x:x+w]
            print(object.shape)
            abc = cv2.resize(object, (64, 64)).reshape(1, -1) / 255

            if demo.nn_model.predict(abc):
                cv2.putText(frame, "Human", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(frame, "Non_Human", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Frame', frame)

    keyboard = cv2.waitKey(30)

    if keyboard == 'q' or keyboard == 30:
        break
