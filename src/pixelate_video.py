#!/usr/bin/env python3


import cv2
import face_recognition
import matplotlib.pyplot as plt


def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def pixelate_image(frame, face_cascade):
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        grayscale_frame,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    print("Found {0} Faces!".format(len(faces)))
    if len(faces):
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        show_image(frame)


def main():
    capture = cv2.VideoCapture("data/input/example1.mp4")
    # writer = cv2.VideoWriter('data/output/output.mp4')

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frame_number = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        pixelate_image(frame, face_cascade)

        frame_number += 1


if __name__ == '__main__':
    main()

