import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def pixelate_image(frame, cascades):
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = []
    for cascade in cascades:
        faces.extend(cascade.detectMultiScale(grayscale_frame, 1.1, 1, maxSize=(35, 35), minSize=(15, 15)))

    print(faces)
    if faces:
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (2550, 0, 0), 2)
        show_image(frame)


def read_frames(path):
    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)
    return frames


def calculate_keypoints(frames, algorithm='sift'):
    if algorithm == 'sift':
        algo = cv2.SIFT_create()
    elif algorithm == 'surf':
        algo = cv2.xfeatures2d.SURF_create()
    elif algorithm == 'orb':
        algo = cv2.ORB_create()
    else:
        raise ValueError('Unknown Algorithm {}'.format(algorithm))
    keypoints = []
    descriptors = []
    for frame in tqdm(frames, desc="calculating keypoints"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoint, descriptor = algo.detectAndCompute(gray, None)
        keypoints.append(keypoint)
        descriptors.append(descriptor)
    return keypoints, descriptors


def cascade_try(capture):
    cascades = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cascades.append(face_cascade)
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    cascades.append(profile_cascade)

    frame_number = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        pixelate_image(frame, cascades)

        frame_number += 1
