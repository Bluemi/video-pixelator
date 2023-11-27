import matplotlib.pyplot as plt
import numpy as np
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
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)
    return frames, fps


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


def point_in_rect(point, rect):
    return rect[0] < point[0] < rect[2] and rect[1] < point[1] < rect[3]


def filter_keypoints(keypoints, rectangle, descriptors=None):
    filtered = []
    filtered_descriptors = []
    for index, keypoint in enumerate(keypoints):
        if point_in_rect(keypoint.pt, rectangle):
            filtered.append(keypoint)
            if descriptors is not None:
                filtered_descriptors.append(descriptors[index])
    return filtered, filtered_descriptors


def normalize_rectangle(rectangle):
    return np.array([
        min(rectangle[0], rectangle[2]), min(rectangle[1], rectangle[3]),
        max(rectangle[0], rectangle[2]), max(rectangle[1], rectangle[3])
    ])


def get_new_rectangle(old_keypoints, new_keypoints, old_descriptors, new_descriptors, old_rectangle):
    if len(old_keypoints) == 0 or len(new_keypoints) == 0:
        return None

    filtered_keypoints, filtered_descriptors = filter_keypoints(old_keypoints, old_rectangle, old_descriptors)

    bf_matcher = cv2.BFMatcher()
    filtered_descriptors = np.array(filtered_descriptors)
    matches = bf_matcher.knnMatch(filtered_descriptors, new_descriptors, k=2)

    movements = []
    for mn, old_keypoint in zip(matches, filtered_keypoints):
        # sort out bad matches, by comparing to second best match
        m, n = mn
        if m.distance < 0.75 * n.distance:
            new_keypoint = new_keypoints[m.trainIdx]
            movements.append((old_keypoint.pt, new_keypoint.pt))

    if len(movements) == 0:
        return None

    movements = np.array(movements)
    movements = movements[:, 1, :] - movements[:, 0, :]
    avg_movement = np.median(movements, axis=0)

    return old_rectangle + np.tile(avg_movement, 2)


def blur_rectangles(image, rectangles):
    blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
    mask = np.zeros(image.shape[:2], dtype=float)
    for rect in rectangles:
        rect = np.round(rect).astype(int)
        rect = np.maximum(rect, 0)
        mask[rect[1]:rect[3], rect[0]:rect[2]] = 1

    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = mask.reshape((image.shape[0], image.shape[1], 1))

    image = np.round(blurred_image * mask + image * (1.0 - mask))

    return np.minimum(np.maximum(image, 0), 255).astype(np.uint8)


def describe(name, a):
    print('--- {} ---'.format(name))
    print('\tshape: {}'.format(str(a.shape)))
    print('\tdtype: {}'.format(str(a.dtype)))
    print('\tmean: {}'.format(str(np.mean(a))))
    print('\tmin: {}'.format(str(np.min(a))))
    print('\tmax: {}'.format(str(np.max(a))))
