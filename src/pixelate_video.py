#!/usr/bin/env python3
import sys

import pygame as pg
import numpy as np
import cv2

from utils import read_frames, calculate_keypoints


SCREEN_SIZE = (1200, 675)


class Main:
    def __init__(self, frames, keypoints, descriptors):
        pg.init()
        self.screen = pg.display.set_mode(SCREEN_SIZE)
        self.running = True
        self.update_needed = False

        self.frames = frames
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.current_frame_index = 0

    def run(self):
        self.render()
        while self.running:
            events = [pg.event.wait()]
            events = events + pg.event.get()
            self.handle_events(events)
        pg.quit()

    def handle_events(self, events):
        for event in events:
            self.handle_event(event)
        # render
        if self.update_needed:
            self.render()
            self.update_needed = False

    def handle_event(self, event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.TEXTINPUT:
            text = event.text.strip()
            if text == 'j':
                self.current_frame_index = min(self.current_frame_index + 1, len(self.frames)-1)
                self.update_needed = True
            elif text == 'k':
                self.current_frame_index = max(self.current_frame_index - 1, 0)
                self.update_needed = True
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                self.running = False
            elif event.key == pg.K_LEFT:
                self.current_frame_index = max(self.current_frame_index - 1, 0)
                self.update_needed = True
            elif event.key == pg.K_RIGHT:
                self.current_frame_index = min(self.current_frame_index + 1, len(self.frames)-1)
                self.update_needed = True

    def render(self):
        current_frame = self.frames[self.current_frame_index]

        current_frame = cv2.drawKeypoints(
            current_frame, self.keypoints[self.current_frame_index], None,
            (0, 255, 0), 4
        )

        # scale to screen size
        y_ratio = SCREEN_SIZE[1] / current_frame.shape[0]
        x_ratio = SCREEN_SIZE[0] / current_frame.shape[1]
        ratio = min(x_ratio, y_ratio)
        new_dim = (int(current_frame.shape[1] * ratio), int(current_frame.shape[0] * ratio))
        current_frame = cv2.resize(current_frame, new_dim)

        # transform for pygame
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        current_frame = np.swapaxes(current_frame, 0, 1)
        pygame_frame = pg.surfarray.make_surface(current_frame)
        self.screen.blit(pygame_frame, (0, 0))
        pg.display.update()


def main():
    path = "data/input/example3.mp4"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    frames = read_frames(path)
    keypoints, descriptors = calculate_keypoints(frames, algorithm='sift')
    main_instance = Main(frames, keypoints, descriptors)
    main_instance.run()


if __name__ == '__main__':
    main()

