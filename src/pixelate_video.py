#!/usr/bin/env python3
import sys

import pygame as pg
import numpy as np
import cv2

from utils import read_frames, calculate_keypoints, normalize_rectangle, get_new_rectangle, point_in_rect

SCREEN_SIZE = (1200, 675)


class Main:
    def __init__(self, frames, keypoints, descriptors, auto_update_rectangles=False):
        pg.init()
        self.screen = pg.display.set_mode(SCREEN_SIZE, pg.RESIZABLE)
        self.running = True
        self.update_needed = False

        # video and keypoints
        self.frames = frames
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.current_frame_index = 0

        # control
        self.edit_rectangle = None
        self.move_rectangle_index = -1
        self.rectangles = [[] for _ in range(len(frames))]  # one list of rectangles for every frame
        self.show_keypoints = False
        self.mouse_position = pg.mouse.get_pos()
        self.auto_update_rectangles = auto_update_rectangles

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
                self.next_frame()
                self.update_needed = True
            elif text == 'k':
                self.prev_frame()
                self.update_needed = True
            elif text == 's':
                self.show_keypoints = not self.show_keypoints
                self.update_needed = True
            elif text == 't':
                self.interpolate_rectangle()
                self.update_needed = True
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                self.running = False
            elif event.key == pg.K_LEFT:
                self.prev_frame()
                self.update_needed = True
            elif event.key == pg.K_RIGHT:
                self.next_frame()
                self.update_needed = True
            elif event.key == pg.K_BACKSPACE or event.key == pg.K_DELETE:
                rects = [r for r in self.get_current_rectangles() if not point_in_rect(self.mouse_position, r)]
                self.set_current_rectangles(rects)
                self.update_needed = True
        elif event.type == pg.MOUSEBUTTONDOWN:
            hovered_rect_index = None
            for index, rect in enumerate(self.get_current_rectangles()):
                if point_in_rect(self.mouse_position, rect):
                    hovered_rect_index = index
                    break
            if hovered_rect_index is not None:
                self.move_rectangle_index = hovered_rect_index
            else:
                self.edit_rectangle = np.array([*event.pos, *event.pos])
            self.update_needed = True
        elif event.type == pg.MOUSEMOTION:
            self.mouse_position = event.pos
            if self.edit_rectangle is not None:
                self.edit_rectangle[2:] = event.pos
            elif self.move_rectangle_index != -1:
                c_rect = self.get_current_rectangles()[self.move_rectangle_index]
                if not (pg.key.get_mods() & pg.KMOD_SHIFT):
                    c_rect[0] += event.rel[0]
                    c_rect[1] += event.rel[1]
                c_rect[2] = max(c_rect[2] + event.rel[0], c_rect[0]+1)
                c_rect[3] = max(c_rect[3] + event.rel[1], c_rect[1]+1)
            self.update_needed = True
        elif event.type == pg.MOUSEBUTTONUP:
            self.move_rectangle_index = -1
            if self.edit_rectangle is not None:
                self.edit_rectangle[2:] = event.pos
                if (self.edit_rectangle[[0, 1]] != self.edit_rectangle[[2, 3]]).all():
                    self.get_current_rectangles().append(normalize_rectangle(self.edit_rectangle))
                self.edit_rectangle = None
            self.update_needed = True
        elif event.type == pg.WINDOWRESIZED or event.type == pg.WINDOWENTER or event.type == pg.WINDOWFOCUSGAINED:
            self.update_needed = True

    def next_frame(self):
        old_frame_index = self.current_frame_index
        self.current_frame_index = min(self.current_frame_index + 1, len(self.frames) - 1)

        if self.auto_update_rectangles and old_frame_index != self.current_frame_index:
            new_rects = self.update_rectangles(old_frame_index, self.current_frame_index)
            self.set_current_rectangles(new_rects)

    def prev_frame(self):
        old_frame_index = self.current_frame_index
        self.current_frame_index = max(self.current_frame_index - 1, 0)

        if self.auto_update_rectangles and old_frame_index != self.current_frame_index:
            new_rects = self.update_rectangles(old_frame_index, self.current_frame_index)
            self.set_current_rectangles(new_rects)

    def update_rectangles(self, old_frame_index, new_frame_index):
        if old_frame_index == new_frame_index:
            raise ValueError('old_frame_index == new_frame_index ({})'.format(old_frame_index))

        current_frame = self.frames[new_frame_index]
        y_ratio = pg.display.get_window_size()[1] / current_frame.shape[0]
        x_ratio = pg.display.get_window_size()[0] / current_frame.shape[1]
        ratio = min(x_ratio, y_ratio)

        new_rectangles = []
        for rectangle in self.rectangles[old_frame_index]:
            new_rectangle = self.get_next_rectangle(rectangle, ratio, old_frame_index, new_frame_index)
            if new_rectangle is not None:
                new_rectangles.append(new_rectangle)

        return new_rectangles

    def get_next_rectangle(self, rectangle, ratio, old_frame_index, new_frame_index):
        print('rect:', rectangle)
        print('old fi:', old_frame_index)
        print('new fi:', new_frame_index)
        scaled_rectangle = rectangle / ratio
        new_rectangle = get_new_rectangle(
            self.keypoints[old_frame_index], self.keypoints[new_frame_index],
            self.descriptors[old_frame_index], self.descriptors[new_frame_index],
            scaled_rectangle
        )
        if new_rectangle is not None:
            return (new_rectangle * ratio).round().astype(int)
        return None

    def render(self):
        self.screen.fill((0, 0, 0))
        current_frame = self.frames[self.current_frame_index]

        y_ratio = pg.display.get_window_size()[1] / current_frame.shape[0]
        x_ratio = pg.display.get_window_size()[0] / current_frame.shape[1]
        ratio = min(x_ratio, y_ratio)

        # draw keypoints
        if self.show_keypoints:
            current_frame = cv2.drawKeypoints(
                current_frame, self.keypoints[self.current_frame_index], None,
                (0, 255, 0), 4
            )
        # scale to screen size
        new_dim = (int(current_frame.shape[1] * ratio), int(current_frame.shape[0] * ratio))
        current_frame = cv2.resize(current_frame, new_dim)

        # render rectangle
        for rectangle in self.get_current_rectangles():
            start_pos = (rectangle[0], rectangle[1])
            end_pos = (rectangle[2], rectangle[3])
            color = (255, 128, 0) if point_in_rect(self.mouse_position, rectangle) else (255, 0, 0)
            current_frame = cv2.rectangle(current_frame, start_pos, end_pos, color, 2)
        if self.edit_rectangle is not None:
            start_pos = (self.edit_rectangle[0], self.edit_rectangle[1])
            end_pos = (self.edit_rectangle[2], self.edit_rectangle[3])
            current_frame = cv2.rectangle(current_frame, start_pos, end_pos, (255, 128, 0), 2)

        # transform for pygame
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        current_frame = np.swapaxes(current_frame, 0, 1)
        pygame_frame = pg.surfarray.make_surface(current_frame)
        self.screen.blit(pygame_frame, (0, 0))
        pg.display.update()

    def get_current_rectangles(self):
        return self.rectangles[self.current_frame_index]

    def set_current_rectangles(self, rectangles):
        self.rectangles[self.current_frame_index] = rectangles

    def interpolate_rectangle(self):
        rectangle = None
        for rect in self.get_current_rectangles():
            if point_in_rect(self.mouse_position, rect):
                rectangle = rect
                break
        if rectangle is not None:
            frame_index = self.current_frame_index + 1

            current_frame = self.frames[frame_index]
            y_ratio = pg.display.get_window_size()[1] / current_frame.shape[0]
            x_ratio = pg.display.get_window_size()[0] / current_frame.shape[1]
            ratio = min(x_ratio, y_ratio)

            while frame_index < len(self.frames):
                rectangle = self.get_next_rectangle(rectangle, ratio, frame_index-1, frame_index)
                if rectangle is None:
                    break
                self.rectangles[frame_index].append(rectangle)
                frame_index += 1


def main():
    path = "data/input/test.mp4"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    frames = read_frames(path)
    keypoints, descriptors = calculate_keypoints(frames, algorithm='sift')
    main_instance = Main(frames, keypoints, descriptors)
    main_instance.run()


if __name__ == '__main__':
    main()
