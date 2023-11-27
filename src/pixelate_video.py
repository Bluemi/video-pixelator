#!/usr/bin/env python3
import sys

import pygame as pg
import numpy as np
import cv2
from tqdm import tqdm

from utils import read_frames, calculate_keypoints, normalize_rectangle, get_new_rectangle, point_in_rect, \
    blur_rectangles, describe

SCREEN_SIZE = (1200, 675)


class Main:
    def __init__(self, frames, keypoints, descriptors, fps, auto_update_rectangles=False):
        pg.init()
        self.screen = pg.display.set_mode(SCREEN_SIZE, pg.RESIZABLE)
        self.running = True
        self.update_needed = False

        # video and keypoints
        self.frames = frames
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.current_frame_index = 0
        self.fps = fps

        # control
        self.edit_rectangle = None
        self.move_rectangle_index = -1
        self.rectangles = [[] for _ in range(len(frames))]  # one list of rectangles for every frame
        self.show_keypoints = False
        self.show_rects = True
        self.show_blur = False
        self.mouse_position = self.get_mouse_in_image_coordinates()
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
            elif text == 'p':
                self.show_keypoints = not self.show_keypoints
                self.update_needed = True
            elif text == 'b':
                self.show_blur = not self.show_blur
                self.update_needed = True
            elif text == 'r':
                self.show_rects = not self.show_rects
                self.update_needed = True
            elif text == 't':
                self.interpolate_rectangle()
                self.update_needed = True
            elif text == 'e':
                self.export()
                self.update_needed = True
            elif event.text == 'd':
                rects = [r for r in self.get_current_rectangles() if not point_in_rect(self.mouse_position, r)]
                self.set_current_rectangles(rects)
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
        elif event.type == pg.MOUSEBUTTONDOWN:
            hovered_rect_index = None
            for index, rect in enumerate(self.get_current_rectangles()):
                if point_in_rect(self.mouse_position, rect):
                    hovered_rect_index = index
                    break
            if hovered_rect_index is not None:
                self.move_rectangle_index = hovered_rect_index
            else:
                self.edit_rectangle = np.array([*self.mouse_position, *self.mouse_position])
            self.update_needed = True
        elif event.type == pg.MOUSEMOTION:
            self.mouse_position = self.get_mouse_in_image_coordinates()
            if self.edit_rectangle is not None:
                self.edit_rectangle[2:] = self.mouse_position
            elif self.move_rectangle_index != -1:
                c_rect = self.get_current_rectangles()[self.move_rectangle_index]
                if not (pg.key.get_mods() & pg.KMOD_SHIFT):
                    ratio = self.get_ratio()
                    c_rect[0] += int(round(event.rel[0] * ratio))
                    c_rect[1] += int(round(event.rel[1] * ratio))
                c_rect[2] = max(c_rect[2] + int(round(event.rel[0])), c_rect[0]+1)
                c_rect[3] = max(c_rect[3] + int(round(event.rel[1])), c_rect[1]+1)
            self.update_needed = True
        elif event.type == pg.MOUSEBUTTONUP:
            self.move_rectangle_index = -1
            if self.edit_rectangle is not None:
                self.edit_rectangle[2:] = self.mouse_position
                if (self.edit_rectangle[[0, 1]] != self.edit_rectangle[[2, 3]]).all():
                    new_rect = normalize_rectangle(self.edit_rectangle)
                    self.get_current_rectangles().append(new_rect)
                self.edit_rectangle = None
            self.update_needed = True
        elif event.type == pg.WINDOWRESIZED or event.type == pg.WINDOWENTER or event.type == pg.WINDOWFOCUSGAINED:
            self.update_needed = True

    def get_ratio(self):
        frame_size = self.frames[0].shape
        y_ratio = pg.display.get_window_size()[1] / frame_size[0]
        x_ratio = pg.display.get_window_size()[0] / frame_size[1]
        return min(x_ratio, y_ratio)

    def get_mouse_in_image_coordinates(self):
        return np.round(np.array(pg.mouse.get_pos()) / self.get_ratio()).astype(int)

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

        new_rectangles = []
        for rectangle in self.rectangles[old_frame_index]:
            new_rectangle = self.get_next_rectangle(rectangle, old_frame_index, new_frame_index)
            if new_rectangle is not None:
                new_rectangles.append(new_rectangle)

        return new_rectangles

    def get_next_rectangle(self, rectangle, old_frame_index, new_frame_index):
        scaled_rectangle = rectangle
        new_rectangle = get_new_rectangle(
            self.keypoints[old_frame_index], self.keypoints[new_frame_index],
            self.descriptors[old_frame_index], self.descriptors[new_frame_index],
            scaled_rectangle
        )
        if new_rectangle is not None:
            return new_rectangle.round().astype(int)
        return None

    def render(self):
        self.screen.fill((0, 0, 0))
        current_frame = self.frames[self.current_frame_index]

        ratio = self.get_ratio()

        # blur rectangles
        if self.show_blur:
            current_frame = blur_rectangles(current_frame, self.get_current_rectangles())

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
        if self.show_rects:
            for rectangle_image in self.get_current_rectangles():
                mouse_touch = point_in_rect(self.mouse_position, rectangle_image)
                rectangle_screen = np.round(rectangle_image * ratio).astype(int)
                start_pos = (rectangle_screen[0], rectangle_screen[1])
                end_pos = (rectangle_screen[2], rectangle_screen[3])
                color = (255, 128, 0) if mouse_touch else (255, 0, 0)
                current_frame = cv2.rectangle(current_frame, start_pos, end_pos, color, 2)
        # render edit rectangle
        if self.edit_rectangle is not None:
            start_pos = (int(round(self.edit_rectangle[0] * ratio)), int(round(self.edit_rectangle[1] * ratio)))
            end_pos = (int(round(self.edit_rectangle[2] * ratio)), int(round(self.edit_rectangle[3] * ratio)))
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

            next_rectangle = rectangle
            while frame_index < len(self.frames):
                next_rectangle = self.get_next_rectangle(next_rectangle, frame_index-1, frame_index)
                if next_rectangle is None:
                    break
                self.rectangles[frame_index].append(next_rectangle)
                frame_index += 1

            frame_index = self.current_frame_index - 1
            next_rectangle = rectangle
            while frame_index >= 0:
                next_rectangle = self.get_next_rectangle(next_rectangle, frame_index+1, frame_index)
                if next_rectangle is None:
                    break
                self.rectangles[frame_index].append(next_rectangle)
                frame_index -= 1

    def export(self):
        frame_size = (self.frames[0].shape[1], self.frames[0].shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter('data/output.avi', fourcc, self.fps, frame_size)

        for frame, rect in tqdm(zip(self.frames, self.rectangles), desc='exporting video', total=len(self.frames)):
            frame = blur_rectangles(frame, rect)
            writer.write(frame)

        writer.release()
        print('exporting video done')


def main():
    path = "data/input/test.mp4"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    frames, fps = read_frames(path)
    keypoints, descriptors = calculate_keypoints(frames, algorithm='sift')
    main_instance = Main(frames, keypoints, descriptors, fps)
    main_instance.run()


if __name__ == '__main__':
    main()
