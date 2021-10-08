import random

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window

from kivy.properties import ObjectProperty, ListProperty, StringProperty
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.base import runTouchApp
from kivy.graphics import Rectangle, Line, Color
from kivy.graphics.instructions import Callback
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import AsyncImage
from kivy.uix.textinput import TextInput
from kivy.cache import Cache

import cv2
import numpy as np

from landmark_demo.image_generation import test, get_image_landmarks
from landmark_demo.model_applying import ModelApplyer

Window.size = (1750, 500)

global colors
colors = {
    'l': (0.45, 0, 0.45, 1),  # lips
    'i': (0.8, 0.75, 0.96, 1),  # inner_lips
    'e': (0, 0, 1, 1),  # eye
    'b': (0, 0.6, 0.9, 1),  # brow
    'c': (0, 0.5, 0, 1),  # cheek
    'n': (1, 0, 0, 1),  # nose
    'd': (0, 0, 0, 0)
}


CUR_COLOR = 'l'


class DrawInput(BoxLayout):
    def on_touch_down(self, touch):
        global CUR_COLOR
        print(touch)
        with self.canvas:
            Color(*colors[CUR_COLOR])
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=2)

    def on_touch_move(self, touch):
        print(touch)
        touch.ud["line"].points += (touch.x, touch.y)

    def on_touch_up(self, touch):
        print("RELEASED!", touch)


class RootWidget(BoxLayout):
    pass


class CustomLayout(FloatLayout):

    def __init__(self, source=None, **kwargs):
        # make sure we aren't overriding any important functionality
        super(CustomLayout, self).__init__(**kwargs)

        with self.canvas.before:
            if source is None:
                Color(0, 0, 0, 1)  # green; colors range from 0-1 instead of 0-255
                self.rect = Rectangle(size=self.size, pos=self.pos)
            else:
                self.rect = Rectangle(
                    source=source,
                    size=self.size,
                    pos=self.pos
                )

        self.bind(size=self._update_rect, pos=self._update_rect)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def set_source(self, image):
        self.rect.source = image


class MyBackground(Widget):
    def __init__(self, image, **kwargs):
        super(MyBackground, self).__init__(**kwargs)
        self.update(image)

        self.bind(pos=self.update_bg)
        self.bind(size=self.update_bg)

    def update_bg(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size

    def update(self, image):
        with self.canvas:
            self.bg = Rectangle(source=image, pos=self.pos, size=self.size)
            # self.cb = Callback(self.calb)

    def calb(self, instr):
        print('Update background')


class MainApp(App):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.root = None

        self.path_input = TextInput(text='')
        self.frame_num = TextInput(text='')
        self.background = None
        self.draw_input = None

        self.btn_paths = Button(
            text="Link paths",
            background_color=(0.5, 0.5, 0.5, 1)
        )
        self.btn_transform = Button(
            text="Transform",
            background_color=(0.5, 0.5, 0.5, 1)
        )
        self.btn_load = Button(
            text="Load landmarks",
            background_color=(0.5, 0.5, 0.5, 1)
        )
        self.btn_change = Button(
            text="Save landmarks",
            background_color=(0.5, 0.5, 0.5, 1)
        )

        self.source = AsyncImage(
            source="http://www.everythingzoomer.com/wp-content/uploads/2013/01/Monday-joke-289x277.jpg",
            size_hint=(1, .5),
            pos_hint={'center_x': .5, 'center_y': .5}
        )
        self.result = AsyncImage(
            source="http://www.stuffistumbledupon.com/wp-content/uploads/2012/04/Get-a-Girlfriend-Meme-empty-wallet.jpg",
            size_hint=(1, .5),
            pos_hint={'center_x': .5, 'center_y': .5}
        )

        self.btn_paths.bind(on_press=self.link_paths)
        self.btn_transform.bind(on_press=self.transform)
        self.btn_change.bind(on_press=self.save_landmarks)
        self.btn_load.bind(on_press=self.load_landmarks)

        self.video_path = None
        self.frame_path = None
        self.landmark_path = 'changed.png'

        self.model_weights = 'model_weights.tar'
        self.finetuned_model = None
        self.embedding = 'e_hat_video.tar'

    def build(self):

        root = RootWidget()

        self.background = MyBackground(image='gy.jpg', size=(500, 500))
        root.add_widget(self.background)
        self.draw_input = DrawInput(size=(500, 500))
        self.background.add_widget(self.draw_input)
        print(self.draw_input.size)

        c = CustomLayout()
        root.add_widget(c)
        c.add_widget(self.source)

        c = CustomLayout()
        c.add_widget(self.result)
        root.add_widget(c)

        h_layout = BoxLayout(
            padding=10,
            orientation="vertical"
        )

        h_layout.add_widget(self.path_input)
        h_layout.add_widget(self.btn_paths)
        h_layout.add_widget(self.btn_change)
        h_layout.add_widget(self.frame_num)
        h_layout.add_widget(self.btn_load)
        h_layout.add_widget(self.btn_transform)

        root.add_widget(h_layout)

        self.root = root

        Window.bind(on_key_down=self.key_action)

        return self.root

    def key_action(self, *args):
        global CUR_COLOR
        if args[-2] in colors.keys():
            CUR_COLOR = args[-2]
        print(CUR_COLOR)

    def link_paths(self, value):
        """
        1: frame
        2: model weights
        3: embedding
        4: finetuned weights
        """

        if not self.path_input.text:
            print('Error')
            return

        print('Link paths')

        paths = self.path_input.text.split('\n')
        self.frame_path = paths[0]
        if len(paths) >= 2:
            self.model_weights = paths[1]
        else:
            self.model_weights = 'model_weights.tar'
        if len(paths) >= 3:
            self.embedding = paths[2]
        else:
            self.embedding = 'e_hat_video.tar'
        if len(paths) >= 4:
            self.finetuned_model = paths[3]
        else:
            self.finetuned_model = None

        self.source.source = self.frame_path
        self.set_landmarks(self.frame_path)

    def transform(self, value):
        print('Start transform')

        model = self.model_weights
        if self.finetuned_model is not None:
            model = self.finetuned_model

        landmarks = cv2.imread(self.landmark_path)
        print(self.landmark_path)
        ma = ModelApplyer(
            model,
            self.embedding
        )
        res = ma.apply(landmarks)
        cv2.imwrite('result.jpg', res)
        self.result.source = 'result.jpg'
        self.result.reload()

        print('End transform')

    def save_landmarks(self, value):
        print('save')
        self.draw_input.export_to_png('changed.png', 4)
        self.landmark_path = 'changed.png'

    def load_landmarks(self, value):
        Cache.remove('kv.image')
        Cache.remove('kv.texture')

        print('load')
        self.landmark_path = self.frame_num.text
        self.background.bg.source = self.landmark_path

    def set_landmarks(self, image_path):
        Cache.remove('kv.image')
        Cache.remove('kv.texture')

        img = cv2.imread(image_path)
        print('change_landmarks:' + image_path)
        landmarks, i1 = get_image_landmarks([img])

        tmp_file = 'pure_land.jpg'
        cv2.imwrite(tmp_file, landmarks)
        # self.background.canvas.clear()
        self.background.bg.source = tmp_file
        # with self.background.canvas:
        #     self.background.bg = Rectangle(
        #         source=tmp_file,
        #         pos=self.background.pos,
        #         size=self.background.size)
        print(self.background.bg.source)

    def start_embedding(self, value):
        self.background.bg.source = 'gy.jpg'

# ----------------------


if __name__ == '__main__':
    MainApp().run()
