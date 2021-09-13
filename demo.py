from random import random
# from kivy.app import App
# from kivy.uix.widget import Widget
# from kivy.uix.button import Button
# from kivy.graphics import Color, Ellipse, Line, Rectangle

from landmark_demo.image_generation import test


# class MyPaintWidget(Widget):

#     def redraw(self, args):
#         self.bg_rect.size = self.size
#         self.bg_rect.pos = self.pos

#     def on_touch_down(self, touch):
#         color = (random(), 1, 1)
#         with self.canvas:
#             Color(*color, mode='hsv')
#             d = 1.
#             Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
#             touch.ud['line'] = Line(points=(touch.x, touch.y))

#     def on_touch_move(self, touch):
#         touch.ud['line'].points += [touch.x, touch.y]


# class MyPaintApp(App):

#     def build(self):
#         parent = Widget()
#         self.painter = MyPaintWidget()
        
#         clearbtn = Button(text='Clear')
#         clearbtn.bind(on_release=self.clear_canvas)
#         parent.add_widget(self.painter)
#         parent.add_widget(clearbtn)
        
#         return parent

#     def clear_canvas(self, obj):
#         self.painter.canvas.clear()
#         with self.painter.canvas:
#             self.painter.bg_rect = Rectangle(source="/home/chern0g0r/workspace/keentools/prototype/Realistic-Neural-Talking-Head-Models/examples/fine_tuning/test_images/test_cranston.jpeg", pos=self.painter.pos, size=self.painter.size)
#         self.painter.bind(pos=self.painter.redraw, size=self.painter.redraw)


if __name__ == '__main__':
    # MyPaintApp().run()
    test()

