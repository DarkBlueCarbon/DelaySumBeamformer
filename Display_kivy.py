from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.lang import Builder
from kivy.uix.checkbox import CheckBox
from kivy.uix.progressbar import ProgressBar
from kivy.config import Config
import audio
import pygame as pg
import scipy
from scipy import signal
from scipy.signal import butter, convolve, lfilter
from scipy.io import wavfile

import time
import threading
import wave
from kivy.core.window import Window as AppWindow
from pydub import AudioSegment

Config.set('graphics', 'fullscreen', 'fake')

Config.set('graphics', 'height', '480')
Config.set('graphics', 'width', '800')
Config.write()

dB = 10

label_slider_dB = Label(text='10 dB' ,size_hint=(0.85,0.10))

label_slider = Label(text='5 sec' ,size_hint=(0.85,0.10))

label_status = Label(text='Standby',size_hint=(1,0.15))

slider_bar_dB=Slider(min =0, max=15, value=10,step =1,size_hint=(1,0.10))

slider_bar=Slider(min =1, max=30, value=5,step =1,size_hint=(0.85,0.10))

checkbox = CheckBox(text='Hello 2',size_hint=(0.15,0.2))

audio.start_Bluetooth()

class TestApp(App):

    
    def btn_record_event(instance):

        label_status.text = 'Recording for ' +label_slider.text
        audio.record_4_mic(int(slider_bar.value))
        
        label_status.text = 'Recorded for ' +label_slider.text
        
        play_filter = 1
        
        if (play_filter == 1):
            # filter it
            
            audio.bessel_bandpass_filter("./file1.wav", "file1_filtered.wav")
            audio.bessel_bandpass_filter("./file2.wav", "file2_filtered.wav")
            audio.bessel_bandpass_filter("./file3.wav", "file3_filtered.wav")
            audio.bessel_bandpass_filter("./file4.wav", "file4_filtered.wav")

            file1 = AudioSegment.from_file("./file1_filtered.wav")
            file2 = AudioSegment.from_file("./file2_filtered.wav")
            file3 = AudioSegment.from_file("./file3_filtered.wav")
            file4 = AudioSegment.from_file("./file4_filtered.wav")
        

            
            
        else:
            # Play unfiltered

            file1 = AudioSegment.from_file("./file1.wav")
            file2 = AudioSegment.from_file("./file2.wav") 
            file3 = AudioSegment.from_file("./file3.wav") 
            file4 = AudioSegment.from_file("./file4.wav") 
            

       
        # combine files
        combined = file1.overlay(file2)
        combined = file3.overlay(combined)
        combined = (file4.overlay(combined))
        combined = combined  + dB
        combined.export("output.wav", format='wav')
        
        
        
        if checkbox.active:

            pg.mixer.init()
            pg.init()
           
           
            pg.mixer.set_num_channels(50) 
            pg.mixer.music.load("output.wav")
            pg.mixer.music.play()
            
            
            label_status.text = 'Playing...'
        
    def btn_play_event(instance):
        pg.mixer.init()
        pg.init()
       
        pg.mixer.set_num_channels(50) 
        pg.mixer.music.load("output.wav")
        pg.mixer.music.play()
        
        
        label_status.text = 'Playing...'
    
    def OnSliderValueChange(instance, value):
        label_slider.text = str(value) + ' Sec'

    def OnSliderValueChange_dB(instance, value):
        dB = int(value)
        label_slider_dB.text = str(dB) + ' dB'
        
        
    def build(self):

    
        layout = StackLayout()

        slider_bar.bind(value = TestApp.OnSliderValueChange)

        slider_bar_dB.bind(value = TestApp.OnSliderValueChange_dB)
        
        btn_record = Button(background_color=(155/255,28/255,13/255,1), text='Record Audio',size_hint=(0.5,0.3))
        btn_record.bind(on_press = TestApp.btn_record_event)
        
        btn_play = Button(background_color=(15/255,69/255,130/255,1), text='Play Audio',size_hint=(0.5,0.3))
        btn_play.bind(on_press = TestApp.btn_play_event)


        
        
        layout.add_widget(label_slider)
        layout.add_widget(Label(text='Autoplay',size_hint=(0.15,0.15)))
        layout.add_widget(slider_bar)
        layout.add_widget(checkbox)
        layout.add_widget(label_slider_dB)
        layout.add_widget(slider_bar_dB)
        layout.add_widget( btn_record )
        layout.add_widget(btn_play)
        layout.add_widget(label_status)


        return layout
  
TestApp().run()

