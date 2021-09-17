#!/usr/bin/env python3

import speech_recognition as sr
import pyaudio

init_rec = sr.Recognizer()
info = {}

with sr.Microphone() as source:
    audio_data = init_rec.record(source, duration=5)
    name = init_rec.recognize_google(audio_data)

info.update({'name': name})

print(info)