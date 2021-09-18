#!/usr/bin/env python

import nltk
from nltk.chat.util import Chat, reflections
pairs = [[r".*hi|hello|hey|what's up.*", ["Hello, How are you?"]],
         [r".*(how are you|and you).*", ["Doing well, thank you.","I'm ok, what's new?"]],
         [r"quit", ["Bye, nice talking to you!", "Talk to you later!"]],
         [r".*Address of hospital|location .*", ["ABC dental clinic, Bangalore 540800"]],
         [r".*Dental pain|sour|Blood|problem.*", ["Don't worry Our Experienced doctors will make you happy"]],
         [r".*Appointment|Booking slot.*", ["Available slots are 8am-10am, 11am-12pm, 3pm-5pm, 6pm-8pm, which slot you prefer?"]],
         [r"I want (.*)", ["I have booked the slot for you"]],
         [r".*", ["I am sorry, I don't understand. I am a very simple chatbot!"]]
         ]
chatbot = Chat(pairs, reflections)
chatbot.converse()