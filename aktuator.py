import os
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT)

def matikan():
    os.system("irsend SEND_ONCE --count=4 Sony_RM-ED035 KEY_SLEEP")
    GPIO.output(23, False)
    GPIO.output(23, True)

def hidupkan():
    os.system("irsend SEND_ONCE --count=4 Sony_RM-ED035 KEY_SLEEP")
    GPIO.output(23, False)