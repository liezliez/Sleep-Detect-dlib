import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)

try:
    while True:
      for x in range(5):
            GPIO.output(23, True)
            time.sleep(0.5)
            GPIO.output(23, False)


            GPIO.output(24, True)
            time.sleep(0.1)
            GPIO.output(24, False)
            GPIO.output(24, True)
            time.sleep(0.1)
            GPIO.output(24, False)
            GPIO.output(24, True)
            time.sleep(0.1)
            GPIO.output(24, False)

except KeyboardInterrupt:
    GPIO.cleanup()