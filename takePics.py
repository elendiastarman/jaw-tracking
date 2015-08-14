import RPi.GPIO as GPIO
import picamera
import time
 
GPIO.setmode(GPIO.BCM)
GPIO.setup(24, GPIO.IN)
#uses pin 24

counter = 0
id = 1 #which Pi is this?
camera = picamera.PiCamera()

camera.resolution = (1920, 1080)
camera.start_preview()
# Camera warm-up time
time.sleep(2)

while True:
    if ( GPIO.input(24) == False ):
        camera.capture('/media/Uploaded_Pictures/cam_' + str(id) + '_' + str(counter) + '.jpg')
        counter = counter + 1