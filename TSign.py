from imutils import video
VideoStream = video.VideoStream
FPS = video.FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import tensorflow  as tf
import random, os, time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
#from traffic2 import imgSize

from time import sleep
import board
import digitalio
import adafruit_character_lcd.character_lcd as characterlcd


imgSize = 32

# Modify this if you have a different sized character LCD
lcd_columns = 16
lcd_rows = 2


lcd_rs = digitalio.DigitalInOut(board.D22)
lcd_en = digitalio.DigitalInOut(board.D17)
lcd_d4 = digitalio.DigitalInOut(board.D25)
lcd_d5 = digitalio.DigitalInOut(board.D24)
lcd_d6 = digitalio.DigitalInOut(board.D23)
lcd_d7 = digitalio.DigitalInOut(board.D18)


# Initialise the lcd class
lcd = characterlcd.Character_LCD_Mono(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6,
                                      lcd_d7, lcd_columns, lcd_rows)

# wipe LCD screen before we start
lcd.clear()

sleep(2)

# Load model
new_model = tf.keras.models.load_model('my_model8.0.h5')

labelNames = open("signnames.csv").read().strip().split("\n")[1:]
class_names = [l.split(",")[1] for l in labelNames]


# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame1 = vs.read()
    frame1 = imutils.resize(frame1, width=400)
    frame = cv2.resize(frame1, (imgSize, imgSize))

    test_image = image.img_to_array(frame)
    test_image = np.expand_dims(test_image, axis = 0)
    
    preds = new_model.predict(test_image)

    lst = (preds.tolist())[0]
    lst1 = []
    for l in lst:
        if l > 0.9:
            position = lst.index(l)
            if class_names[position] in class_names:
                lst1.append(class_names[position])
            
    print(lst1)
    lcd.clear()
    dval = ''.join(lst1)
    lcd.message = dval

    # show the output frame
    cv2.imshow("Frame", frame1)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
p()

