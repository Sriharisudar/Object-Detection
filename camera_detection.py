#from numba import njit

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory, img_to_array, load_img, array_to_img

from sklearn.metrics import classification_report
from adabelief_tf import AdaBeliefOptimizer
import numpy as np
import time
import cv2
import os

DATA_DIR = "D:\\split-garbage-dataset\\train"
CLASSES = os.listdir(DATA_DIR)
#resizing image
size = (300,300)


#loading model

model = keras.models.load_model('garbage_detector.model')

vid = cv2.VideoCapture(0)
print("Camera connection successfully established")
i = 0
while(True):  
    r, frame = vid.read() 
    cv2.imshow('frame', frame)
    cv2.imwrite('D:\old_lap\College\AI_bin\dataset\CV_data_realtime\picked_data'+str(i)+".jpg", frame)
    test_image = load_img('D:\old_lap\College\AI_bin\dataset\CV_data_realtime\picked_data'+str(i)+".jpg", target_size = size)
    test_image = img_to_array(test_image,dtype=np.uint8)
    #test_image = np.expand_dims(test_image, axis = 0)
    test_image = np.array(test_image)/255.0
    result = model.predict(test_image[np.newaxis, ...])
    print(result)
    
    print("Maximum Probability: ",np.max(result[0], axis=-1))  #axis -1 or 0 for row
    predicted_class = CLASSES[np.argmax(result[0], axis=-1)]
    print(predicted_class)
    
    os.remove('D:\old_lap\College\AI_bin\dataset\CV_data_realtime\picked_data'+str(i)+".jpg")
    i = i + 1
    time.sleep(3)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
vid.release() 
cv2.destroyAllWindows()

# show a nicely formatted classification report

print(classification_report(test_dataset.argmax(axis=1), predIdxs,target_names=lb.classes_))

