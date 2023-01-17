from tensorflow.keras.preprocessing.image import img_to_array,load_img,array_to_img
from PIL import Image
import time
import os

start_time = time.time()

ds_path = "E:\\AI_bin\\dataset\\dataset-original"
img_files = os.listdir(ds_path)
#splitfolders.ratio(ds_path, output="G:\\sem 8\\swedish leaves dataset\\datasest", seed=1337, ratio=(.7, 0.2,0.1))

def create_train_folder(object_type,cls):
    count_objects=1
    train_class_path=os.path.join("E:\\AI_bin\\dataset\\dataset-original",cls)
    for file in os.listdir(train_class_path):

        #join path
        path = os.path.join(train_class_path,file)
        #print(path)
        #main_img = cv2.imread(os.path.join(train_class_path,file))

        #load image and resize (300,300)
        img = load_img(path)
        img = img.resize((300,300),Image.ANTIALIAS)
        img_array = img_to_array(img)
        img_array = img_array * 0.7  #reduce brightness
        array_img = array_to_img(img_array)
        
        if(count_objects<10):
            array_img.save("E:\AI_bin\dataset\pre_processed_data"+"\\"+cls+"\\"+str(cls)+"0"+str(count_objects)+".jpg")
        else:
            array_img.save("E:\AI_bin\dataset\pre_processed_data"+"\\"+cls+"\\"+str(cls)+str(count_objects)+".jpg")
        count_objects+=1

def create_train_dataset():
    object_type=0
    for cls in os.listdir("E:\AI_bin\dataset\dataset-original"):
        #print(cls)
        if cls=='cardboard':
            object_type='cardboard'
            create_train_folder(object_type,cls)
        elif cls=='glass':
            object_type='glass'
            create_train_folder(object_type,cls)
        elif cls=='metal':
            object_type='metal'
            create_train_folder(object_type,cls)
        elif cls=='paper':
            object_type='paper'
            create_train_folder(object_type,cls)
        elif cls=='trash':
            object_type='trash'
            create_train_folder(object_type,cls)
        elif cls=='plastic':
            object_type='plastic'
            create_train_folder(object_type,cls)

create_train_dataset()

print("-------- %s seconds ---------" % (time.time() - start_time))
