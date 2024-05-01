
import json
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import random

import visualkeras

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam

import numpy as np


DATA_PATH = "./traffic_data/multiple_12000.json"
EPOCHS = 20
INIT_LR = 1e-3
BATCH_SIZE = 32
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
SCHEDULE_HEIGHT = 300

def drawTrafficImage(snapshot, width, height):
    vehicles = snapshot["vehicles"]
    xMax = -10000
    xMin = 10000
    yMax = -10000
    yMin = 10000
    for vehicle in vehicles:
        if(xMax < vehicle["position"]["x"]):
            xMax = vehicle["position"]["x"]
        if(xMin > vehicle["position"]["x"]):
            xMin = vehicle["position"]["x"]
        if(yMax < vehicle["position"]["y"]):
            yMax = vehicle["position"]["y"]
        if(yMin > vehicle["position"]["y"]):
            yMin = vehicle["position"]["y"]
        
    # Initialize image
    image = np.zeros([height, width, 3], dtype=np.uint8)
    
    # Define colors in BGR
    vehicleColor = (200, 200, 0)

    #print(xMax, xMin,yMax,yMin)
    for vehicle in vehicles:
        x = vehicle["position"]["x"]
        x -= xMin 
        x = x/(xMax-xMin) * (width)
        y = -vehicle["position"]["y"]
        y -= yMin
        y = y/(yMax-yMin) * height
        image = cv2.circle(image, (int(x), int(y)), radius=0, color=vehicleColor, thickness=2)
        
    return image

def drawScheduleImage(snapshot, height):
    schedules = snapshot["schedules"]
    scheduleWidth = 0
    scheduleHeight = height
    for i in range(0, len(schedules)):
        schedule = schedules[i]
        scheduleWidthI = len(schedule) * 3
        if(scheduleWidth < scheduleWidthI): 
            scheduleWidth = scheduleWidthI
        
    # Initialize image
    image = np.zeros([scheduleHeight, scheduleWidth, 3], dtype=np.uint8)
    
    # Define colors in BGR
    scheduleColor = (200, 200, 200)
    for i in range(0, len(schedules)):
        schedule = schedules[i]
        y = i * 3
        for j in range(len(schedule)):
            x = j * 3
            if schedule[j] == '1':
                image = cv2.circle(image, (x, y), radius=0, color=scheduleColor, thickness=2)
    return image

def getRequestData(snapshot):
    vehicles = snapshot["vehicles"]
    requestData = np.zeros([100], dtype=np.uint8)
    # [0, 0, 0, ..., 0, 0, 0, ..., 0]

    # vehicle = 
    #    {
    #      "position": {
    #        "x": 362.55957600000056,
    #        "y": 201.8,
    #        "z": 0.5
    #      },
    #      "speed": 13.333333333333336,
    #      "selected": 1,
    #      "incomingLaneId": 42
    #    },


    for vehicle in vehicles:
        if vehicle["selected"] == 1:
            requestData[vehicle["incomingLaneId"]] = 1
    # [0, 0, 1, ..., 0, 1, 1, ..., 0]
    return np.array(requestData)

def loadData(path, width, height):
    trafficData = json.load(open(path))
    snapshots = trafficData["snapshots"]
    xTrafficImages = []
    xScheduleImages = []
    xRequestData = []
    yLabel = []
    yLabel2 = []

    for i in range(0, len(snapshots)):
        #if i >= 100:
        #    break
        #if(i != 800):
            #continue
        snapshot = snapshots[i]
        print("Generating image for snapshot", i, "...")

        # Generate traffic image
        image = drawTrafficImage(snapshot, width, height)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        xTrafficImages.append(image)
        cv2.imwrite("image.jpg", image)

        # Generate schedule image
        scheduleImage = drawScheduleImage(snapshot, SCHEDULE_HEIGHT)
        #scheduleImage = cv2.cvtColor(scheduleImage, cv2.COLOR_BGR2RGB)
        xScheduleImages.append(scheduleImage)
        cv2.imwrite("scheduleImage.jpg", scheduleImage)

        # Generate requesting incoming lane data
        requestData = getRequestData(snapshot)
        xRequestData.append(requestData)

        yLabel.append(snapshot["label"])
        yLabel2.append(snapshot["label2"])

    return (np.array(xTrafficImages), np.array(xScheduleImages), np.array(xRequestData), np.array(yLabel), np.array(yLabel2))

(xTrafficImages, xScheduleImages, xRequestData, yLabel, yLabel2) = loadData(DATA_PATH, IMAGE_WIDTH, IMAGE_HEIGHT)


print("Loading traffic data...")

lb = LabelBinarizer()
# input: [red, blue, green, green]
yLabel = lb.fit_transform(yLabel)
# input: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]

lb2 = LabelBinarizer()
yLabel2 = lb2.fit_transform(yLabel2)

num_classes = len(yLabel[0])
print("Number of labels:", num_classes)

num_classes2 = len(yLabel2[0])
print("Number of labels:", num_classes2)

schedule_width = xScheduleImages[0].shape[1]

SHUFFLE_SEED = 100
random.seed(SHUFFLE_SEED)
random.shuffle(xTrafficImages)

random.seed(SHUFFLE_SEED)
random.shuffle(xScheduleImages)

random.seed(SHUFFLE_SEED)
random.shuffle(xRequestData)

random.seed(SHUFFLE_SEED)
random.shuffle(yLabel)

random.seed(SHUFFLE_SEED)
random.shuffle(yLabel2)

#print(type(xRequestData))
#print(xRequestData[0])
#print(yLabel)

split = train_test_split(xTrafficImages, xScheduleImages, xRequestData, yLabel, yLabel2, test_size = 0.2, random_state=100)

(xTrafficImages_train, xTrafficImages_test, 
xScheduleImages_train, xScheduleImages_test, 
xRequestData_train, xRequestData_test, 
yLabel_train, yLabel_test, 
yLabel2_train, yLabel2_test) = split

# Input 1: traffic image
input_1 = keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
dense1 = layers.Rescaling(1./255)(input_1)

dense1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(dense1)
dense1 = layers.MaxPooling2D()(dense1)
dense1 = layers.Dropout(0.25)(dense1)

dense1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(dense1)
dense1 = layers.MaxPooling2D()(dense1)
dense1 = layers.Dropout(0.25)(dense1)

dense1 = layers.Flatten()(dense1)
dense1 = layers.Dense(32, activation='relu')(dense1)

# Input 2: schedule image
input_2 = keras.Input(shape=(SCHEDULE_HEIGHT, schedule_width, 3))
dense2 = layers.Rescaling(1./255)(input_2)

dense2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(dense2)
dense2 = layers.MaxPooling2D()(dense2)
dense2 = layers.Dropout(0.25)(dense2)

dense2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(dense2)
dense2 = layers.MaxPooling2D()(dense2)
dense2 = layers.Dropout(0.25)(dense2)

dense2 = layers.Flatten()(dense2)
dense2 = layers.Dense(32, activation='relu')(dense2)

# Input 3: request incoming lane ids
input_3 = keras.Input(shape=(100))

dense3 = layers.Flatten()(input_3)
dense3 = layers.Dense(100, activation='relu')(input_3)


concatenated = layers.concatenate([dense1, dense2, dense3], axis=-1)

denseAll = layers.Dense(360, activation='relu')(concatenated)
denseAll = layers.Dense(num_classes)(denseAll)

# Output 1: 
predictions = layers.Activation("softmax", name="category_output")(denseAll)

denseAll2 = layers.Dense(360, activation='relu')(concatenated)
denseAll2 = layers.Dense(num_classes2)(denseAll2)
predictions2 = layers.Activation("softmax", name="category_output2")(denseAll2)


model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[predictions, predictions2])

losses = {
    "category_output": "categorical_crossentropy",
    "category_output2": "categorical_crossentropy"
}
lossWeights = {
    "category_output": 1.0,
    "category_output2": 1.0
}
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
              metrics=['accuracy'])

model.summary()

#visualkeras.graph_view(model).show()
#print(schedule_width)

keras.utils.plot_model(model, "my_network.png", show_shapes=True)

#yLabel2_train = yLabel_train
#yLabel2_test = yLabel_test

history = model.fit(
  x=[xTrafficImages_train, xScheduleImages_train, xRequestData_train],
  y={ "category_output": yLabel_train, "category_output2": yLabel2_train },
  validation_data=([xTrafficImages_test, xScheduleImages_test, xRequestData_test], { "category_output": yLabel_test, "category_output2": yLabel2_test} ),
  epochs=EPOCHS,
  batch_size = 32,
  verbose=1
)
