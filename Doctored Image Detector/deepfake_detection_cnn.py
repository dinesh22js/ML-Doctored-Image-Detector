import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import  models
import matplotlib.pylab as plt
import matplotlib as mpl
import cv2
import numpy as np
import sys

import contextlib
import io


deepfakeDetectionModel = models.load_model("trained_models/cnn_trained.keras")

input_image = cv2.imread(sys.argv[1])
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image,(256,256))
#cropping
input_image = input_image/255

with contextlib.redirect_stdout(io.StringIO()):
    result = deepfakeDetectionModel.predict(np.array([input_image]))

output = ['doctored','not doctored']
result_class = np.argmax(result[0]) 
print("The image is",output[result_class],"\nConfidence:",int(np.max(result[0])*100),"%")


mpl.rcParams['toolbar'] = 'None'
plt.figure(num="Input Image")
plt.imshow(input_image)
plt.axis('off')
plt.show()

