#Downloading the dataset
get_ipython().system('pip install kaggle')
get_ipython().system(' mkdir ~/.kaggle')
get_ipython().system(' cp kaggle.json ~/.kaggle/')
get_ipython().system(' chmod 600 ~/.kaggle/kaggle.json')
get_ipython().system(' kaggle datasets download -d jonathanoheix/face-expression-recognition-dataset')


#Unzipping the dataset
get_ipython().system('unzip face-expression-recognition-dataset')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Flatten, Dense
from keras.models import Model
from tensorflow.keras.utils import ImageDataGenerator , img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from keras.applications.mobilenet import MobileNet, preprocess_input 
from keras.losses import categorical_crossentropy


# Training Data
train_datagen = ImageDataGenerator(
     zoom_range = 0.2, 
     shear_range = 0.2, 
     horizontal_flip=True, 
     rescale = 1./255
)

train_data = train_datagen.flow_from_directory(directory= "/content/images/images/train", 
                                               target_size=(224,224), 
                                               batch_size=32,
                                  )


train_data.class_indices


# Testing Data
val_datagen = ImageDataGenerator(rescale = 1./255 )

val_data = val_datagen.flow_from_directory(directory= "/content/images/images/validation", 
                                           target_size=(224,224), 
                                           batch_size=32)
val_data.class_indices


# Visualize the images in the training data
t_img , label = train_data.next()

def plotImages(img_arr, label):
  count = 0
  for im, l in zip(img_arr,label) :
    plt.imshow(im)
    plt.title(im.shape)
    plt.axis = False
    plt.show()
    
    count += 1
    if count == 10:
      break

#function call to plot the images 
plotImages(t_img, label)


#Using MobileNet for its pre-trained weights

base_model = MobileNet( input_shape=(224,224,3), include_top= False )

for layer in base_model.layers:
  layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(units=7 , activation='softmax' )(x)



#Creating the model
model = Model(base_model.input, x)
model.compile(optimizer='adam', loss= categorical_crossentropy , metrics=['accuracy']  )


#Training the model
hist = model.fit_generator(train_data, 
                           steps_per_epoch= 10, 
                           epochs= 100, 
                           validation_data= val_data, 
                           validation_steps= 10)


#Saving the built model
# Save the model in h5 format 
model.save('final_model.h5')


#Train accuracy v/s Testing accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'] , c = "red")
plt.title("training_acc vs validation_acc")
plt.show()


#Training Loss v/s Testing Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'] , c = "red")
plt.title("train_loss vs validation_loss")
plt.show()


#Testing the model using images
op = dict(zip( train_data.class_indices.values(), train_data.class_indices.keys()))



# path for the image to see if it predics correct class
path = "/content/images/images/validation/angry/10052.jpg"
img = load_img(path, target_size=(224,224) )

i = img_to_array(img)/255
input_arr = np.array([i])
input_arr.shape

pred = np.argmax(model.predict(input_arr))

print(f" the image is of {op[pred]}")

#display the image  
plt.imshow(input_arr[0])
plt.title("input image")
plt.show()

