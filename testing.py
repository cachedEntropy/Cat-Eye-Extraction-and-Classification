import sys, cv2, os
from keras.models import Model, load_model
import numpy as np
import pandas as pd

img_size = 224
labels = {'Blue': 0, 'Cyan': 1, 'Dark Img': 2, 'Green': 3, 'Yellow': 4}
labels = dict((v,k) for k,v in labels.items())
#base_path = '.'
#file_list = sorted(os.listdir(base_path))

#Loading the trained models

#Path to models
bounding_box_model_name = 'bounding_box.h5' 
landmarks_model_name = 'landmarks.h5'
color_model_name = 'color_model_cyan_eyes.h5'
bounding_box_model = load_model(bounding_box_model_name)
landmarks_model = load_model(landmarks_model_name)
color_model = load_model(color_model_name)

labels = {'Blue': 0, 'Cyan': 1, 'Dark/black': 2, 'Green': 3, 'Yellow': 4}
labels = dict((v,k) for k,v in labels.items())


#Utilities

# Function to expand the image with inter_cubic interpolation to keep the colors intact
def image_resize(image, width = None, height = None, inter = cv2.INTER_CUBIC):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

#For resizing the image down to a fixed value
def resize_img(im):
  old_size = im.shape[:2] # old_size is in (height, width) format
  ratio = float(img_size) / max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  # new_size should be in (width, height) format
  im = cv2.resize(im, (new_size[1], new_size[0]))
  delta_w = img_size - new_size[1]
  delta_h = img_size - new_size[0]
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=[0, 0, 0])
  return new_im, ratio, top, left

#Finding the radius of the eye while keeping the perspective in mind
def radius(cords):
  left_eye = cords[0]
  right_eye = cords[1]
  dist = ((left_eye[0]-right_eye[0])**2 + (left_eye[1]-right_eye[1])**2)**0.5
  return(round(dist/4))

#Cropping the Region of Interest 
def roi(img, x, y, r):
  img_roi = img[y-r:y+r,x-r:x+r].copy()
  return(img_roi)


# For selecting the Color of the eye from left and right prediction
def color_selection(img, left_eye, right_eye, r):
  
  #LEFT EYE COLOR PREDICTION
  left_eye_img = roi(img,left_eye[0],left_eye[1],r)
  test = cv2.cvtColor(left_eye_img,cv2.COLOR_BGR2RGB)
  test = resize_img(test)[0]
  inputs = (test.astype('float32') / 255).reshape((1, 224, 224, 3))
  pred_left = sorted([(i,x) for i,x in enumerate(color_model.predict(inputs)[0].tolist())], key=lambda x: x[1], reverse=True)                                     
  
  #RIGHT EYE COLOR PREDICTION
  right_eye_img = roi(img,right_eye[0],right_eye[1],r)
  test = cv2.cvtColor(right_eye_img,cv2.COLOR_BGR2RGB)
  test = resize_img(test)[0]
  inputs = (test.astype('float32') / 255).reshape((1, 224, 224, 3))
  pred_right = sorted([(i,x) for i,x in enumerate(color_model.predict(inputs)[0].tolist())], key=lambda x: x[1], reverse=True) 
  
  left_max, left_max2 = pred_left[0], pred_left[1]
  right_max, right_max2 = pred_right[0], pred_right[1]

  #Printing the colors of both left and right eye.
  print(f'Left:{(labels[left_max[0]],left_max[1])},Right{(labels[right_max[0]],right_max[1])}')
  
  #If right and left eye have the same color prediction
  if(left_max[0]==right_max[0]): 
    #returning the eye with max probility
    if(left_max==max(left_max[1],right_max[1])):return(labels[left_max[0]])
    else: return(labels[right_max[0]])
  #If different colors
    #If any one has GREEN color return GREEN
  elif(left_max[0]==3 or right_max[0]==3): return(labels[3])
    #If any one has BLUE color return BLUE
  elif(left_max[0]==0 or right_max[0]==0): return(labels[0])
    #ELSE Return the one with higher probability 
  elif(left_max[1]>right_max[1]):return(labels[left_max[0]])
  else: return(labels[right_max[0]])
  



img = cv2.imread('IMAGE PATH')
ori_img = img.copy()
result_img = img.copy()

# resizing the image for feeding into the model
img, ratio, top, left = resize_img(img)
inputs = (img.astype('float32') / 255).reshape((1, img_size, img_size, 3))

#prediticting the bounding box for the face of the cat
pred_bb = bounding_box_model.predict(inputs)[0].reshape((-1, 2))

# compute bounding box of original image
ori_bb = ((pred_bb - np.array([left, top])) / ratio).astype(np.int)

# compute lazy bounding box for detecting landmarks
center = np.mean(ori_bb, axis=0)
face_size = max(np.abs(ori_bb[1] - ori_bb[0]))
new_bb = np.array([center - face_size * 0.6,center + face_size * 0.6]).astype(np.int)
new_bb = np.clip(new_bb, 0, 99999)

# predict landmarks
face_img = ori_img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]
face_img, face_ratio, face_top, face_left = resize_img(face_img)

face_inputs = (face_img.astype('float32') / 255).reshape((1, img_size, img_size, 3))

#predicting landmarks on the extract face image
pred_landmarks = landmarks_model.predict(face_inputs)[0].reshape((-1, 2))

# computing landmarks on original image
new_landmarks = ((pred_landmarks - np.array([face_left, face_top])) / face_ratio).astype(np.int)
ori_landmarks = new_landmarks + new_bb[0]


#Caluclating the radius of the eye
r = int(radius(list(ori_landmarks)))

#Caluclating Bounding Boxes for the eyes
left_eye = ori_landmarks[0]
top_left = (left_eye[0]-r,left_eye[1]-r)
bottom_left = (left_eye[0]+r,left_eye[1]+r)

right_eye = ori_landmarks[1]
top_right = (right_eye[0]-r,right_eye[1]-r)
bottom_right = (right_eye[0]+r,right_eye[1]+r)


#Showing the eye images 
x = np.full((100,30,3), 255)
left_eye_img = image_resize(roi(ori_img,left_eye[0],left_eye[1],r),100,100)
right_eye_img = image_resize(roi(ori_img,right_eye[0],right_eye[1],r),100,100)
cv2.imshow(np.hstack((left_eye_img,x,right_eye_img)))

#Predicting the eye color
eye_color = color_selection(ori_img,left_eye,right_eye,r)

# visualizing

#Bounding Box for the cat face
cv2.rectangle(ori_img, pt1=tuple(ori_bb[0]), pt2=tuple(ori_bb[1]), color=(255, 255, 255), thickness=2)

#Bounding Box for the left eye
cv2.rectangle(ori_img, pt1=top_left, pt2=bottom_left, color=(255, 255, 255), thickness=2)

#Bounding Box for the right eye
cv2.rectangle(ori_img, pt1=top_right, pt2=bottom_right, color=(255, 255, 255), thickness=2)

#Text for the Color prediction
cv2.putText(ori_img, str(eye_color), bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)


#Displaying the cat image 
cv2.imshow(np.hstack((ori_img,result_img)))

cv2.waitKey(0)


