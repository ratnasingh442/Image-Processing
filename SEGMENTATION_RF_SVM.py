#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import pandas as pd
 

import pickle
from matplotlib import pyplot as plt
import os


# In[2]:


def feature_extraction(img):
    df = pd.DataFrame()


#All features generated must match the way features are generated for TRAINING.
#Feature1 is our original image pixels
    img2 = img.reshape(-1)
    df['Original Image'] = img2

#Generate Gabor features
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
#               print(theta, sigma, , lamda, frequency)
                
                    gabor_label = 'Gabor' + str(num)
#                    print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter image and add values to new column
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Modify this to add new column for each gabor
                    num += 1
########################################
#Geerate OTHER FEATURES and add them to the data frame
#Feature 3 is canny edge
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    from skimage.filters import roberts, sobel, scharr, prewitt

#Feature 4 is Roberts edge
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

#Feature 5 is Sobel
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

#Feature 6 is Scharr
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #Feature 7 is Prewitt
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    #Feature 8 is Gaussian with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    #Feature 9 is Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    #Feature 10 is Median with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1

    #Feature 11 is Variance with size=3
#    variance_img = nd.generic_filter(img, np.var, size=3)
#    variance_img1 = variance_img.reshape(-1)
#    df['Variance s3'] = variance_img1  #Add column to original dataframe


    return df


# In[ ]:


image_dataset = pd.DataFrame() 
img_path = "J:/IMAGE+SEGMENTATION/training/dd1/"
for image in os.listdir(img_path):  #iterate through each file 
    print(image)
    img1= cv2.imread(img_path+image)
    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    df = feature_extraction(img)
    image_dataset = image_dataset.append(df)


# In[ ]:


mask_dataset = pd.DataFrame()  #Create dataframe to capture mask info.

mask_path = "J:/IMAGE+SEGMENTATION/training/masks/"    
for mask in os.listdir(mask_path):  #iterate through each file to perform some action
    print(mask)
    
    df2 = pd.DataFrame()  #Temporary dataframe to capture info for each mask in the loop
    input_mask = cv2.imread(mask_path + mask)
    
    #Check if the input mask is RGB or grey and convert to grey if RGB
    if input_mask.ndim == 3 and input_mask.shape[-1] == 3:
        label = cv2.cvtColor(input_mask,cv2.COLOR_BGR2GRAY)
    elif input_mask.ndim == 2:
        label = input_mask
    else:
        raise Exception("The module works only with grayscale and RGB images!")

    #Add pixel values to the data frame
    label_values = label.reshape(-1)
    df2['Label_Value'] = label_values
    df2['Mask_Name'] = mask
    print(df2)
    
    mask_dataset = mask_dataset.append(df2)


# In[ ]:


print(mask_dataset)


# In[ ]:


print(image_dataset)


# In[ ]:


dataset = pd.concat([image_dataset, mask_dataset], axis=1)  


# In[ ]:


dataset = dataset[dataset.Label_Value != 0]


# In[ ]:


print(dataset)


# In[ ]:


X = dataset.drop(labels = ["Mask_Name", "Label_Value"], axis=1) 

#Assign label values to Y (our prediction)
Y = dataset["Label_Value"].values 

#Encode Y values to 0, 1, 2, 3, .... (NOt necessary but makes it easy to use other tools like ROC plots)
from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(Y)


##Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
## Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 50, random_state = 42)

## Train the model on training data
model.fit(X_train, y_train)

prediction_RF = model.predict(X_test)


#######################################################
# STEP 5: Accuracy check
#########################################################

from sklearn import metrics
#Print the prediction accuracy
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy using Random Forest= ", metrics.accuracy_score(y_test, prediction_RF))


# In[ ]:


from yellowbrick.classifier import ROCAUC

print("Classes in the image are: ", np.unique(Y))

#ROC curve for RF
roc_auc=ROCAUC(model, classes=[0,1,2])  #Create object
roc_auc.fit(X_train, y_train)
roc_auc.score(X_test, y_test)
roc_auc.show()


# In[ ]:


model_name = "fabric_segmentation"
pickle.dump(model, open(model_name, 'wb'))


# In[ ]:


features_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
print(feature_imp)


# In[20]:


import pickle
#from matplotlib import pyplot as plt
from skimage import io


filename = "fabric_segmentation"
loaded_model = pickle.load(open(filename, 'rb'))

path = "J:/IMAGE+SEGMENTATION/test/"
import os
for image in os.listdir(path):  #iterate through each file to perform some action
    print(image)
    img1= cv2.imread(path+image)
    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
    #Call the feature extraction function.
    X = feature_extraction(img)
    result = loaded_model.predict(X)
    segmented = result.reshape((img.shape))
    segmented = segmented.astype(np.int8)
    io.imsave('J:/IMAGE+SEGMENTATION/segment/'+ image, segmented)




# In[ ]:




