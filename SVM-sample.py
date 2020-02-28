#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import os
import sys
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets,svm,metrics,model_selection
from skimage import feature as ft


path = os.path.abspath(os.path.dirname(sys.argv[0]))+'\image'
categorys = os.listdir(path)
X = []
Y_label= []
for category in categorys:
    images = os.listdir(path+'/'+category)
    for image in images:
        im = ft.hog(Image.open(path+'/'+category+'/'+image).resize((256,256)),
                    orientations=9, 
                    pixels_per_cell=(8,8), 
                    cells_per_block=(8,8), 
                    block_norm = 'L2-Hys', 
                    transform_sqrt = True, 
                    feature_vector=True, 
                    visualise=False
                    )
        X.append(im)
        Y_label.append(category)
                
X = np.array(X)
Y_label = np.array(Y_label)
Y = LabelEncoder().fit_transform(Y_label)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)

classifier = svm.SVC(kernel = 'linear',C =0.01,max_iter =8)
classifier.fit(x_train,y_train)
y_predict = classifier.predict(x_test)

print(classifier.score(x_train,y_train),classifier.score(x_test,y_test))
print("Classification report for classifier %s:\n%s\n"
      %(classifier, metrics.classification_report(y_test, y_predict)))


# In[ ]:




