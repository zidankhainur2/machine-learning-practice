#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
import pickle
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread("C:/Users/zidan/Downloads/Compressed/sandstone_data_for_ML/data_for_3D_Unet/train_images_256_256_256.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

# Menyimpan pixel gambar asli dalam DataFrame
img2 = img.reshape(-1)
df = pd.DataFrame()
df['Original Image'] = img2


# In[41]:


num = 1  
kernels = []
for theta in range(2):   
    theta = theta / 4. * np.pi
    for sigma in (1, 3):  
        for lamda in np.arange(0, np.pi, np.pi / 4):  
            for gamma in (0.05, 0.5):  
                gabor_label = 'Gabor' + str(num)
                ksize = 9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)

                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  


# In[42]:


# Canny Edge
edges = cv2.Canny(img, 100, 200)
df['Canny Edge'] = edges.reshape(-1)

# Roberts, Sobel, Scharr, Prewitt
edge_roberts = roberts(img)
edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)

df['Roberts'] = edge_roberts.reshape(-1)
df['Sobel'] = edge_sobel.reshape(-1)
df['Scharr'] = edge_scharr.reshape(-1)
df['Prewitt'] = edge_prewitt.reshape(-1)

# Gaussian Filter
gaussian_img = nd.gaussian_filter(img, sigma=3)
gaussian_img2 = nd.gaussian_filter(img, sigma=7)

df['Gaussian s3'] = gaussian_img.reshape(-1)
df['Gaussian s7'] = gaussian_img2.reshape(-1)

# Median Filter
median_img = nd.median_filter(img, size=3)
df['Median s3'] = median_img.reshape(-1)

print(df.head())  # Melihat hasil fitur yang sudah diekstrak

# Menampilkan hasil fitur dalam subplot
features = [
    ("Original Image", img),
    ("Canny Edge", edges),
    ("Roberts", edge_roberts),
    ("Sobel", edge_sobel),
    ("Scharr", edge_scharr),
    ("Prewitt", edge_prewitt),
    ("Gaussian s3", gaussian_img),
    ("Gaussian s7", gaussian_img2),
    ("Median s3", median_img)
]

plt.figure(figsize=(15, 10))

for i, (title, image) in enumerate(features, 1):
    plt.subplot(3, 3, i)  # Buat grid dengan 3 kolom
    plt.imshow(image, cmap="gray")  # Tampilkan dalam grayscale
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()


# In[43]:


labeled_img = cv2.imread('C:/Users/zidan/Downloads/Compressed/sandstone_data_for_ML/data_for_3D_Unet/train_masks_256_256_256.tif')
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
df['Labels'] = labeled_img.reshape(-1)


# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Mendefinisikan label (Y) dan fitur (X)
Y = LabelEncoder().fit_transform(df["Labels"].values)
X = df.drop(labels=["Labels"], axis=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


# In[53]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.svm import LinearSVC

# Melatih model Random Forest
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

# Memprediksi hasil pada data uji
prediction_RF = model.predict(X_test)

# Melatih model SVM
model_SVM = LinearSVC(dual=False, max_iter=100)  
model_SVM.fit(X_train, y_train)

# Memprediksi hasil pada data uji
prediction_SVM = model_SVM.predict(X_test)

# Mengukur akurasi
print("Accuracy using Random Forest = ", metrics.accuracy_score(y_test, prediction_RF))
print("Accuracy using SVM = ", metrics.accuracy_score(y_test, prediction_SVM))


# In[48]:


from yellowbrick.classifier import ROCAUC

print("Classes in the image are: ", np.unique(Y))

#ROC curve for RF
roc_auc=ROCAUC(model, classes=[0, 1, 2, 3])  #Create object
roc_auc.fit(X_train, y_train)
roc_auc.score(X_test, y_test)
roc_auc.show()
  
#ROC curve for SVM
roc_auc=ROCAUC(model_SVM, classes=[0, 1, 2, 3])  #Create object
roc_auc.fit(X_train, y_train)
roc_auc.score(X_test, y_test)
roc_auc.show()


# In[51]:


# Menyimpan model
filename = "sandstone_model"
pickle.dump(model, open(filename, 'wb'))

# Memuat kembali model untuk prediksi
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X)

# Visualisasi hasil segmentasi
segmented = result.reshape((img.shape))

plt.imshow(segmented, cmap='jet')

