# -*- coding: utf-8 -*-
# loading the Face_embeddings_dict pickle file
from Extract import face_size,detector,models,Extract
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

with open("data/face_embeddings_dict.pkl", "rb") as data:
    face_embeddings_dict = pickle.load(data)
data.close()

facenet_embeddings = np.asarray(face_embeddings_dict['facenet'])

embedds = facenet_embeddings
e_mean = np.mean(embedds,axis = 0)
e_Cov = np.cov(embedds.T)

e_eigval,e_eigvec = np.linalg.eig(e_Cov)
e_sort_eigval = np.sort(e_eigval)

sort_ind = np.argsort(e_eigval) ## return indices for eigenvalues which are in ascending order
sort_ind = sort_ind[::-1] ## To invert the sort_ind to descending order

e_eigsum = sum(e_eigval)
#print(eig_val_sum)
temp_sum = 0
principal_eig_vec = []
principal_eig_val = []
i=0
while(temp_sum<0.95*e_eigsum):
    principal_eig_vec.append(e_eigvec[sort_ind[i]])
    principal_eig_val.append(e_eigval[sort_ind[i]])
    temp_sum += e_eigval[sort_ind[i]]
    i += 1
print("Number of components for facenet is "+str(i))

fig, ax = plt.subplots()

ax.plot(principal_eig_val)
ax.plot(e_sort_eigval[::-1],'k--')
ax.set_title("Eigen-values corresponding to the principal components of facenet")

plt.tight_layout()

Q_hat = np.matrix(principal_eig_vec)
e_trans_embed = []
for i in range(embedds.shape[0]):
    vec = embedds[i,:]
    vec = vec-e_mean.T
    trans_embed = np.linalg.pinv(Q_hat).T@vec
    e_trans_embed.append(np.ravel(trans_embed))
face_trans_embed = np.array(e_trans_embed)


# Make Ball Tree 
from sklearn.neighbors import BallTree

facenet_tree =  BallTree(face_trans_embed, leaf_size = 40)

with open('data/face_dict.json') as json_file:
    f_data = json.load(json_file)
json_file.close()

with open('data/face_identity.json') as json_file:
    f_identity = json.load(json_file)
json_file.close()

def predict(directory,data_directory):
    
    extract_info = Extract(directory, face_size, detector, models, predict=1)
    face_embed_models = extract_info.get_face_data(directory, face_size, models)
    test_embed = np.asarray(face_embed_models['facenet'])
    test_embed = test_embed - e_mean.T
    test_trans = np.linalg.pinv(Q_hat).T@test_embed.T
    test_trans = test_trans.T
    # predicting
    o_i = 100
    f_dist, f_ind = facenet_tree.query(np.asarray(test_trans.T), k=o_i)
    
    f_ranklist_dist = {}
    imgs = set()
    for i in range(o_i):
        f_ranklist_dist[str(f_ind[0,i])] = i+1
        imgs.add(f_ind[0,i])
        
    imgs = list(imgs)
    resrank_dist = [f_ranklist_dist]
    final_ranks_dist = {};frank_d = []
    for i in imgs:
        s = 0
        for j in resrank_dist:
            #print(i)
            #print(j)
            #print(str(i) in j)
            if str(i) in j:
                r = j[str(i)]    
                
            else:
                r = o_i+1
            #print('%d %d'%(r,s))
            s+=o_i+1-r
        final_ranks_dist[str(i)] = s
        #print(scorelist)
    frank_d = sorted(imgs,key = lambda x:final_ranks_dist[str(x)],reverse = True)
    frank_d = frank_d[:10]
    
    result_imgs_d = []
    for i in frank_d:
        result_imgs_d.append(f_identity[i])
    return result_imgs_d

data_directory = '/home/pratik/Projects/Ensemble Face Search/Images'
directory = '/home/pratik/Projects/Ensemble Face Search/Test Image/robert downey jr.jpeg'
result_imgs_d = predict(directory,data_directory)
 

import cv2
face_imgs = []
for face in result_imgs_d:
    name = face
    fldr = f_data[face]['name']
    x,y,w,h = f_data[face]['box']
    i = face.split('_')[-1]
    path = data_directory + '/' + fldr + '/' + i
    img = cv2.imread(path)
    face_img = img[y:y+h,x:x+w]
    face_img = cv2.resize(face_img,(64,64))
    if len(face_imgs)==0:
        face_imgs = face_img
    else:
        face_imgs = np.concatenate((face_imgs,face_img),axis = 1)
cv2.imwrite('Result.png',face_imgs)
cv2.imshow()
if cv2.waitKey(0) & 0xFF == 'c':
    cv2.destroyAllWindows()

#img = cv2.imread('C:/Users/student/Desktop/P/Ensemble Face Search_2/test1.jpeg')
#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

wcss = []

for i in range(1,110):
    kmeans = KMeans(n_clusters = i , max_iter = 300 , n_init = 10 , init = 'k-means++' , random_state = 0)
    kmeans.fit(face_trans_embed)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(10,60) , wcss[10:60])
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

#Applying the Kmeans to dataset
kmeans = KMeans(n_clusters = 5 , max_iter = 300 , init = 'k-means++' , n_init = 10 , random_state = 0)
y_kmeans = kmeans.fit_predict(face_trans_embed)