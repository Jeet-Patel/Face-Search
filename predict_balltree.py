# -*- coding: utf-8 -*-

# Importing Libraries
from Extract import face_size,detector,models,Extract
import pickle
import json
import numpy as np

# load ball trees
with open("Ball Trees/facenet_tree.pkl", "rb") as data:
    f_tree = pickle.load(data)
data.close()

with open("Ball Trees/deepface_tree.pkl", "rb") as data:
    d_tree = pickle.load(data)
data.close()

with open("Ball Trees/vggface_tree.pkl", "rb") as data:
    v_tree = pickle.load(data)
data.close()

with open("Ball Trees/openface_tree.pkl", "rb") as data:
    o_tree = pickle.load(data)
data.close()

with open('data/face_dict.json') as json_file:
    f_data = json.load(json_file)
json_file.close()

with open('data/face_identity.json') as json_file:
    f_identity = json.load(json_file)
json_file.close()

with open("data/face_embeddings_dict.pkl", "rb") as data:
    face_embeddings_dict = pickle.load(data)
data.close()

def predict(directory,data_directory):
    
    extract_info = Extract(directory, face_size, detector, models, predict=1)
    face_embed_models = extract_info.get_face_data(directory, face_size, models)
    
    # predicting
    o_i = 100
    f_dist, f_ind = f_tree.query(np.asarray(face_embed_models['facenet']), k=o_i)
    d_dist, d_ind = d_tree.query(np.asarray(face_embed_models['deepface']), k=o_i)
    v_dist, v_ind = v_tree.query(np.asarray(face_embed_models['vggface']), k=o_i)
    o_dist, o_ind = o_tree.query(np.asarray(face_embed_models['openface']), k=o_i)
    tree_inds = {'facenet':f_ind,
                 'deepface':d_ind,
                 'vggface':v_ind,
                 'openface':o_ind}
    
    
    norm_embeds = {'facenet':[],
                   'deepface':[],
                   'vggface':[],
                   'openface':[]}
    for keys in list(face_embeddings_dict.keys()):
        embeds = face_embeddings_dict[keys]
        for i,embed in enumerate(embeds):
            #print(embed)
            norm = np.linalg.norm(embed)
            #print(norm)
            norm_embeds[keys].append(embed/norm)
    
    test_norm_embed = {}
    for keys in list(face_embed_models.keys()):
        embeds = face_embed_models[keys]
        for i,embed in enumerate(embeds):
            #print(embed)
            norm = np.linalg.norm(embed)
            #print(norm)
            test_norm_embed[keys] = embed/norm
    
    
    tree_cos = {'facenet':[],
                'deepface':[],
                'vggface':[],
                'openface':[]}
    for key in tree_inds.keys():
        #print(key)
        t_ind = tree_inds[key][0,:]
        # print(tree_inds[key])
        q = test_norm_embed[key]
        for i in t_ind:
            emb = norm_embeds[key][i]
            s = 1 + (np.inner(emb,q))/2
            tree_cos[key].append(s)
    
    t_cos_ranks = {}
    for k in tree_inds.keys():
        t_ind =[x for x in tree_inds[k][0,:]]
        a = sorted(t_ind,key = lambda x:tree_cos[k][t_ind.index(x)], reverse = True)
        t_cos_ranks[k] = a
    
    
    f_ranklist_cos = {};d_ranklist_cos = {};v_ranklist_cos = {};o_ranklist_cos = {}
    f_ranklist_dist = {};d_ranklist_dist = {};v_ranklist_dist = {}; o_ranklist_dist = {}
    imgs = set()
    for i in range(o_i):
        f_ranklist_dist[str(f_ind[0,i])] = i+1;f_ranklist_cos[str(t_cos_ranks['facenet'][i])] = i+1
        d_ranklist_dist[str(d_ind[0,i])] = i+1;d_ranklist_cos[str(t_cos_ranks['deepface'][i])] = i+1
        v_ranklist_dist[str(v_ind[0,i])] = i+1;v_ranklist_cos[str(t_cos_ranks['vggface'][i])] = i+1
        o_ranklist_dist[str(o_ind[0,i])] = i+1;o_ranklist_cos[str(t_cos_ranks['openface'][i])] = i+1
        imgs.add(f_ind[0,i])
        imgs.add(d_ind[0,i])
        imgs.add(v_ind[0,i])
        imgs.add(o_ind[0,i])
    
    imgs = list(imgs)
    resrank_dist = [f_ranklist_dist,d_ranklist_dist,v_ranklist_dist,o_ranklist_dist]
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
    
    # Cos rank prediction
    peoples = {}
    for k in t_cos_ranks.keys():
        t_lst = t_cos_ranks[k]
        for i in t_lst:
            if f_data[f_identity[i]]['name'] not in peoples:
                peoples[f_data[f_identity[i]]['name']] = 1/40
            else:
                peoples[f_data[f_identity[i]]['name']]+=1/40
    
    resrank_cos = [f_ranklist_cos,d_ranklist_cos,v_ranklist_cos,o_ranklist_cos]
    final_ranks_cos = {};frank_c = []
    for i in imgs:
        s = 0
        for j in resrank_cos:
            if str(i) in j:
                r = j[str(i)]
            else:
                r = 10+1
            s +=peoples[f_data[f_identity[i]]['name']]/r
        final_ranks_cos[str(i)] = s
    frank_c = sorted(imgs,key = lambda x:final_ranks_cos[str(x)],reverse = True)    
    frank_c = frank_c[:10]
    
    result_imgs_c = []
    for i in frank_c:
        result_imgs_c.append(f_identity[i])
    return result_imgs_d,result_imgs_c


data_directory = 'Images'
directory = 'Test Image/robert downey jr.jpeg'
result_imgs_d,result_imgs_c = predict(directory,data_directory)
 

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
cv2.imwrite('Result_balltree_dis_rdj.png',face_imgs)
# cv2.imshow()
if cv2.waitKey(0) & 0xFF == 'c':
    cv2.destroyAllWindows()
