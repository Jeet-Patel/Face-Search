# -*- coding: utf-8 -*-

# Importing Libraries
import cv2
from Extract import face_size, detector, models, Extract
import pickle
import json
import numpy as np
import annoy
from annoy import AnnoyIndex

f_tree = AnnoyIndex(128, 'euclidean')
f_tree.load('Annoy Trees/facenet_annoytree.ann')

d_tree = AnnoyIndex(4096, 'euclidean')
d_tree.load('Annoy Trees/deepface_annoytree.ann')

v_tree = AnnoyIndex(2622, 'euclidean')
v_tree.load('Annoy Trees/vggface_annoytree.ann')

o_tree = AnnoyIndex(128, 'euclidean')
o_tree.load('Annoy Trees/openface_annoytree.ann')

# load ball trees
with open("Ball Trees/facenet_tree.pkl", "rb") as data:
    f_balltree = pickle.load(data)
data.close()

with open("Ball Trees/deepface_tree.pkl", "rb") as data:
    d_balltree = pickle.load(data)
data.close()

with open("Ball Trees/vggface_tree.pkl", "rb") as data:
    v_balltree = pickle.load(data)
data.close()

with open("Ball Trees/openface_tree.pkl", "rb") as data:
    o_balltree = pickle.load(data)
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


def predict(directory, data_directory):

    extract_info = Extract(directory, face_size, detector, models, predict=1)
    face_embed_models = extract_info.get_face_data(
        directory, face_size, models)

    # predicting
    o_i = 100
    f_dist_balltree, f_ind_balltree = f_balltree.query(np.asarray(
        face_embed_models['facenet']), k=o_i)
    d_dist_balltree, d_ind_balltree = d_balltree.query(np.asarray(
        face_embed_models['deepface']), k=o_i)
    v_dist_balltree, v_ind_balltree = v_balltree.query(np.asarray(
        face_embed_models['vggface']), k=o_i)
    o_dist_balltree, o_ind_balltree = o_balltree.query(np.asarray(
        face_embed_models['openface']), k=o_i)
    tree_inds_balltree = {'facenet': f_ind_balltree,
                 'deepface': d_ind_balltree,
                 'vggface': v_ind_balltree,
                 'openface': o_ind_balltree}

    norm_embeds_balltree = {'facenet': [],
                   'deepface': [],
                   'vggface': [],
                   'openface': []}
    f_ind, f_dist = f_tree.get_nns_by_vector(np.asarray(
        face_embed_models['facenet'][0]), o_i, include_distances=True)
    d_ind, d_dist = d_tree.get_nns_by_vector(np.asarray(
        face_embed_models['deepface'][0]), o_i, include_distances=True)
    v_ind, v_dist = v_tree.get_nns_by_vector(np.asarray(
        face_embed_models['vggface'][0]), o_i, include_distances=True)
    o_ind, o_dist = o_tree.get_nns_by_vector(np.asarray(
        face_embed_models['openface'][0]), o_i, include_distances=True)
    tree_inds = {'facenet': f_ind,
                 'deepface': d_ind,
                 'vggface': v_ind,
                 'openface': o_ind}

    norm_embeds = {'facenet': [],
                   'deepface': [],
                   'vggface': [],
                   'openface': []}

    for keys in list(face_embeddings_dict.keys()):
        embeds = face_embeddings_dict[keys]
        for i, embed in enumerate(embeds):
            #print(embed)
            norm = np.linalg.norm(embed)
            #print(norm)
            norm_embeds_balltree[keys].append(embed/norm)

    test_norm_embed_balltree = {}
    for keys in list(face_embed_models.keys()):
        embeds = face_embed_models[keys]
        for i, embed in enumerate(embeds):
            #print(embed)
            norm = np.linalg.norm(embed)
            #print(norm)
            test_norm_embed_balltree[keys] = embed/norm

    for keys in list(face_embeddings_dict.keys()):
        embeds = face_embeddings_dict[keys]
        for i, embed in enumerate(embeds):
            #print(embed)
            norm = np.linalg.norm(embed)
            #print(norm)
            norm_embeds[keys].append(embed/norm)

    test_norm_embed = {}
    for keys in list(face_embed_models.keys()):
        embeds = face_embed_models[keys]
        for i, embed in enumerate(embeds):
            #print(embed)
            norm = np.linalg.norm(embed)
            #print(norm)
            test_norm_embed[keys] = embed/norm

    tree_cos_balltree = {'facenet': [],
                'deepface': [],
                'vggface': [],
                'openface': []}
    for key in tree_inds_balltree.keys():
        #print(key)
        t_ind = tree_inds_balltree[key][0, :]
        # print(tree_inds_balltree[key])
        q = test_norm_embed_balltree[key]
        for i in t_ind:
            emb = norm_embeds_balltree[key][i]
            s = 1 + (np.inner(emb, q))/2
            tree_cos_balltree[key].append(s)

    t_cos_ranks_balltree = {}
    for k in tree_inds_balltree.keys():
        t_ind = [x for x in tree_inds_balltree[k][0, :]]
        a = sorted(
            t_ind, key=lambda x: tree_cos_balltree[k][t_ind.index(x)], reverse=True)
        t_cos_ranks_balltree[k] = a

    tree_cos = {'facenet': [],
                'deepface': [],
                'vggface': [],
                'openface': []}
    for key in tree_inds.keys():
        #print(key)
        t_ind = tree_inds[key]
        # print(tree_inds[key])
        q = test_norm_embed[key]
        for i in t_ind:
            emb = norm_embeds[key][i]
            s = 1 + (np.inner(emb, q))/2
            tree_cos[key].append(s)

    t_cos_ranks = {}
    for k in tree_inds.keys():
        t_ind = [x for x in tree_inds[k]]
        a = sorted(
            t_ind, key=lambda x: tree_cos[k][t_ind.index(x)], reverse=True)
        t_cos_ranks[k] = a

    f_ranklist_cos_balltree = {}
    d_ranklist_cos_balltree = {}
    v_ranklist_cos_balltree = {}
    o_ranklist_cos_balltree = {}
    f_ranklist_dist_balltree = {}
    d_ranklist_dist_balltree = {}
    v_ranklist_dist_balltree = {}
    o_ranklist_dist_balltree = {}
    f_ranklist_cos = {}
    d_ranklist_cos = {}
    v_ranklist_cos = {}
    o_ranklist_cos = {}
    f_ranklist_dist = {}
    d_ranklist_dist = {}
    v_ranklist_dist = {}
    o_ranklist_dist = {}
    imgs = set()
    for i in range(o_i):
        f_ranklist_dist_balltree[str(f_ind_balltree[0, i])] = i+1
        f_ranklist_cos_balltree[str(t_cos_ranks_balltree['facenet'][i])] = i+1
        d_ranklist_dist_balltree[str(d_ind_balltree[0, i])] = i+1
        d_ranklist_cos_balltree[str(t_cos_ranks_balltree['deepface'][i])] = i+1
        v_ranklist_dist_balltree[str(v_ind_balltree[0, i])] = i+1
        v_ranklist_cos_balltree[str(t_cos_ranks_balltree['vggface'][i])] = i+1
        o_ranklist_dist_balltree[str(o_ind_balltree[0, i])] = i+1
        o_ranklist_cos_balltree[str(t_cos_ranks_balltree['openface'][i])] = i+1
        imgs.add(f_ind_balltree[0, i])
        imgs.add(d_ind_balltree[0, i])
        imgs.add(v_ind_balltree[0, i])
        imgs.add(o_ind_balltree[0, i])

        f_ranklist_dist[str(f_ind[i])] = i+1
        f_ranklist_cos[str(t_cos_ranks['facenet'][i])] = i+1
        d_ranklist_dist[str(d_ind[i])] = i+1
        d_ranklist_cos[str(t_cos_ranks['deepface'][i])] = i+1
        v_ranklist_dist[str(v_ind[i])] = i+1
        v_ranklist_cos[str(t_cos_ranks['vggface'][i])] = i+1
        o_ranklist_dist[str(o_ind[i])] = i+1
        o_ranklist_cos[str(t_cos_ranks['openface'][i])] = i+1
        imgs.add(f_ind[i])
        imgs.add(d_ind[i])
        imgs.add(v_ind[i])
        imgs.add(o_ind[i])

    imgs = list(imgs)
    resrank_ensemble = [f_ranklist_dist_balltree, d_ranklist_dist_balltree,
                        v_ranklist_dist_balltree, o_ranklist_dist_balltree, 
                        f_ranklist_cos_balltree, d_ranklist_cos_balltree, 
                        v_ranklist_cos_balltree, o_ranklist_cos_balltree,
                        f_ranklist_dist, d_ranklist_dist,
                        v_ranklist_dist, o_ranklist_dist,
                        f_ranklist_cos, d_ranklist_cos,
                        v_ranklist_cos, o_ranklist_cos]
    final_ranks = {}
    frank = []
    for i in imgs:
        s = 0
        for j in resrank_ensemble:
            #print(i)
            #print(j)
            #print(str(i) in j)
            if str(i) in j:
                r = j[str(i)]
            else:
                r = o_i+1
            #print('%d %d'%(r,s))
            s += o_i+1-r
        final_ranks[str(i)] = s
        #print(scorelist)
    frank = sorted(
        imgs, key=lambda x: final_ranks[str(x)], reverse=True)
    frank = frank[:10]

    result_imgs_ensemble = []
    for i in frank:
        result_imgs_ensemble.append(f_identity[i])

    resrank_ensemble_d = [f_ranklist_dist_balltree, d_ranklist_dist_balltree,
                          v_ranklist_dist_balltree, o_ranklist_dist_balltree,
                          f_ranklist_dist, d_ranklist_dist,
                          v_ranklist_dist, o_ranklist_dist]
    final_ranks_dis = {}
    frank_dis = []
    for i in imgs:
        s = 0
        for j in resrank_ensemble_d:
            #print(i)
            #print(j)
            #print(str(i) in j)
            if str(i) in j:
                r = j[str(i)]
            else:
                r = o_i+1
            #print('%d %d'%(r,s))
            s += o_i+1-r
        final_ranks_dis[str(i)] = s
        #print(scorelist)
    frank_dis = sorted(
        imgs, key=lambda x: final_ranks_dis[str(x)], reverse=True)
    frank_dis = frank_dis[:10]

    result_imgs_d = []
    for i in frank_dis:
        result_imgs_d.append(f_identity[i])

    # Cos rank prediction
    peoples_balltree = {}
    for k in t_cos_ranks_balltree.keys():
        t_lst = t_cos_ranks_balltree[k]
        for i in t_lst:
            if f_data[f_identity[i]]['name'] not in peoples_balltree:
                peoples_balltree[f_data[f_identity[i]]['name']] = 1/40
            else:
                peoples_balltree[f_data[f_identity[i]]['name']] += 1/40

    resrank_ensemble_c = [f_ranklist_cos_balltree, d_ranklist_cos_balltree,
                          v_ranklist_cos_balltree, o_ranklist_cos_balltree,
                          f_ranklist_cos, d_ranklist_cos,
                          v_ranklist_cos, o_ranklist_cos]
    final_ranks_cos_balltree = {}
    frank_c_balltree = []
    for i in imgs:
        s = 0
        for j in resrank_ensemble_c:
            if str(i) in j:
                r = j[str(i)]
            else:
                r = 10+1
            s += peoples_balltree[f_data[f_identity[i]]['name']]/r
        final_ranks_cos_balltree[str(i)] = s
    frank_c_balltree = sorted(imgs, key=lambda x: final_ranks_cos_balltree[str(x)], reverse=True)
    frank_c_balltree = frank_c_balltree[:10]

    result_imgs_c = []
    for i in frank_c_balltree:
        result_imgs_c.append(f_identity[i])

    return result_imgs_ensemble, result_imgs_c, result_imgs_d

data_directory = 'Images'
directory = 'Test Image/robert downey jr.jpeg'
result_imgs_ensemble, result_imgs_c, result_imgs_d = predict(directory, data_directory)
# print(result_imgs_ensemble)
# print(result_imgs_c)
# print(result_imgs_d)
face_imgs_e = []
face_imgs_c = []
face_imgs_d = []
face_imgs = []
for face in result_imgs_ensemble:
    name = face
    fldr = f_data[face]['name']
    x, y, w, h = f_data[face]['box']
    i = face.split('_')[-1]
    path = data_directory + '/' + fldr + '/' + i
    img = cv2.imread(path)
    face_img = img[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (64, 64))
    if len(face_imgs_e) == 0:
        face_imgs_e = face_img
    else:
        face_imgs_e = np.concatenate((face_imgs_e, face_img), axis=1)
for face in result_imgs_c:
    name = face
    fldr = f_data[face]['name']
    x, y, w, h = f_data[face]['box']
    i = face.split('_')[-1]
    path = data_directory + '/' + fldr + '/' + i
    img = cv2.imread(path)
    face_img = img[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (64, 64))
    if len(face_imgs_c) == 0:
        face_imgs_c = face_img
    else:
        face_imgs_c = np.concatenate((face_imgs_c, face_img), axis=1)
for face in result_imgs_d:
    name = face
    fldr = f_data[face]['name']
    x, y, w, h = f_data[face]['box']
    i = face.split('_')[-1]
    path = data_directory + '/' + fldr + '/' + i
    img = cv2.imread(path)
    face_img = img[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (64, 64))
    if len(face_imgs_d) == 0:
        face_imgs_d = face_img
    else:
        face_imgs_d = np.concatenate((face_imgs_d, face_img), axis=1)

face_imgs = face_imgs_e
face_imgs = np.concatenate((face_imgs, face_imgs_c), axis=0)
face_imgs = np.concatenate((face_imgs, face_imgs_d), axis=0)
cv2.imwrite('Result_ensemble_rdj.png', face_imgs)
# cv2.imshow()
if cv2.waitKey(0) & 0xFF == 'c':
    cv2.destroyAllWindows()
