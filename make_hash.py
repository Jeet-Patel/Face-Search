# -*- coding: utf-8 -*-

# loading the Face_embeddings_dict pickle file
import pickle
import numpy as np
from sklearn.neighbors import BallTree
import lshashpy3
from lshashpy3 import LSHash
#import json

with open("data/face_embeddings_dict.pkl", "rb") as data:
    face_embeddings_dict = pickle.load(data)
data.close()

facenet_embeddings = np.asarray(face_embeddings_dict['facenet'])
deepface_embeddings = np.asarray(face_embeddings_dict['deepface'])
vggface_embeddings = np.asarray(face_embeddings_dict['vggface'])
openface_embeddings = np.asarray(face_embeddings_dict['openface'])

lsh_facenet = LSHash(hash_size=10, input_dim=128, num_hashtables=100,
             storage_config={'dict': None},
             matrices_filename='LS Hashes/facenet.npz',
             hashtable_filename='LS Hashes/facenet_hash.npz',
             overwrite=False)
i=0
for row in facenet_embeddings:
    i+=1
    lsh_facenet.index(row, extra_data=f"face{i}")
lsh_facenet.save()

lsh_deepface = LSHash(hash_size=10, input_dim=4096, num_hashtables=100,
                     storage_config={'dict': None},
                     matrices_filename='LS Hashes/deepface.npz',
                     hashtable_filename='LS Hashes/deepface_hash.npz',
                     overwrite=False)
i = 0
for row in deepface_embeddings:
    i += 1
    lsh_deepface.index(row, extra_data=f"face{i}")
lsh_deepface.save()

lsh_vggface = LSHash(hash_size=10, input_dim=2622, num_hashtables=100,
                     storage_config={'dict': None},
                     matrices_filename='LS Hashes/vggface.npz',
                     hashtable_filename='LS Hashes/vggface_hash.npz',
                     overwrite=False)
i = 0
for row in vggface_embeddings:
    i += 1
    lsh_vggface.index(row, extra_data=f"face{i}")
lsh_vggface.save()

lsh_openface = LSHash(hash_size=10, input_dim=128, num_hashtables=100,
                     storage_config={'dict': None},
                     matrices_filename='LS Hashes/openface.npz',
                     hashtable_filename='LS Hashes/openface_hash.npz',
                     overwrite=False)
i = 0
for row in openface_embeddings:
    i += 1
    lsh_openface.index(row, extra_data=f"face{i}")
lsh_openface.save()
