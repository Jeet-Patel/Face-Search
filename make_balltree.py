# -*- coding: utf-8 -*-

# loading the Face_embeddings_dict pickle file
import pickle
import numpy as np
from sklearn.neighbors import BallTree
#import json

with open("data/face_embeddings_dict.pkl", "rb") as data:
    face_embeddings_dict = pickle.load(data)
data.close()
"""
with open('data/face_dict.json') as json_file:
    f_data = json.load(json_file)
json_file.close()

with open('data/face_identity.json') as json_file:
    f_identity = json.load(json_file)
json_file.close()
"""
facenet_embeddings = np.asarray(face_embeddings_dict['facenet'])
deepface_embeddings = np.asarray(face_embeddings_dict['deepface'])
vggface_embeddings = np.asarray(face_embeddings_dict['vggface'])
openface_embeddings = np.asarray(face_embeddings_dict['openface'])

# Make Ball Tree
facenet_tree = BallTree(facenet_embeddings, leaf_size=40)
deepface_tree = BallTree(deepface_embeddings, leaf_size=40)
vggface_tree = BallTree(vggface_embeddings, leaf_size=40)
openface_tree = BallTree(openface_embeddings, leaf_size=40)

# Saving the trees
with open('Ball Trees/facenet_tree.pkl', 'wb') as output:
    pickle.dump(facenet_tree, output)
output.close()
with open('Ball Trees/deepface_tree.pkl', 'wb') as output:
    pickle.dump(deepface_tree, output)
output.close()
with open('Ball Trees/vggface_tree.pkl', 'wb') as output:
    pickle.dump(vggface_tree, output)
output.close()
with open('Ball Trees/openface_tree.pkl', 'wb') as output:
    pickle.dump(openface_tree, output)
output.close()
