# -*- coding: utf-8 -*-

# Load images and extract faces for all images in a directory
import pickle
import json
#import Extract
from Extract import face_size,detector,models,Extract

directory = 'Images'


extract_info = Extract(directory, face_size, detector, models, 0)
face_dict,face_identity,face_embed_models = extract_info.get_face_data(directory, face_size, models)

with open('data/face_dict.json', 'w') as outfile:
    json.dump(face_dict, outfile)

with open('data/face_identity.json', 'w') as outfile:
    json.dump(face_identity, outfile)

with open('data/face_embeddings_dict.pkl', 'wb') as output:
    # Pickle dictionary using protocol 0.
    pickle.dump(face_embed_models, output)