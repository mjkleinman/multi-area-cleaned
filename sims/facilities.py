import os
import pickle

"""
Store/retrieve arbitrary python objects.
"""


def store_object(obj, name, path):
    f = open(os.path.join(path, name), 'wb')
    pickle.dump(obj, f)
    f.close()


def retrieve_object(name, path):
    f = open(os.path.join(path, name), 'rb')
    obj = pickle.load(f)
    f.close()
    return obj
