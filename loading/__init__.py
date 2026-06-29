# loading.py:
# This module contains the code responsible for
# loading refined ssc files and audio feature files into the input formats
# expected by the models.

from collections import namedtuple 

import numpy as np
import torch
import pickle

import random

"""
A featureview is a tuple that represents a slice of the content of a features array.
The real content starts at start, and ends at start+len, but it may be ok to access
the array outside of this range, considering there might be padding.
"""
FeatureView = namedtuple('FeatureView', ['array', 'start', 'len'])


CONTEXT_RADIUS=7

# right now the cache is not really necessary since we store everything in memory during training anyway.
class LoadFeaturesCached():
    """
    Loads features from disk applying padding and caching on paths.
    """
    def __init__(self, normalize_features=True):
        self.cache = {}
        self.normalize_features = normalize_features

    def __call__(self, path):
        if path in self.cache:
            return self.cache[path]

        with open(path, 'rb') as f:
            features = pickle.load(f)

            view = prepare_features(features, normalize=self.normalize_features)

            self.cache[path] = view

            return view

def prepare_features(features, normalize=True):
    """Adds 7 frames of padding before and after the given features, padded with minimum values of the given features,
       and normalizes features to mean 0 and standard deviation 1.
       Returns a feature view of the non-padded features. 
       """

    if normalize:
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    default_value = np.min(features, axis=0)
    padding = np.tile(default_value.reshape((1, 80, 3)), (CONTEXT_RADIUS, 1, 1))
    padded = np.concat([padding, features, padding])

    return FeatureView(padded, CONTEXT_RADIUS, features.shape[0])

def get_all_song_context_features(features_view, first_frame, last_frame, upshape=False):
    """Returns a list with all the 15-frame context windows of the given feature view,
       within the given inclusive frame range"""

    features, start, length = features_view

    frame_indices = start + np.arange(first_frame,last_frame+1)
    total_frames = frame_indices.shape[0]

    context_indices = np.tile(np.arange(-7, 8), (total_frames, 1)).transpose((1,0)) + frame_indices
    context_indices = context_indices.transpose((1, 0))
    frame_features = features[context_indices]

    return frame_features[None, :] if upshape else frame_features


def MaskAndPaddingTransform(unroll_length, skip=None):
    """
    This transform function takes a tuple containing multiple tensors X1, ..., Xn of
    possibly different shapes, but all matching the size of their their main axis as `k` (X1.shape[0] = ... = Xn.shape[0] = k).
    If k is less than the unroll length n, then all those vectors are padded with zeroes in their main axis to size n.
    Additionally, an additional element `mask` is added to the returned tuple, containing a vector of shape (n,) with ones
    in the first k positions, and zero in the last n-k positions.
    """
    def maskAndPad(stuff):
        assert isinstance(stuff, tuple)

        size    = stuff[0].shape[0]

        new_stuff = tuple(
            np.pad(x,[(0, unroll_length - x.shape[0])] + [(0, 0)] * (len(x.shape)-1))
            if skip is None or i != skip else x
            for i, x in enumerate(stuff))

        mask = np.pad(np.ones(size), (0, unroll_length - stuff[0].shape[0]))

        return *new_stuff, mask

    return maskAndPad  
