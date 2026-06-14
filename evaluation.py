
FRAMES_PER_SECOND=100
CONTEXT_RADIUS=7

from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy

import sklearn

from collections import namedtuple, defaultdict

Evaluation = namedtuple('Evaluation', [
                        'true_positives',
                        'label_positives',
                        'total_positives',
                        'mean_loss',
                        'raw_auc_score',
                        'aligned_auc_score'])

def measure_onset_performance(model, chart, features, loss_fn, device):
    first_frame = frame_of(chart.steps[0])
    last_frame = frame_of(chart.steps[-1])

    frame_features = _get_all_song_context_features(features, first_frame, last_frame)

    difficulties = np.zeros(25)
    difficulties[chart.difficulty-1] = 1.0
    difficulties = np.tile(difficulties, (frame_features.shape[0], 1))

    step_frames = np.array([frame_of(step) for step in chart.steps]) - first_frame
    ys = np.zeros(frame_features.shape[0], dtype=np.bool)
    ys[step_frames] = True

    with torch.no_grad():
        ys_tensor = torch.tensor(ys).float().to(device)
        frame_features = torch.tensor(frame_features).transpose(1, 3).transpose(2, 3).float().to(device)
        difficulties = torch.tensor(difficulties).float().to(device)

        log_scores = model(frame_features, difficulties)

        scores = F.sigmoid(log_scores).detach().cpu().numpy()
        mean_loss = torch.mean(loss_fn(log_scores, ys_tensor)).detach().cpu().numpy()

    pred_onsets = _get_pred_onsets(scores)
    real_onsets = set(frame_of(step) for step in chart.steps)

    total_positives = len(real_onsets)
    label_positives = len(pred_onsets)
    true_positives = _compute_true_positives(pred_onsets, real_onsets)

    aligned_scores = ddc_align_scores_for_sklearn(real_onsets, pred_onsets, scores)

    raw_auc_score = precision_recall_auc(ys, scores)
    aligned_auc_score = precision_recall_auc(ys, aligned_scores)

    return Evaluation(true_positives, label_positives, total_positives, mean_loss, raw_auc_score, aligned_auc_score)

def precision_recall_auc(ys, scores):
    precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(ys, scores)
    return sklearn.metrics.auc(recalls, precisions)


def ddc_align_scores_for_sklearn(true_onsets, pred_onsets, scores, tolerance=2):
    # This function is adapted from DDC.

    # Build one-to-many dicts of candidate matches
    true_to_pred = defaultdict(list)
    pred_to_true = defaultdict(list)

    for true_idx in true_onsets:
        for pred_idx in range(true_idx - tolerance, true_idx + tolerance + 1):
            if pred_idx in pred_onsets:
                true_to_pred[true_idx].append(pred_idx)
                pred_to_true[pred_idx].append(true_idx)

    # Create alignments
    true_to_pred_confidence = {}
    pred_idxs_used = set()
    for pred_idx, true_idxs in pred_to_true.items():
        true_idx_use = true_idxs[0]
        if len(true_idxs) > 1:
            for true_idx in true_idxs:
                if len(true_to_pred[true_idx]) == 1:
                    true_idx_use = true_idx
                    break
        true_to_pred_confidence[true_idx_use] = scores[pred_idx]
        assert pred_idx not in pred_idxs_used
        pred_idxs_used.add(pred_idx)

    # Create confidence list
    y_scores = np.zeros_like(scores)
    for true_idx, confidence in true_to_pred_confidence.items():
        y_scores[true_idx] = confidence

    # Add remaining false positives
    for fp_idx in pred_onsets - pred_idxs_used:
        y_scores[fp_idx] = scores[fp_idx]

    # the original DDC limited the scores between the valid range,
    # but we dont have to do that because scores is already restrained
    # for that range
    return y_scores

def _get_pred_onsets(scores):
    window_width = 5
    
    window = np.hamming(window_width)
    onsets = np.convolve(scores, window, mode='same')

    # inspired by DDC
    maxima = scipy.signal.argrelextrema(onsets, np.greater_equal, order=1)[0]
    return set(maxima)

def _compute_true_positives(pred_onsets, real_onsets):
    align_tolerance = 2
    offset_range = list(range(-align_tolerance, align_tolerance+1))

    return sum(
        any(real_onset + offset in pred_onsets for offset in offset_range)
        for real_onset in real_onsets
    )


def frame_of(step):
    return floor(step.time_in_seconds * FRAMES_PER_SECOND)

def _get_all_song_context_features(features_view, first_frame, last_frame):
    """Returns a list with all the 15-frame context windows of the given feature view"""
    features, start, length = features_view

    frame_indices = start + np.arange(first_frame,last_frame+1)
    total_frames = frame_indices.shape[0]

    context_indices = np.tile(np.arange(-7, 8), (total_frames, 1)).transpose((1,0)) + frame_indices
    context_indices = context_indices.transpose((1, 0))
    frame_features = features[context_indices]

    return frame_features

    

    




    
