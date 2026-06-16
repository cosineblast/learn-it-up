
# This module implements DDC-style performance evaluation for a model in a chart.
# It is primarily used during training experimentation, to tell if the models are actually learning anything.

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

import loading

OnsetEvaluation = namedtuple('Evaluation', [
                        'precision',
                        'recall',
                        'fscore',
                        'mean_loss',
                        'threshold_ideal',
                        'raw_auc_score',
                        'aligned_auc_score',
                        'accuracy'
                    ])

def measure_onset_performance(model, chart, features, loss_fn, device):

    # loading model input

    first_frame = frame_of(chart.steps[0])
    last_frame = frame_of(chart.steps[-1])

    frame_features = _get_all_song_context_features(features, first_frame, last_frame)

    difficulties = np.zeros(25)
    difficulties[chart.difficulty-1] = 1.0
    difficulties = np.tile(difficulties, (frame_features.shape[0], 1))

    step_frames = np.array([frame_of(step) for step in chart.steps])
    step_onsets = step_frames - first_frame
    ys = np.zeros(frame_features.shape[0], dtype=np.bool)
    ys[step_onsets] = True

    # running the model

    with torch.no_grad():
        ys_tensor = torch.tensor(ys).float().to(device)
        frame_features = torch.tensor(frame_features).transpose(1, 3).transpose(2, 3).float().to(device)
        difficulties = torch.tensor(difficulties).float().to(device)

        log_scores = model(frame_features, difficulties)

        scores = F.sigmoid(log_scores).detach().cpu().numpy()
        mean_loss = torch.mean(loss_fn(log_scores, ys_tensor)).detach().cpu().numpy()

    # analyzing onsets 

    pred_onsets = _get_pred_onsets(scores)
    real_onsets = set(step_onsets)

    # computing main metrics 

    raw_precisions, raw_recalls, _ = sklearn.metrics.precision_recall_curve(ys, scores)
    raw_auc_score = sklearn.metrics.auc(raw_recalls, raw_precisions)

    aligned_scores = ddc_align_scores_for_sklearn(real_onsets, pred_onsets, scores)

    precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(ys, aligned_scores)
    aligned_auc_score = sklearn.metrics.auc(recalls, precisions)

    precision, recall, fscore, threshold_ideal = get_best_metrics(precisions, recalls, thresholds)

    # computing accuracy

    predicted_steps = np.where(aligned_scores >= threshold_ideal)
    y_labels = np.zeros(aligned_scores.shape[0], dtype=int)
    y_labels[predicted_steps] = 1
    accuracy = sklearn.metrics.accuracy_score(ys.astype(int), y_labels)

    return OnsetEvaluation(precision, recall, fscore, mean_loss, threshold_ideal, raw_auc_score, aligned_auc_score, accuracy)

def get_best_metrics(precisions, recalls, thresholds):
    fscores_denom = precisions + recalls
    fscores_denom[np.where(fscores_denom == 0.0)] = 1.0
    fscores = (2 * (precisions * recalls)) / fscores_denom
    fscore_max_idx = np.argmax(fscores)
    return precisions[fscore_max_idx], recalls[fscore_max_idx], fscores[fscore_max_idx], thresholds[fscore_max_idx]



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

StepEvaluation = namedtuple('StepEvaluation', ['mean_loss', 'accuracy'])

def measure_selection_performance(model, chart, loss_fn, device):

    x, delta, y = loading.steps_to_model_input(chart.steps, 0, len(chart.steps))

    real_y = y

    x = np.stack([x])
    delta = np.stack([delta])
    y = np.stack([y])

    # running the model
    with torch.no_grad():
        x = torch.tensor(x).float().to(device)
        delta = torch.tensor(delta).float().to(device)
        y = torch.tensor(y).to(device)

        log_scores = model(x, delta)

        log_scores = torch.flatten(log_scores,start_dim=0, end_dim=1)
        y = torch.flatten(y)

        predictions = torch.argmax(F.softmax(log_scores), dim=1).detach().cpu().numpy()
        mean_loss = torch.mean(loss_fn(log_scores, y)).detach().cpu().numpy()

    # computing main metrics 

    accuracy = sklearn.metrics.accuracy_score(real_y, predictions)

    return StepEvaluation(mean_loss, accuracy)    

    




    
