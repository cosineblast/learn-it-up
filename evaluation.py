
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

import ssc_util

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

    frame_features = loading.get_all_song_context_features(features, first_frame, last_frame, upshape=True)

    difficulties = np.zeros((1, 25))
    difficulties[0, chart.difficulty-1] = 1.0

    step_frames = np.array([frame_of(step) for step in chart.steps])
    step_onsets = step_frames - first_frame
    ys = np.zeros((1, frame_features.shape[1]), dtype=np.bool)
    ys[0, step_onsets] = True

    # running the model

    with torch.no_grad():
        ys_tensor = torch.tensor(ys).float().to(device)
        frame_features = torch.tensor(frame_features).float().to(device)
        difficulties = torch.tensor(difficulties).float().to(device)

        log_scores = model(frame_features, difficulties)

        scores = F.sigmoid(log_scores).detach().cpu().numpy()
        mean_loss = torch.mean(loss_fn(log_scores, ys_tensor)).detach().cpu().numpy()

        scores = scores.flatten()
        ys = ys.flatten()

    # analyzing onsets 

    real_onsets = set(step_onsets)

    return evaluate_frame_onset_performance(real_onsets, scores, ys, mean_loss)

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

        predictions = torch.argmax(F.softmax(log_scores, dim=1), dim=1).detach().cpu().numpy()
        mean_loss = torch.mean(loss_fn(log_scores, y)).detach().cpu().numpy()

    # computing main metrics 

    accuracy = sklearn.metrics.accuracy_score(real_y, predictions)

    return StepEvaluation(mean_loss, accuracy)    

    

def measure_aligned_onset_performance(model, chart, features, loss_fn, device):
    x, nps, bpms = get_aligned_onset_input(features, chart)
    ys = get_aligned_onset_expected_output(chart)

    with torch.no_grad():
        log_scores = model(
            torch.tensor(x).float().to(device),
            torch.tensor(nps).float().to(device),
            torch.tensor(bpms).float().to(device)
        )

        loss = loss_fn(
            log_scores[0],
            torch.tensor(ys).float().to(device)
        )

        beat_scores = F.sigmoid(log_scores[0]).detach().cpu().numpy()
        mean_loss = torch.mean(loss).detach().cpu().numpy()

    frame_scores = beat_scores_to_frame_scores(chart, beat_scores, frame_count=features.len)
    
    real_onsets = set(frame_of(step) for step in chart.steps)

    frame_ys = np.zeros(features.len)
    frame_ys[list(real_onsets)] = True

    return evaluate_frame_onset_performance(real_onsets, frame_scores, frame_ys, mean_loss)

def beat_scores_to_frame_scores(chart, beat_scores, frame_count):
    beat_count = beat_scores.shape[0]

    beat_fractions = [i/48 for i in range(beat_count * 48)]
    beat_fraction_times = ssc_util.compute_multiple_beats_absolute_times(0.0, chart.bpms, beat_fractions)

    frame_scores = defaultdict(float)

    for i in range(beat_count):
        times = beat_fraction_times[i*48:(i+1)*48]
        scores = beat_scores[i, :]

        assert len(times) == len(scores) == 48

        for j in range(48):
            time = times[j]
            score = scores[j]

            frame = floor(time * FRAMES_PER_SECOND)
            frame_scores[frame] = max(frame_scores[frame], score)

    result = np.zeros(frame_count)
    for i in range(len(frame_scores)):
        result[i] = frame_scores[i]
    return result


        

def get_aligned_onset_input(features, chart):
    import audio_util
    # refactor this and loader implementation into separate function
    array, start, len = features
    
    slice = array[start:len]
    default_value = np.min(slice, axis=0)

    resampled = audio_util.resample_features(slice, chart.beat_start_end_times)

    padding = np.tile(default_value.reshape((1, 1, 80, 3)), (2, 32, 1, 1))
    padded = np.concat([padding, resampled, padding])
    offset = 2

    result = []

    beat_count = floor(chart.steps[-1].time_in_beats)+1

    for beat_index in range(beat_count):
        a = beat_index - 2
        b = beat_index + 2 +1
        result.append(padded[a+offset:b+offset])

    x = np.array(result)[None, :]
    nps = np.array([chart.nps])
    bpms = np.array(chart.beat_bpms[:beat_count])[None, :]

    return x, nps, bpms

def get_aligned_onset_expected_output(chart):
    beat_count = floor(chart.steps[-1].time_in_beats)+1

    return np.array(chart.beat_onset_vectors[:beat_count], dtype=bool)


def evaluate_frame_onset_performance(real_onsets, scores, ys, mean_loss):
    pred_onsets = _get_pred_onsets(scores)

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

