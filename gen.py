
import argparse

import re
import sys
import itertools
import time

from pathlib import Path

from collections import defaultdict

FRAMES_PER_SECOND=100

def generate_stepfile(audio_path,
                      difficulty_str,
                      onset_model_path,
                      selection_model_path,
                      result_path,
                      device,
                      seed=None):

    difficulty = check_difficulty(difficulty_str)

    print('Loading libraries... (let the bass kick)')
    import torch
    import audio_util
    import loading
    import scipy
    import models
    import numpy as np

    if seed is not None:
        torch.manual_seed(seed)

    print("Extracting audio features of {}... (JOOOOOOOO)".format(audio_path))
    loader = audio_util.AudioFeatureLoader(use_tqdm=True)
    features = loader.load(audio_path)

    print("Loading onset model file {}... (AAAAE)".format(onset_model_path))
    onset_model_state = torch.load(onset_model_path, weights_only=False, map_location=torch.device(device))
    onset_model = models.PumpItUpConvolutionCNNOnset()
    onset_model.load_state_dict(onset_model_state)

    print()

    print("Loading selection model file {}... (A-A-I-A-U)".format(selection_model_path))
    selection_model_state = torch.load(selection_model_path, weights_only=False, map_location=torch.device(device))
    selection_model = models.PumpItUpConvolutionSelectionLSTM()
    selection_model.load_state_dict(selection_model_state)

    print()

    print('Running onset model... (JOOOOOOOOO)')

    placements = run_onset_model(features, onset_model, difficulty, device)

    # now run selection model
    print('Running selection model... (AAE-O-A-A-U-U-A Eeeeeeee)')

    steps = run_selection_model(placements, selection_model, device)

    build_stepfile(placements, steps, audio_path, result_path, difficulty_str)

def run_onset_model(features, onset_model, difficulty, device):
    import torch
    import numpy as np
    import loading
    import torch.nn.functional as F
    import scipy
    
    features = loading.prepare_features(features)
    frame_features = loading.get_all_song_context_features(features, 0, features.len-1)

    difficulties = np.zeros(25)
    difficulties[difficulty-1] = 1.0
    difficulties = np.tile(difficulties, (frame_features.shape[0], 1))

    onset_model.eval()

    with torch.no_grad():
        frame_features = torch.tensor(frame_features).transpose(1, 3).transpose(2, 3).float().to(device)
        difficulties = torch.tensor(difficulties).float().to(device)

        log_scores = onset_model(frame_features, difficulties)
        scores = F.sigmoid(log_scores).detach().cpu().numpy()
    
    # TODO: save best threhold in alongside model
    threshold = BEST_THRESHOLDS[difficulty]

    window = np.hamming(5)
    onsets = np.convolve(scores, window, mode='same')
    maxima = scipy.signal.argrelextrema(onsets, np.greater_equal, order=1)[0]
    maxima = set(maxima)

    return sorted(onset / FRAMES_PER_SECOND for onset in maxima if scores[onset] > threshold)

# Derived by inspecting the best onsets in onset_notebook.py
BEST_THRESHOLDS = {
    1: 0.09573398530483246,
    2: 0.09573398530483246,
    3: 0.19311115145683289,
    4: 0.0552724227309227,
    5: 0.26338380575180054,
    6: 0.3407045900821686,
    7: 0.1542433351278305,
    8: 0.2203090637922287,
    9: 0.2203090637922287,
    10: 0.2543484568595886,
    11: 0.19764167070388794,
    12: 0.2639240026473999,
    13: 0.23878812789916992,
    14: 0.1600670963525772,
    15: 0.29371264576911926,
    16: 0.2168826311826706,
    17: 0.21294879913330078,
    18: 0.31380966305732727,
    19: 0.22536088526248932,
    20: 0.1528964787721634,
    21: 0.22844624519348145,
    22: 0.18381500244140625,
    23: 0.1304573118686676,
    24: 0.16991640627384186,
    25: 0.16991640627384186,
}
    

def run_selection_model(placements, selection_model, device):
    import torch.nn.functional as F
    import torch
    import numpy as np
    import loading
    
    steps = []

    previous_prediction = None
    previous_state = None
    for i, placement in enumerate(placements):

        is_first = 1.0 if i == 0 else 0.0
        time_before = 0.0 if i == 0 else placements[i] - placements[i-1]
        time_after = 0.0 if i == len(placements)-1 else placements[i+1] - placements[i]
        delta = np.array([time_before, time_after, is_first])

        x = (np.zeros((5, 4))
            if previous_prediction is None 
            else loading.stepcode_to_bag_tensor(loading.index_to_stepcode(previous_prediction)))

        x = x.reshape((1, 1, 5, 4))
        delta = delta.reshape((1, 1, 3))

        with torch.no_grad():
            #selection_model.eval()
            
            x = torch.tensor(x).float().to(device)
            delta = torch.tensor(delta).float().to(device)

            log_scores, state = selection_model(x, delta, initial_state=previous_state, return_state=True)

            log_scores = torch.flatten(log_scores,start_dim=0, end_dim=1)
            predictions = torch.argmax(F.softmax(log_scores, dim=1), dim=1).detach().cpu().numpy()

        steps.append(loading.index_to_stepcode(predictions[0]))

        previous_state = state
        previous_prediction = predictions[0]

    return steps

def check_difficulty(difficulty_str):
    if not re.match(r'S\d+', difficulty_str):
        print('error: Invalid difficulty! Expecting diffculty of the form S1...S25')
        sys.exit(1)

    difficulty = int(difficulty_str[1:])

    if not (1 <= difficulty <= 25):
        print('error: Difficulty must be in 1..25 range')
        sys.exit(1)

    return difficulty
    

def build_stepfile(placements, steps, audio_path, result_path, difficulty_str):
    import msdparser

    stepfile=[]

    def add(key, value):
        stepfile.append(msdparser.MSDParameter([key, value]))

    add('VERSION', '0.83')
    add('TITLE', 'unknown')
    add('SUBTITLE', '')
    add('ARTIST', 'PumpItUpConvolutionV0')
    add('TITLETRANSLIT', '')
    add('SUBTITLETRANSLIT', '')
    add('ARTISTTRANSLIT', '')
    add('GENRE', '')
    add('CREDIT', '')
    add('MUSIC',
        Path(audio_path).name if Path(audio_path).parent.absolute() == Path(result_path).parent.absolute()
        else str(Path(audio_path).absolute()))
    add('BANNER', '')
    add('BACKGROUND', '')
    add('CDTITLE', '')
    add('SAMPLESTART', '0.000000')
    add('SAMPLELENGTH', '0.000000')
    add('SELECTABLE', 'YES')
    add('OFFSET', str(-placements[0]))
    add('BPMS', '0.000000=120.0000')
    add('STOPS', '')
    add('BGCHANGES', '')
    add('FGCHANGES', '')
    add('NOTEDATA', '')

    add('STEPSTYPE', 'pump-single');
    add('DESCRIPTION', '');
    add('DIFFICULTY', 'Edit');
    add('METER', '15');
    add('RADARVALUES', '0,0,0,0,0');

    notes_per_measure = defaultdict(list)

    for placement, step in zip(placements, steps):
        measure = int(placement / SECONDS_PER_MEASURE)
        notes_per_measure[measure].append((placement, step))

    measure_blocks = []

    for measure in range(max(notes_per_measure.keys())):
        measure_block = absolute_steps_to_measure_block(notes_per_measure[measure], measure)
        measure_blocks.append(measure_block)

    add('NOTES', '\n' + '\n,\n'.join('\n'.join(block) for block in measure_blocks) + '\n')

    with open(result_path, 'w') as f:
        for attribute in stepfile:
            f.write(str(attribute))
            f.write('\n')

    print('Wrote to', result_path)
    

# at 120BPM, a measure has two seconds
SECONDS_PER_MEASURE = 2.0
MEASURE_SUBDIVISION = 192

def absolute_steps_to_measure_block(steps, measure_number):
    steps_per_subdivision = defaultdict(lambda: '00000')

    for absolute_time, step in steps:
        measure_time = absolute_time / SECONDS_PER_MEASURE

        measure_offset = measure_time - measure_number

        assert 0 <= measure_offset < 1

        subdivision = round(measure_offset * MEASURE_SUBDIVISION)

        if subdivision in steps_per_subdivision:
            print('WARNING! Multiple steps in same subdivision', subdivision)

        steps_per_subdivision[subdivision] = step

    return [steps_per_subdivision[i] for i in range(MEASURE_SUBDIVISION)]



def main():
    parser =cli_parser()

    args = parser.parse_args()

    generate_stepfile(args.audio, args.difficulty, args.placement, args.selection, args.out, args.device, args.seed)

def cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('audio', help='Path of audio file to process')
    parser.add_argument('-d', '--difficulty', required=True,
                        help='Target difficulty of chart to generate (S1 up to S25)')
    parser.add_argument('-p', '--placement', required=True,
                        help='Path to the onset/step placement model to use')
    parser.add_argument('-s', '--selection', required=True,
                        help='Path to the step selection model to use')
    parser.add_argument('-o', '--out', required=True,
                        help='Path of .ssc file to generate')
    parser.add_argument('--device', required=False, default='cpu',
                        help='The pytorch device to use')
    parser.add_argument('--seed', required=False, type=int, help='The seed to use. Pass none to use a random seed')

    return parser

if __name__ == "__main__":
    main()
