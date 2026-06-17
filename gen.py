
import argparse

import re
import sys
import itertools
import time

from collections import defaultdict

FRAMES_PER_SECOND=100

def generate_stepfile(audio_path,
                      difficulty_str,
                      onset_model_path,
                      selection_model_path,
                      result_path):

    # TODO: pass audio start and end offsets

    if not re.match(r'S\d+', difficulty_str):
        print('error: Invalid difficulty! Expecting diffculty of the form S1...S25')
        sys.exit(1)


    difficulty = int(difficulty_str[1:])

    if not (1 <= difficulty <= 25):
        print('error: Difficulty must be in 1..25 range')
        sys.exit(1)

    print('Loading libraries... (let the bass kick)')
    import torch
    import torch.nn.functional as F
    import audio_util
    import loading
    import scipy
    import models
    import numpy as np

    print("Extracting audio features of {}... (JOOOOOOOO)".format(audio_path))
    loader = audio_util.AudioFeatureLoader(use_tqdm=True)
    features = loader.load(audio_path)
            

    device = 'cpu'

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
    threshold = 0.2

    window = np.hamming(5)
    onsets = np.convolve(scores, window, mode='same')
    maxima = scipy.signal.argrelextrema(onsets, np.greater_equal, order=1)[0]
    maxima = set(maxima)

    placements = sorted(onset / FRAMES_PER_SECOND for onset in maxima if scores[onset] > threshold)

    # now run selection model
    print('Running selection model... (AAE-O-A-A-U-U-A Eeeeeeee)')

    steps = []

    previous_placement = None
    previous_prediction = None
    previous_state = None
    for placement in placements:

        delta = (np.array([0.0, 1.0])
            if previous_placement is None
            else np.array([placement - previous_placement, 0.0]))
        
        x = (np.zeros((5, 4))
            if previous_prediction is None 
            else loading.stepcode_to_bag_tensor(loading.index_to_stepcode(previous_prediction)))

        x = x.reshape((1, 1, 5, 4))
        delta = delta.reshape((1, 1, 2))

        with torch.no_grad():
            #selection_model.eval()
            
            x = torch.tensor(x).float().to(device)
            delta = torch.tensor(delta).float().to(device)

            log_scores, state = selection_model(x, delta, initial_state=previous_state, return_state=True)

            log_scores = torch.flatten(log_scores,start_dim=0, end_dim=1)
            predictions = torch.argmax(F.softmax(log_scores, dim=1), dim=1).detach().cpu().numpy()

        steps.append(loading.index_to_stepcode(predictions[0]))

        previous_placement = placement
        previous_state = state
        previous_prediction = predictions[0]

    build_stepfile(placements, steps, audio_path, result_path, difficulty_str)

def build_stepfile(placements, steps, audio_path, result_path, difficulty_str):
    import msdparser

    stepfile=[]

    def add(key, value):
        stepfile.append(msdparser.MSDParameter([key, value]))

    add('VERSION'         , '0.81')
    add('TITLE'           , 'PROTOTYPE')
    add('SUBTITLE'        , '')
    add('ARTIST'          , 'UNKNOWN')
    #add('GENRE'           , 'WORLD MUSIC')
    #add('ORIGIN'          , '')
    #add('CREDIT'          , 'DECRYPT')
    #add('BANNER'          , '16A9_B.png')
    #add('BACKGROUND'      , '16A9.png')
    #add('PREVIEWVID'      , '16A9_P.mpg')
    #add('CDTITLE'         , '')
    add('MUSIC'           , audio_path)
    #add('OFFSET'          , '0.0')
    add('SELECTABLE'      , 'YES')
    #add('SONGTYPE'        , 'ARCADE')
    #add('SONGCATEGORY'    , 'ORIGINAL')
    add('VOLUME'          , '100')
    add('DISPLAYBPM'      , '120.000000')
    add('BPMS'            , '0.000=120.000')
    add('TIMESIGNATURES'  , '0.000=4=4')
    add('TICKCOUNTS'      , '0.000=4')
    add('COMBOS'          , '0.000=1')
    add('SPEEDS'          , '0.000=1.000=0.000=0')
    add('SCROLLS'         , '0.000=1.000')
    #add('LABELS'          , '0.000=Song Start')
    #add('LASTSECONDHINT'  , '99.264000')

    add('NOTEDATA','')
    add('STEPSTYPE','pump-single')
    add('DESCRIPTION', difficulty_str)
    add('DIFFICULTY','Edit')
    add('METER','1')
    #add('RADARVALUES','-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000')
    #add('CREDIT','IDK')
    add('OFFSET','0.000000')
    #add('BPMS','0.000000=120.000000')
    add('STOPS','')
    add('DELAYS','')
    add('WARPS','')
    add('TIMESIGNATURES','0.000000=4=4')
    add('TICKCOUNTS','0.000000=2')
    add('COMBOS','0.000000=1')
    add('SPEEDS','0.000000=1.000000=0.000000=0')
    add('SCROLLS','0.000000=1.000000')
    add('FAKES','')
    #add('LABELS','0.000000=Song Start')
    #add('LASTSECONDHINT','104.376000')
    #add('CHARTSTYLE', 'ACTIVE');


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

    # simulating...

    from playsound3 import playsound

    def show_step(step, placement):
        code = "".join(["-" if x == "0" else x for x in step])
        print(
            "[{:10.4f}] {}".format(
                placement, code
            )
        )

    playsound(audio_path, block=False, backend='ffplay')
    time.sleep(placements[0])
        
    show_step(steps[0], placements[0])

    base_music_time = placements[0]
    base_real_time = time.time()

    for step, placement in zip(steps[1:], placements[1:]):
        to_sleep = (placement - base_music_time) - (
            time.time() - base_real_time
        )

        time.sleep(max(0, to_sleep))
        show_step(step, placement)

    time.sleep(2)

    

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

    generate_stepfile(args.audio, args.difficulty, args.placement, args.selection, args.out)

def cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('audio')
    parser.add_argument('-d', '--difficulty', required=True)
    parser.add_argument('-p', '--placement', required=True)
    parser.add_argument('-s', '--selection', required=True)
    parser.add_argument('-o', '--out', required=True)

    return parser

if __name__ == "__main__":
    main()
