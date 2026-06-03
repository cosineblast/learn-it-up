
# Utilities for parsing and analyzing SSC files

import msdparser
import itertools
import time

from collections import namedtuple
from typing import NamedTuple

STEPFILE_KEYS = {'TITLE', 'ARTIST', 'MUSIC', 'OFFSET', 'BPMS', 'TIMESIGNATURES'}

CHART_KEYS = {'NOTES', 'OFFSET', 'BPMS', 'DESCRIPTION', 'STOPS', 'DELAYS', 'WARPS', 'TIMESIGNATURES', 'FAKES'}

Chart = namedtuple('Chart', CHART_KEYS)

StepFile = namedtuple('StepFile', ['info', 'charts'])

class StepInfo(NamedTuple):
    measure_index: int
    measure_length: int
    offset_in_measure: int
    time_in_beats: float
    time_in_seconds: float
    stepcode: str

def run_chart(steps: list[StepInfo]):

    steps = [step for step in steps if step.stepcode != '00000']

    input("PRESS ENTER FOR FIRST STEP...")

    def show_step(step):
        code = ''.join(['-' if x == '0' else x for x in step.stepcode])
        print('[{:10.4f}]({:10.4f}) {}'.format(step.time_in_beats, step.time_in_seconds, code))

    show_step(steps[0])

    base_music_time = steps[0].time_in_seconds
    base_real_time = time.time()

    for step in steps[1:]:
        to_sleep = (step.time_in_seconds - base_music_time) - (time.time() - base_real_time)
        
        time.sleep(max(0, to_sleep))
        show_step(step)

    
_EPSILON = 1e-6

def compute_steps_absolute_times(offset, bpms, notes) -> list[StepInfo]:
    """
    A stepfile chart is a list of measures, in which each measure is a list of steps.
    A measure is a unit of time consisting of four beats.
    A beat is a unit of time, depending on the BPM (beats per minute).
    The goal of this function is to compute the absolute time of each step (including empty ones) in a chart,
    given information in the original stepfile format.

    Algorithm inspired by DDCL
    """
    segment_durations = compute_segment_durations(bpms)

    result = []
    for measure_num, measure in enumerate(notes):
        measure_len = len(measure)

        for i, code in enumerate(measure):
            beat = measure_num * 4.0 + 4.0 * (float(i) / measure_len)
            beat_abs = compute_beat_absolute_time(offset, bpms, segment_durations, beat)

            info = StepInfo(measure_index=measure_num,
                            measure_length=measure_len,
                            offset_in_measure=i,
                            time_in_beats=beat,
                            time_in_seconds=beat_abs,
                            stepcode=code)

            result.append(info)

    return result

def compute_beat_absolute_time(offset, bpms, segment_durations, beat):
    """
    Computes the absolute time of a beat in seconds.
    """
    segment_index = sum(1 for _ in itertools.takewhile(lambda bpm: beat + _EPSILON > bpm[0], bpms)) - 1
    segment_start, bpm = bpms[segment_index]

    # prefix sum when?
    time_before_this_segment = sum(segment_durations[:segment_index])
    this_segment_spb = bpm_to_spb(bpm)
    time_since_start_of_this_segment = this_segment_spb * (beat - segment_start)

    return time_before_this_segment + time_since_start_of_this_segment - offset 

def compute_segment_durations(bpms):
    """
    Computes the amount of time between BPM changes.

    Example:
    0.0: 60.0
    10.0: 120.0
    20.0: 240.0

    ->

    10 seconds, 5 seconds
    """
    assert len(bpms) > 0

    return [bpm_to_spb(bpm) * (nexttime - time) for (time, bpm), (nexttime, nextbpm) in zip(bpms, bpms[1:])]


def bpm_to_spb(bpm):
    return 60.0 / bpm

def is_applicable_chart(chart: Chart):
    name_ok = ('UCS' not in chart.DESCRIPTION and
        not chart.DESCRIPTION.startswith('D') and
        'D' not in chart.DESCRIPTION and
        'QUEST' not in chart.DESCRIPTION and
        'PRO' not in chart.DESCRIPTION and
        'JUMP' not in chart.DESCRIPTION and
        'HIDDEN' not in chart.DESCRIPTION)

    # TODO: Implement gimmick filtering
    delay_ok = len(chart.DELAYS) == 0
    warp_ok = len(chart.WARPS) == 0
    time_sig_ok = len(chart.TIMESIGNATURES) == 1 and chart.TIMESIGNATURES[0] == (0, 4, 4)
    stop_ok = len(chart.STOPS) == 0
    fake_ok = len(chart.FAKES) == 0

    return name_ok and delay_ok and warp_ok and time_sig_ok


def parse_ssc(filename: str) -> StepFile:
    with open(filename) as f:
        content = msdparser.parse_msd(file=f)
        blocks = split_chart_blocks(content)

    header_rows = blocks[0]
    all_chart_rows = blocks[1:]

    stepfile_info = {row.key:row.value for row in header_rows if row.key in STEPFILE_KEYS}

    charts = [
        {row.key:row.value for row in rows if row.key in CHART_KEYS}
        for rows in all_chart_rows
    ]

    fill_default_chart_values(charts, stepfile_info, filename)

    for chart in charts:
        parse_chart_strings(chart)


    return StepFile(info=stepfile_info, charts=[Chart(**chart) for chart in charts])

def parse_chart_strings(chart: dict):

    chart['OFFSET'] = float(chart['OFFSET'])
    chart['BPMS'] = parse_equals_pair_list(chart['BPMS'])
    chart['STOPS'] = parse_equals_pair_list(chart['STOPS'])
    chart['DELAYS'] = parse_equals_pair_list(chart['DELAYS'])
    chart['TIMESIGNATURES'] = parse_equals_pair_list(chart['TIMESIGNATURES'])
    chart['FAKES'] = parse_equals_pair_list(chart['FAKES'])

    chart['NOTES'] = parse_notes(chart['NOTES'])

    # Songs that have titles attached have INFOBAR TITLE in their descriptions,
    # but they're still valid, so we just remove the INFOBAR TITLE thing
    if chart['DESCRIPTION'].endswith('INFOBAR TITLE'):
        chart['DESCRIPTION'] = chart['DESCRIPTION'][0:-len('INFOBAR TITLE')].strip()


def parse_notes(notes):
    notes = notes.strip()
    notes = notes.split(',')
    notes = [block.split('\n') for block in notes]
    notes = [[row.strip() for row in block] for block in notes]
    notes = [[row for row in block if row != ''] for block in notes]
    return notes
    

def parse_equals_pair_list(string):
    items = [string.strip().split('=') for string in string.split(',') if string != ''] 
    pairs = [tuple([float(value) for value in item]) for item in items] 
    return pairs

def fill_default_chart_values(charts, stepfile_info, filename):
    default_chart_values = {
         'OFFSET': stepfile_info['OFFSET'],
         'BPMS': stepfile_info['BPMS'],
         'DESCRIPTION': 'NONE',
         'STOPS': '',
         'DELAYS': '',
         'WARPS': '',
         'TIMESIGNATURES': stepfile_info['TIMESIGNATURES'],
         'FAKES': ''
    }

    for index, chart in enumerate(charts):
        for key in CHART_KEYS:
            if key not in chart:
                if key in default_chart_values:
                    chart[key] = default_chart_values[key]
                else:
                    raise Exception('The chart {} in the file {} does not have the required key {}'.format(index, filename, key))

def split_chart_blocks(content):
    return [list(block)
            for is_notedata_row, block in itertools.groupby(content, lambda row: row.key == 'NOTEDATA')
            if not is_notedata_row]
 
