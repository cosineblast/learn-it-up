# Utilities for parsing and analyzing SSC files


# A ssc file is a sequence of a bunch of key value pairs
# of the form #KEY:VALUE;
#
# The first key of a file is VERSION (we expect 0.81)
# then, the file will contain several attributes about all the charts contained in the file
# such as song name, song author, genre, expected bpm, audio file, background video file.
#
# After the overall information about the file, comes the information about each chart.
# Each chart starts with the key NOTEDATA and an empty value, followed by the chart data.
# The most important attributes are OFFSET, BPMS (Beats Per Minute) and NOTES.
#
# - NOTES is a newline separated
# list of numbers representing the steps at each position. The notes value
# also has lines containing a single comma separating measures.
#
# - OFFSET is a number that tells how much time (in seconds) should the music audio be displaced
# to align with the chart.
#
# - BPMS is a list of pairs informing the moments of BPM changes in the songs. The list
# items are separated by commas, and the pair are of the form TIME=BPM.
#
# For visual gimmicks, the main keys are STOPS, DELAY, WARP, TIMESIGNATURES,
# TICKCOUNTS, COMBOS, SPEEDS, SCROLLS and FAKES.
# Out of these, SPEEDS and SCROLLS do not affect note timing.
#
# LASTSECONDHINT is used by the game to figure out the song length in seconds.

# There is a project that implements parsing ssc files more conveniently, but it uses a
# broken python dependency fs, so we use this stream-based msdparser instead.

import msdparser
import itertools
import time
import typing
import pickle

from pathlib import Path
from collections import namedtuple
from typing import NamedTuple

STEPFILE_KEYS = {"TITLE", "ARTIST", "MUSIC", "OFFSET", "BPMS", "TIMESIGNATURES"}

CHART_KEYS = {
    "NOTES",
    "OFFSET",
    "BPMS",
    "DESCRIPTION",
    "STOPS",
    "DELAYS",
    "WARPS",
    "TIMESIGNATURES",
    "FAKES",
}

# Some information about a SSC file chart, in memory. 
class Chart(NamedTuple): 
    NOTES: list[list[str]]
    OFFSET: float
    BPMS: list[tuple[float, float]]
    DESCRIPTION: str
    TIMESIGNATURES: list[tuple[float, float, float]]
    STOPS: list[tuple[float, ...]]
    DELAYS: list[tuple[float, ...]]
    WARPS: list[tuple[float, ...]]
    FAKES: list[tuple[float, ...]]

# Some information about a SSC file, in memory. 
class StepFile(NamedTuple):
    info: dict[str, str]
    charts: list[Chart]

# All time-related information about a step in a chart
class StepInfo(NamedTuple):
    measure_index: int
    measure_length: int
    offset_in_measure: int
    time_in_beats: float
    time_in_seconds: float
    stepcode: str

# A chart informatino, includes absolute time information
class RefinedChart(NamedTuple):
    steps: list[StepInfo]
    offset: float
    bpms: list[tuple[float, float]]
    description: str
    measure_start_end_times: list[tuple[float, float]]

# absolute time information in charts and absolute music file path
class RefinedStepFile(NamedTuple):
    info: dict[str, str]
    charts: list[RefinedChart]

def load_ssc(filename: str) -> StepFile:
    """Load a .ssc file from disk into a StepFile object."""

    with open(filename) as f:
        content = msdparser.parse_msd(file=f)
        blocks = _split_chart_blocks(content)

    header_rows = blocks[0]
    all_chart_rows = blocks[1:]

    stepfile_info = {
        row.key: row.value for row in header_rows if row.key in STEPFILE_KEYS
    }

    charts = [
        {row.key: row.value for row in rows if row.key in CHART_KEYS}
        for rows in all_chart_rows
    ]

    _fill_default_chart_values(charts, stepfile_info, filename)

    charts = [_parse_chart_strings(chart) for chart in charts]

    return StepFile(info=stepfile_info, charts=charts)

def _split_chart_blocks(content):
    return [
        list(block)
        for is_notedata_row, block in itertools.groupby(
            content, lambda row: row.key == "NOTEDATA"
        )
        if not is_notedata_row
    ]

def _fill_default_chart_values(charts, stepfile_info, filename):
    """Fills a dictionary with default chart attribute keys as necessary."""

    default_chart_values = {
        "OFFSET": stepfile_info["OFFSET"],
        "BPMS": stepfile_info["BPMS"],
        "DESCRIPTION": "NONE",
        "STOPS": "",
        "DELAYS": "",
        "WARPS": "",
        "TIMESIGNATURES": stepfile_info["TIMESIGNATURES"],
        "FAKES": "",
    }

    for index, chart in enumerate(charts):
        for key in CHART_KEYS:
            if key not in chart:
                if key in default_chart_values:
                    chart[key] = default_chart_values[key]
                else:
                    raise Exception(
                        "The chart {} in the file {} does not have the required key {}".format(
                            index, filename, key
                        )
                    )

def _parse_chart_strings(chart: dict) -> Chart:
    """Parses the strings of a dictionary with chart attribute keys into a chart object."""

    return Chart(
        OFFSET = float(chart["OFFSET"]),
        BPMS = _parse_equals_pair_list(chart["BPMS"]),
        STOPS = _parse_equals_pair_list(chart["STOPS"]),
        DELAYS = _parse_equals_pair_list(chart["DELAYS"]),
        TIMESIGNATURES = _parse_equals_pair_list(chart["TIMESIGNATURES"]),
        FAKES = _parse_equals_pair_list(chart["FAKES"]),
        WARPS = _parse_equals_pair_list(chart["WARPS"]),
        NOTES = _parse_notes(chart["NOTES"]),

        # Songs that have titles attached have INFOBAR TITLE in their descriptions,
        # but they're still valid, so we just remove the INFOBAR TITLE thing
        DESCRIPTION = (
            chart["DESCRIPTION"][0 : -len("INFOBAR TITLE")].strip() if
            chart["DESCRIPTION"].endswith("INFOBAR TITLE") else
            chart["DESCRIPTION"]
        )
    )

def _parse_notes(notes):
    """Converts a ssc notes string (measures sep by ',' lines sep by newline)
    into a nested list of stepcodes"""
    notes = notes.strip()
    notes = notes.split(",")
    notes = [block.split("\n") for block in notes]
    notes = [[row.strip() for row in block] for block in notes]
    notes = [[row for row in block if row != ""] for block in notes]
    return notes


def _parse_equals_pair_list(string):
    """Converts a string of the form '1=2,3=4' into a list of tuples [(1,2), (3,4)]"""
    items = [string.strip().split("=") for string in string.split(",") if string != ""]
    pairs = [tuple([float(value) for value in item]) for item in items]
    return pairs

def is_applicable_chart(chart: Chart, stepfile_name=None):
    """Determines if a chart should be skipped or kept."""

    name_ok = (
        "UCS" not in chart.DESCRIPTION
        and not chart.DESCRIPTION.startswith("D")
        and "D" not in chart.DESCRIPTION
        and "QUEST" not in chart.DESCRIPTION
        and "PRO" not in chart.DESCRIPTION
        and "JUMP" not in chart.DESCRIPTION
        and "HIDDEN" not in chart.DESCRIPTION
    )

    # TODO: Implement gimmick filtering
    delay_ok = len(chart.DELAYS) == 0
    warp_ok = len(chart.WARPS) == 0
    time_sig_ok = len(chart.TIMESIGNATURES) == 1 and chart.TIMESIGNATURES[0] == (
        0,
        4,
        4,
    )
    stop_ok = len(chart.STOPS) == 0
    fake_ok = len(chart.FAKES) == 0

    notes_ok = False

    def check_notes_ok():
        # 0 is idle, 1 is hit, 2 is start hold, 3 is end hold
        # xyzXYZ are for coop
        expected_notes = set("0123xyzXYZ")

        for measure in chart.NOTES:
            for step in measure:
                if any((value not in expected_notes for value in step)):
                    if stepfile_name is not None:
                        print(
                            "WARNING: Chart {} of {} has invalid step {}".format(
                                chart.DESCRIPTION, stepfile_name, step
                            )
                        )
                    return False

        return True

    return check_notes_ok() and name_ok and delay_ok and warp_ok and time_sig_ok


def refine_stepfile(stepfile: StepFile, original_path) -> RefinedStepFile:
    """Refines a stepfile to add absolute time information and absolute music path."""

    absolute_music_path = (Path(original_path).parent / Path(stepfile.info["MUSIC"]).name).resolve()

    return RefinedStepFile(
       info={**stepfile.info, 'MUSIC':str(absolute_music_path)},
       charts=[refine_chart(chart) for chart in stepfile.charts]
    )

def refine_chart(chart: Chart) -> RefinedChart:
    """Refines a chart to add absolute step time information to it."""

    return RefinedChart(
        steps=_compute_steps_absolute_times(chart.OFFSET, chart.BPMS, chart.NOTES),
        offset=chart.OFFSET,
        bpms= chart.BPMS,
        description= chart.DESCRIPTION,
        measure_start_end_times=_compute_measure_times(chart.OFFSET, chart.BPMS, chart.NOTES)
    )

def _compute_steps_absolute_times(offset, bpms, notes) -> list[StepInfo]:
    """
    A stepfile chart is a list of measures, in which each measure is a list of steps.
    A measure is a unit of time consisting of four beats.
    A beat is a unit of time, depending on the BPM (beats per minute).
    The goal of this function is to compute the absolute time of each step (including empty ones) in a chart,
    given information in the original stepfile format.

    Algorithm inspired by DDC
    """
    segment_durations = _compute_segment_durations(bpms)

    result = []
    for measure_num, measure in enumerate(notes):
        measure_len = len(measure)

        for i, code in enumerate(measure):
            beat = measure_num * 4.0 + 4.0 * (float(i) / measure_len)
            beat_abs = _compute_beat_absolute_time(offset, bpms, segment_durations, beat)

            info = StepInfo(
                measure_index=measure_num,
                measure_length=measure_len,
                offset_in_measure=i,
                time_in_beats=beat,
                time_in_seconds=beat_abs,
                stepcode=code,
            )

            result.append(info)

    return result

def _compute_segment_durations(bpms):
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

    return [
        _bpm_to_spb(bpm) * (nexttime - time)
        for (time, bpm), (nexttime, nextbpm) in zip(bpms, bpms[1:])
    ]

def _bpm_to_spb(bpm):
    """Converts from Beats Per Minute to Seconds Per Beat"""

    return 60.0 / bpm


def _compute_beat_absolute_time(offset, bpms, segment_durations, beat):
    """Computes the absolute time of a beat in seconds"""
    segment_index = (
        sum(1 for _ in itertools.takewhile(lambda bpm: beat + _EPSILON > bpm[0], bpms))
        - 1
    )
    segment_start, bpm = bpms[segment_index]

    # prefix sum when?
    time_before_this_segment = sum(segment_durations[:segment_index])
    this_segment_spb = _bpm_to_spb(bpm)
    time_since_start_of_this_segment = this_segment_spb * (beat - segment_start)

    # why - offset you may ask?
    # well, offsets are stored in negative numbers, an audio delay of 1 second is stored as -1,
    # so we have to subtract the offset.
    return time_before_this_segment + time_since_start_of_this_segment - offset


def _compute_measure_times(offset, bpms, notes) -> list[tuple[float, float]]:
    durations = _compute_segment_durations(bpms)

    result = [(_compute_beat_absolute_time(offset, bpms, durations, i * 4),
               _compute_beat_absolute_time(offset, bpms, durations, (i+1) * 4))
         for (i, _measure) in enumerate(notes)]

    return result

def run_chart(chart: RefinedChart):
    """Simulates in real time, the notes of a refined chart."""
        
    steps = [step for step in chart.steps if step.stepcode != "00000"]

    input("PRESS ENTER FOR FIRST STEP...")

    def show_step(step):
        code = "".join(["-" if x == "0" else x for x in step.stepcode])
        print(
            "[{:10.4f}]({:10.4f}) {}".format(
                step.time_in_beats, step.time_in_seconds, code
            )
        )

    show_step(steps[0])

    base_music_time = steps[0].time_in_seconds
    base_real_time = time.time()

    for step in steps[1:]:
        to_sleep = (step.time_in_seconds - base_music_time) - (
            time.time() - base_real_time
        )

        time.sleep(max(0, to_sleep))
        show_step(step)


_EPSILON = 1e-6











def dump_refined_stepfile(stepfile: RefinedStepFile, file):
    pickle.dump(stepfile, file)

def load_refined_stepfile(file) -> RefinedStepFile:
    return pickle.load(file)

