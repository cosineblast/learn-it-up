
# A ssc file is a sequence of a bunch of key value pairs
# of the form #KEY:VALUE;
# 
# The first key of a file is VERSION (we expect 0.81)
# then, the file will contain several attributes about all the charts contained in the file
# such as song name, song author, genre, expected bpm, audio file, background video file.
#
# After the overall information about the file, comes the information about each chart.
# Each chart starts with the key NOTEDATA and an empty value, followed by the chart data.
# The most important ones are OFFSET, BPMS and NOTES.
#
# - NOTES is a newline separated
# list of numbers representing the steps at each position. The notes value
# also has lines containing a single comma separating measures [i think].
#
# - OFFSET is a number that tells how much time (in seconds [i think]) should the music audio be displaced
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
# broken python dependency fs, so we use this stream-based parser instead.


import msdparser
import itertools
from collections import namedtuple

STEPFILE_KEYS = {'TITLE', 'ARTIST'}

CHART_KEYS = {'NOTES', 'OFFSET', 'BPMS', 'DESCRIPTION', 'STOPS', 'DELAYS', 'WARPS', 'TIMESIGNATURES', 'FAKES'}

Chart = namedtuple('Chart', CHART_KEYS)

StepFile = namedtuple('StepFile', ['info', 'charts'])

def main():
    # filename = 'Beethoven_Virus.ssc'
    # filename = 'BRAIN_POWER.ssc'
    filename = 'Switronic.ssc'

    print('Reading file {}'.format(filename))

    stepfile = parse_ssc(filename)

    charts = [chart for chart in stepfile.charts if is_applicable_chart(chart)]

    print('Available charts:', [chart.DESCRIPTION for chart in charts])

        

def is_applicable_chart(chart: Chart):
    name_ok = ('UCS' not in chart.DESCRIPTION and
        'D' not in chart.DESCRIPTION and
        'QUEST' not in chart.DESCRIPTION and
        'PRO' not in chart.DESCRIPTION and
        'JUMP' not in chart.DESCRIPTION)

    # TODO: Implement gimmick filtering
    delay_ok = len(chart.DELAYS) == 0
    warp_ok = len(chart.WARPS) == 0
    time_sig_ok = len(chart.TIMESIGNATURES) <= 1
    fake_ok = len(chart.FAKES) == 0

    return name_ok and delay_ok and warp_ok and time_sig_ok


def parse_ssc(filename: str):
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

    fill_default_chart_values(charts)

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

    # TODO: handle charts with 'INFOBAR TITLE' in the name like switronic



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

def fill_default_chart_values(charts):
    default_chart_values = {
         'DESCRIPTION': 'NONE',
         'STOPS': [],
         'DELAYS': [],
         'WARPS': [],
         'TIMESIGNATURES': [],
         'FAKES': []
    }

    for chart in charts:
        for key in CHART_KEYS:
            if key not in chart:
                if key in default_chart_values:
                    chart[key] = default_chart_values[key]
                else:
                    raise 'A chart in the file {} does not have the required key {}'.format(filename, key)

def split_chart_blocks(content):
    return [list(block)
            for is_notedata_row, block in itertools.groupby(content, lambda row: row.key == 'NOTEDATA')
            if not is_notedata_row]
        
if __name__ == '__main__':
    main()

