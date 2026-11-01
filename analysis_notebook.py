import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Pump it Up Dataset Analysis

    This is the notebook where I play with the data as a REPL and do data analysis.

    You need to run `main.py parse_all` first.
    """)
    return


@app.cell(hide_code=True)
def _(glob):
    files = list(glob.glob('data/parsed/*.ssc.bin'))
    return (files,)


@app.cell(hide_code=True)
def _(pickle):
    def get_stepfile(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    return (get_stepfile,)


@app.cell(hide_code=True)
def _(files, get_stepfile):
    all_stepfiles = [get_stepfile(file) for file in files]
    return (all_stepfiles,)


@app.cell(hide_code=True)
def _(all_stepfiles):
    # filtering files with flipped _V or _H prefix

    stepfiles = [stepfile._replace(charts=[chart for chart in stepfile.charts if '_' not in chart.description]) 
                 for stepfile in all_stepfiles]
    return (stepfiles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Chart Count Analysis

    Note that these stats are all in relation to the step _after_ unwanted charts were filtered out (ex: UCS, warp gimmicks, etc.)
    """)
    return


@app.cell(hide_code=True)
def _(np, stepfiles):
    all_chartcounts = np.array([len(stepfile.charts) for stepfile in stepfiles])

    chartcounts = all_chartcounts[all_chartcounts > 0] 
    return (chartcounts,)


@app.cell
def _(stepfiles):
    print('Number of song artists:', len(set(stepfile.info['ARTIST'] for stepfile in stepfiles)))
    print('Number of step authors:', len(set(chart.credit for stepfile in stepfiles for chart in stepfile.charts)))
    print('Song count:', len(stepfiles))
    print('Chart count:', sum(len(stepfile.charts) for stepfile in stepfiles))

    # obtained with the nushell command: rg GENRE | lines | parse "{file}:{genre}" | get genre | histogram
    print('Genres: ORIGINAL, WORLD MUSIC, K-POP, XROSS')
    # technically this doesn't exclude filtered charts but 谁问？


    def _classify_step(stepcode):
        if '2' not in stepcode and '1' not in stepcode:
            assert '3' in stepcode
            return 'RELEASE'
        
        stepcode = stepcode.replace('3', '0')
    
        if '1' not in stepcode:
            return 'HOLD'
        
        if stepcode in {'10000', '01000', '00100', '00010', '00001'}:
            return 'STEP'
    
        return 'JUMP'
        
    _all_step_kinds = [_classify_step(step.stepcode) for stepfile in stepfiles for chart in stepfile.charts for step in chart.steps]
    _step_kinds = [kind for kind in _all_step_kinds if kind != 'RELEASE']

    _total = len(_step_kinds)
    _normal_total = sum(1 for kind in _step_kinds if kind == 'STEP')
    _release_total = sum(1 for kind in _step_kinds if kind == 'RELEASE')
    _jump_total = sum(1 for kind in _step_kinds if kind == 'JUMP')
    _hold_total = sum(1 for kind in _step_kinds if kind == 'HOLD')

    print('Total step count:', _total)

    print('Normal step :', _normal_total / _total * 100, '%') 
    print('Jump Stepcount:', _jump_total / _total * 100, '%') 
    print('Hold Stepcount:',  _hold_total / _total * 100, '%') 

    return


@app.cell(hide_code=True)
def _(chartcounts, mo, np):
    mo.md(f"""
    ### Chart count statistics per stepfile

    - Max: {np.max(chartcounts)}
    - Min: {np.min(chartcounts)}
    - Mean: {np.mean(chartcounts)}
    - Median: {np.median(chartcounts)}
    - Standard deviation {np.std(chartcounts)}
    """)
    return


@app.cell(hide_code=True)
def _(chartcounts, plt):
    make_gray_and_black(hist=plt.hist(chartcounts))
    plt.title('Histogram of chart count of filtered stepfiles')
    return


@app.cell
def _(md_list, mo, stepfiles):
    stepfiles_with_weird_offsets = [stepfile for stepfile in stepfiles if len(set(chart.offset for chart in stepfile.charts)) > 1]

    mo.md(f'''
    ### Stepfiles with more than one offset in charts
    {md_list(['- ' + file.title for file in stepfiles_with_weird_offsets])}
    '''.format())
    return


@app.cell(hide_code=True)
def _(md_list, mo, stepfiles):
    chartless_files = [stepfile.title for stepfile in stepfiles if len(stepfile.charts) == 0]

    mo.md(f"""
    ### Stepfiles whose all charts were filtered out (e.g have gimmicks)

    {md_list(chartless_files)}

    """)
    return


@app.cell(hide_code=True)
def _(md_list, mo, stepfiles):
    least_charted_files = [stepfile.title for stepfile in stepfiles if len(stepfile.charts) == 1]
    least_charted_files

    mo.md(f'''
    ### Least charted files:
    {md_list(least_charted_files)}
    ''')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Chart Difficuly Analysis
    """)
    return


@app.cell(hide_code=True)
def _(stepfiles):
    difficulties_per_stepfile = [[chart.description for chart in stepfile.charts] for stepfile in stepfiles]
    return (difficulties_per_stepfile,)


@app.cell(hide_code=True)
def _(difficulties_per_stepfile):
    def remove_s(diff):
        assert diff.startswith('S')
        return int(diff[1:])

    difficulties = [remove_s(diff) for diffs in difficulties_per_stepfile for diff in diffs]
    return (difficulties,)


@app.cell(hide_code=True)
def _(difficulties, mo, np):
    mo.md(f"""
    ### Chart difficulty stats
    - Mean: {np.mean(difficulties)}
    - Median: {np.median(difficulties)}
    - Standard deviation: {np.std(difficulties)}
    """)
    return


@app.cell(hide_code=True)
def _(difficulties, np):
    diff_range, counts = np.unique(difficulties, return_counts=True)
    return counts, diff_range


@app.cell(hide_code=True)
def _(counts, diff_range, difficulties, np, plt):
    max_diff = max(difficulties) 

    plt.xticks(diff_range)

    make_gray_and_black(bars=plt.bar(diff_range, counts))

    plt.hlines(np.arange(5.0, 60.0, 5.0), xmin=0.0, xmax=max_diff, linestyles='dotted')
    plt.yticks(np.arange(0.0, 60.0, 5.0))

    plt.title('Histogram of difficulties in dataset')

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Chart authorship analysis
    """)
    return


@app.cell(hide_code=True)
def _(np, stepfiles):
    authors_per_stepfile = [[chart.credit for chart in stepfile.charts] for stepfile in stepfiles]
    all_authors = [author for file_authors in authors_per_stepfile for author in file_authors]

    author_values, author_counts = np.unique(all_authors, return_counts=True)
    return author_counts, author_values


@app.cell(hide_code=True)
def _(author_counts, author_values, md_list, mo):
    _values_counts = reversed(sorted(list(zip(author_values, author_counts)), key=lambda x: x[1]))

    mo.md(f'''
    ### Number of charts per author:

    {md_list([f'{author}: {count}' for author, count in _values_counts])}
    ''')
    return


@app.cell(hide_code=True)
def _(author_counts, author_values, plt):

    plt.pie(author_counts, labels=author_values, autopct='%1.1f%%')
    plt.title('Percentages of chart authorships over all charts')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Analyzing the discrepancy between the real song BPM and the presented chart BPM
    """)
    return


@app.cell
def _(stepfiles):
    charts_with_displaybpm = [chart for stepfile in stepfiles for chart in stepfile.charts if 'DISPLAYBPM' in stepfile.info]
    return (charts_with_displaybpm,)


@app.cell
def _(stepfiles):
    displaybpms = [float(stepfile.info['DISPLAYBPM']) for stepfile in stepfiles for chart in stepfile.charts if 'DISPLAYBPM' in stepfile.info]
    return (displaybpms,)


@app.cell
def _(charts_with_displaybpm, displaybpms, np):
    ratios = [displaybpm / chart.avg_bpm for displaybpm, chart in zip(displaybpms, charts_with_displaybpm)]
    ratios = np.array(ratios)
    return (ratios,)


@app.cell
def _(np, ratios):
    print('mean:', np.mean(ratios))
    print('median:', np.median(ratios))
    print('stddev:', np.std(ratios))
    print('min:', np.min(ratios))
    print('max:', np.max(ratios))
    return


@app.cell
def _(np, plt, ratios):
    make_gray_and_black(hist=plt.hist(ratios, np.arange(16)))
    plt.title('Distribution of ratios between real BPM and chart BPM')
    plt.xlabel('Ratio')
    plt.xticks(np.arange(16))
    plt.hlines(np.arange(25.0, 175.0, 25.0), xmin=0.0, xmax=16, linestyles='dotted')
    plt.ylabel('Occurence Count')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Appendix
    """)
    return


@app.cell
def _():
    md_list = lambda xs: '\n'.join(['- ' + f'`{element}`' for element in xs])
    return (md_list,)


@app.function
def make_gray_and_black(*, bars=None, hist=None):
    assert bars is not None or hist is not None

    if hist is not None:
        columns = hist[2]
    else:
        columns = bars

    color1 = 'gray'
    color2 = 'lightgray'
    color = color2
    for column in columns:
        column.set_facecolor(color)
        color = color2 if color == color1 else color1


@app.cell
def _():
    import marimo as mo


    import glob
    import pickle

    import matplotlib.pyplot as plt
    import numpy as np

    return glob, mo, np, pickle, plt


if __name__ == "__main__":
    app.run()
