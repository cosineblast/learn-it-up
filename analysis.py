import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Pump it Up Dataset Analysis

    This is the notebook where I play with the data as a REPL and do data analysis.
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
    stepfiles = [get_stepfile(file) for file in files]
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
    all_chartcounts = np.array([len(stepfile.charts) / 2 for stepfile in stepfiles])
    # / 2 because of mirroring

    chartcounts = all_chartcounts[all_chartcounts > 0] 
    return (chartcounts,)


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
    p = plt.hist(chartcounts)
    plt.title('Histogram of chart count of filtered stepfiles')
    return


@app.cell(hide_code=True)
def _(mo, stepfiles):
    stepfiles_with_weird_offsets = [stepfile for stepfile in stepfiles if len(set(chart.offset for chart in stepfile.charts)) > 1]

    mo.md(f'''
    ### Stepfiles with more than one offset in charts
    {('\n'.join(['- ' + file.info['TITLE'] for file in stepfiles_with_weird_offsets]))}
    ''')
    return


@app.cell(hide_code=True)
def _(md_list, mo, stepfiles):
    chartless_files = [stepfile.info['TITLE'] for stepfile in stepfiles if len(stepfile.charts) == 0]

    mo.md(f"""
    ### Stepfiles whose all charts were filtered out (e.g have gimmicks)

    {md_list(chartless_files)}

    """)
    return


@app.cell(hide_code=True)
def _(md_list, mo, stepfiles):
    least_charted_files = [stepfile.info['TITLE'] for stepfile in stepfiles if len(stepfile.charts) / 2 == 1]
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
    difficulties_per_stepfile = [[chart.description for chart in stepfile.charts if not chart.description.endswith('_V') 
                     and not chart.description.endswith('_H')] for stepfile in stepfiles]
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

    _columns = plt.bar(diff_range, counts)

    plt.hlines(np.arange(5.0, 60.0, 5.0), xmin=0.0, xmax=max_diff, linestyles='dotted')
    plt.yticks(np.arange(0.0, 60.0, 5.0))
    _color1 = 'gray'
    _color2 = 'lightgray'
    _color = _color2
    for column in _columns:
        column.set_facecolor(_color)
        _color = _color2 if _color == _color1 else _color1

    plt.title('Histogram of difficulties in dataset')

    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ### Chart authorship analysis

    **TODO**: Include CREDIT attribute in refined charts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Appendix
    """)
    return


@app.cell
def _(stepfiles):
    festival_of_death_moon = [file for file in stepfiles if file.info['TITLE'] == 'Festival of Death Moon'][0]
    chart = festival_of_death_moon.charts[2]
    return (chart,)


@app.cell
def _(chart):
    chart.measure_start_end_times
    return


@app.cell
def _():
    md_list = lambda xs: '\n'.join(['- ' + f'`{element}`' for element in xs])
    return (md_list,)


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
