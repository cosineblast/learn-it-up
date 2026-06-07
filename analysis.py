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


@app.cell
def _():
    return


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
    # Appendix
    """)
    return


@app.cell
def _(stepfiles):
    file = [file for file in stepfiles if file.info['TITLE'] == 'Murdoch vs Otada'][0]


    chart = [chart for chart in file.charts if chart.description == 'S16'][0]

    chart.steps[0:8]
    chart.beat_onset_vectors[1]
    chart.bpms
    return


@app.cell
def _(np):
    stuff = np.random.randn(2,2)
    return (stuff,)


@app.cell
def _(pickle):
    with open('samples/Murdoch_vs_Otada.S9.chart.feat.bin', 'rb') as _f:
        file1 = pickle.load(_f)
    return (file1,)


@app.cell
def _(pickle):
    with open('/tmp/Murdoch_vs_Otada.S9.chart.feat.bin', 'rb') as _f:
        file2 = pickle.load(_f)
    return (file2,)


@app.cell
def _(file1, file2, np):
    np.mean((file1 - file2) ** 2)
    return


@app.cell
def _(np, stuff):

    ix = np.array([True, False, True])

    thing = np.array([[1,2], [3,4]])

    np.where(ix[:, None, None], np.stack([stuff, stuff, thing]), -1)

    return


@app.cell
def _(np):
    np.linspace(np.array([0.0, 10.0, 20.0]), np.array([1.0, 11.0, 21.0]), num=10)
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
