
# Learn it Up

This project contains the (work in progress) code, and reference material for [Learn it Up](https://cosineblast.github.io/learn-it-up), a project that tries
to apply Machine Learning for Pump it Up chart generation.

It currently implements an adaptation of the [DanceDanceConvolution](https://arxiv.org/abs/1703.06891) architecture with a CNN placement and LSTM placement models.

Developed and tested with python 3.11 and pytorch 2.6.0.

## Running

The dependencies for this project are listed in the file `pyproject.toml`.
This project is compatible with `uv`.

This project assumes that the song packs are in `./data/songs`, in usual `packdir/songdir` fashion. Step files must be in SSC.
Chart `DESCRIPTION` attributes are assumed to represent the difficulties (`S1`...`S25`)

Right now, this project does not work with stepfiles contaning StepP1 special notes such as vanishes, and charts including stop and warp gimmicks are ignored.

Use `python3 main.py --help` to get a list of available operations.

In order to train and run models, you will need to run the following operations:

- `python3 main.py parse_all` to parse all SSC files into `data/parsed`.
- `python3 main.py extract_all` to extract features from all audio files into `data/features`.
- `python3 main.py partition` to create a `data/partitions.json` file containing the partitions of stepfiles that will be used for training, testing and validation.

- In the notebook `python3 -m marimo edit onset_notebook.py` you will be able to train the step placement model.
- Do the same with `python3 -m marimo edit selection_notebook.py` for the step selection model.

Once you have the models trained, use `python3 gen.py` to generate your files.

Example:

```bash
python3 gen.py samples/OshamaScramble.mp3 \
  --difficulty S16 \
  --selection selection.pth \
  --placement onset.pth \
  -o samples/oshama_scramble.ssc 
```

For more information, run `python3 gen.py --help`

## Pipeline

The processing pipeline consists of:

- Parse SSC files
- Extract relevant keys from SSC (e.g DESCRIPTION, OFFSET, BPMS, NOTES) (`Chart` and `StepFile`) in code.
- Introduce mirroring chart variations (vertical only for S17 and below)
- Extract absolute time information for all charts
- Save processed ssc files to disk, at path `./data/parsed/${PACK_NAME}___${SONG_TITLE}.ready.ssc.bin` (I wanted to use JSON but serializing and deserializing python classes to json sucks in general and i didnt want to add another dependency)

## Notebooks

The dataset analysis marimo notebook is in `analysis.py`, that can be run with `marimo edit analysis.py` or `uv marimo edit analysis.py`
