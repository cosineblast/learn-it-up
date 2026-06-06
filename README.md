
# learn it up

This project contains the (work in progress) code, and reference material for [Learn it Up](https://cosineblast.github.io/learn-it-up), a project that tries
to adapt the LSTM and Transformer architectures for Pump it Up chart generation.

## Running

The dependencies for this project are listed in `pyproject.toml`.

Use `python3 main.py --help` to get a list of available operations.

This project is compatible with `uv`.

## File structure

This project assumes that the songs packs are in `./data/songs`, in usual `packdir/songdir` fashion. Step files must be in SSC.

The processing pipeline consists of:

- Parse SSC files
- Extract relevant keys from SSC (e.g DESCRIPTION, OFFSET, BPMS, NOTES) (`Chart` and `StepFile`) in code.
- Introduce mirroring chart variations (vertical only for S17 and below)
- Extract absolute time information for all charts
- Save processed ssc files to disk, at path `./data/parsed/${PACK_NAME}___${SONG_TITLE}.ready.ssc.bin` (I wanted to use JSON but serializing and deserializing python classes to json sucks in general and i didnt want to add another dependency)

