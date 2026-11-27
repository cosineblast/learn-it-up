import hmac
import os
import re
import tempfile
import tomllib

from flask import Flask, render_template, request, send_file
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

import gen

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ONSET_MODEL_PATH'] = 'website_onset_model.pth'
app.config['SELECTION_MODEL_PATH'] = 'website_selection_model.pth'
app.config['THRESHOLD_FILE_PATH'] = 'thresholds.json'

if os.path.isfile('config.toml'):
    app.config.from_file('config.toml', load=tomllib.load, text=False)
app.config.from_prefixed_env()

assert 'USAGE_KEY' in app.config, (
    'Error: The USAGE_KEY for deployment is not defined.'
    '(Can be set with FLASK_USAGE_KEY environment variable)')

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        return process_form()
    else:
        return render_template("form.html")


VALID_DIFFICULTIES = {str(x) for x in range(2, 24)}

ALLOWED_EXTENSIONS = {'.mp3', '.ogg'}

def is_filename_allowed(filename):
    return os.path.splitext(filename)[1] in ALLOWED_EXTENSIONS

class Error:
    def __init__(self, message):
        self.message = message
    

def parse_form() -> Error | tuple[int, int, str, FileStorage]:
    if 'difficulty' not in request.form:
        return Error("Missing difficulty")

    difficulty = request.form["difficulty"]

    if difficulty not in VALID_DIFFICULTIES:
        return Error("Invalid difficulty")

    if 'seed' not in request.form:
        return Error("Missing seed")
    seed = request.form["seed"]

    if not re.fullmatch(r"\d{1,4}", seed):
        return Error("Invalid seed")

    if 'key' not in request.form:
        return Error('Missing usage key')
    key = request.form['key']

    if 'file' not in request.files:
        return Error('Missing music file')

    form_file = request.files['file']

    if not form_file or form_file.filename == '':
        return Error('No selected file')

    if not is_filename_allowed(form_file.filename):
        return Error('File extension not allowed')

    return int(difficulty), int(seed), key, form_file



def process_form():
    result  = parse_form()

    if isinstance(result, Error):
        return render_template("form.html", error=result.message)

    difficulty, seed, key, form_file = result

    if not hmac.compare_digest(app.config['USAGE_KEY'], key):
        return render_template("form.html", error='bad key :/')

    with tempfile.NamedTemporaryFile() as input_file:
        form_file.save(input_file)

        # we need to not use a with clause here because send_file closes the file
        out_file = tempfile.NamedTemporaryFile()

        try:
            gen.generate_stepfile(
                audio_path=input_file.name,
                audio_name=form_file.filename,
                difficulty=difficulty,
                onset_model_path=app.config['ONSET_MODEL_PATH'],
                selection_model_path=app.config['SELECTION_MODEL_PATH'],
                result_path=out_file.name,
                device='cpu',
                seed=seed,
                cnn_onset=False,
                threshold_file=app.config['THRESHOLD_FILE_PATH']
            )

            out_file.seek(0)

            download_filename  = os.path.splitext(secure_filename(str(form_file.filename)))[0] + '.ssc'
        except Exception:
            out_file.close()
            raise


        return send_file(out_file, mimetype='application/octet-stream', as_attachment=True, download_name=download_filename)
