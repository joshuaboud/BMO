#!/usr/bin/env python3

import torch
import numpy as np
from typing import List, Tuple
from flask import Flask, redirect, url_for, request, Response, render_template
import argparse
import os
from pathlib import Path

from TTS.api import TTS
# from TTS.tts.configs.bark_config import BarkConfig
# from TTS.tts.models.bark import Bark

# model_dir = os.environ['TTS_MODEL_DIR']
# if not Path(model_dir).is_dir():
#     raise ValueError(f"Error: {model_dir}: not a directory")

# config = BarkConfig(

# )
# model = Bark.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)

# # cloning a speaker.
# # It assumes that you have a speaker file in `bark_voices/speaker_n/speaker.wav` or `bark_voices/speaker_n/speaker.npz`
# output_dict = model.synthesize(text, config, speaker_id="bmo_if_i_was_grown", voice_dirs="clone_srcs/")


def init_tts() -> TTS:
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'TTS device: {device} (cpu|cuda)')
    # Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/bark", gpu=True, progress_bar=False).to(device)
    return tts


def pre_process_text(text: str) -> str:
    new_text = text.upper()
    DICTIONARY: List[Tuple[str, str]] = [
        ('BMO', 'BEEMO'),
    ]
    for old, new in DICTIONARY:
        new_text = new_text.replace(old, new)
    print(f"orig: '{text}'")
    print(f"processed: '{new_text}'")
    return new_text


def synthesize(text: str) -> np.ndarray:
    text = pre_process_text(text)
    print("Generating audio from text...")
    audio = np.array(
        tts.tts(
            text=text,
            voice_dir='clone_srcs/',
            speaker='bmo_if_i_was_grown',
            # speaker_wav=["clone_srcs/bmo_best_of.wav", "clone_srcs/bmo_if_i_was_grown.wav", "clone_srcs/bmo_i_am_a_little_living_boy.wav"],
            # language="en"
            ), dtype='float32')
    return audio


tts = init_tts()

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form["text"]
        pcm = synthesize(input_text)
        pcm_bytes = pcm.tobytes()
        return Response(pcm_bytes, content_type=f'audio/pcm;rate={tts.synthesizer.output_sample_rate};encoding=float;bits=32')
    return render_template("index.html")


@app.route('/api/v1/synthesize', methods=['POST'])
def api_synthesize():
   MAX_CONTENT_LENGTH = 2**20
   if request.method == 'POST':
       if request.content_type != 'application/json':
           return f"Invalid content type: '{request.content_type}'. Expected 'application/json'", 400
       if request.content_length > MAX_CONTENT_LENGTH:
           return f"Too many bytes!", 400
       input_json = request.get_json()
       if 'text' not in input_json:
           return f"Object missing 'text' property!", 400
       input_text = input_json['text']
       pcm = synthesize(input_text)
       pcm_bytes = pcm.tobytes()
       return Response(pcm_bytes, content_type=f'audio/pcm;rate={tts.synthesizer.output_sample_rate};encoding=float;bits=32')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-port', '--port', type=int, required=False, default=80, metavar='PORT', help="Server port number (default 80)")
    args = parser.parse_args()
    app.run(debug=True, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
