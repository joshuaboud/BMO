#!/usr/bin/env python3

import argparse
import sounddevice as sd
import requests
import numpy as np
import soundfile as sf
from typing import Tuple


def parse_content_type(content_type: str) -> dict:
    content_type_base, *params = content_type.split(';')
    params = dict(map(lambda x: x.split('='), params))
    if content_type_base == 'audio/pcm':
        return {
            'samplerate': int(params['rate']),
            'dtype': np.dtype(f"{params['encoding']}{params['bits']}") if 'encoding' in params and 'bits' in params else np.int16,
        }
    raise ValueError(f'Invalid content type: {content_type}')


def get_pcm(text: str, host: str, port: int) -> Tuple[np.ndarray, dict]:
    print("Requesting speech synthesis...")
    r = requests.post(f'http://{host}:{port}/api/v1/synthesize', json={'text': text})
    r.raise_for_status()
    audio_info = parse_content_type(r.headers['Content-Type'])
    print(audio_info)
    pcm = np.frombuffer(r.content, dtype=audio_info['dtype'])
    return pcm, audio_info


def say(text: str, host: str, port: int) -> None:
    pcm, audio_info = get_pcm(text, host, port)
    print("Playing audio...")
    audio_stream = sd.OutputStream(samplerate=audio_info['samplerate'], channels=1, blocksize=2048, dtype=audio_info['dtype'])
    audio_stream.start()
    audio_stream.write(pcm)
    audio_stream.stop()


def save_wav(text: str, host: str, port: int, file_path: str) -> None:
    pcm, audio_info = get_pcm(text, host, port)
    with sf.SoundFile(file_path, mode='w', samplerate=audio_info['samplerate'], channels=1, subtype='FLOAT', format='WAV') as wv:
        wv.write(pcm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', type=str, required=False, default='127.0.0.1', metavar='HOST', help="Server host")
    parser.add_argument('-p', '--port', type=int, required=True, metavar='PORT', help="Server port number")
    parser.add_argument('-w', '--output-wav', type=str, required=False, default=None, dest='output_wav_path', metavar='PATH.wav', help="Save output to .wav instead of playing")
    parser.add_argument('text', type=str, action='append', metavar='TEXT', help="Text to send")
    args = parser.parse_args()
    input_text = ' '.join(args.text)
    if args.output_wav_path is not None:
        return save_wav(input_text, args.host, args.port, args.output_wav_path)
    say(input_text, args.host, args.port)

if __name__ == "__main__":
    main()
