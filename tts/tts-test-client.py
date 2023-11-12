#!/usr/bin/env python3

import argparse
import sounddevice as sd
import requests
import numpy as np

def parse_content_type(content_type: str) -> dict:
    content_type_base, *params = content_type.split(';')
    params = dict(map(lambda x: x.split('='), params))
    if content_type_base == 'audio/pcm':
        return {'samplerate': int(params['rate']), 'dtype': np.dtype(f"{params['encoding']}{params['bits']}")}
    raise ValueError(f'Invalid content type: {content_type}')


def say(text: str, host: str, port: int) -> None:
    print("Requesting speech synthesis...")
    r = requests.post(f'http://{host}:{port}/api/v1/synthesize', json={'text': text})
    r.raise_for_status()
    audio_info = parse_content_type(r.headers['Content-Type'])
    print(audio_info)
    pcm = np.frombuffer(r.content, dtype=audio_info['dtype'])
    print("Playing audio...")
    audio_stream = sd.OutputStream(samplerate=audio_info['samplerate'], channels=1, blocksize=2048, dtype=audio_info['dtype'])
    audio_stream.start()
    audio_stream.write(pcm)
    audio_stream.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', type=str, required=False, default='127.0.0.1', metavar='HOST', help="Server host")
    parser.add_argument('-p', '--port', type=int, required=True, metavar='PORT', help="Server port number")
    parser.add_argument('text', type=str, action='append', metavar='TEXT', help="Text to send")
    args = parser.parse_args()
    say(' '.join(args.text), args.host, args.port)

if __name__ == "__main__":
    main()
