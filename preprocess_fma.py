"""Convierte MP3 a .wav de 16kHz con los que trabaja SEGAN.

Recorre data/raw_audio/fma_small/{000.155}/*.mp3, procesa cada mp3 a .wav
de 16kHz, tomando los primeros 5 segundos y escribiendo a data/clean_fma/
"""

import argparse
import os
import glob
from multiprocessing import Pool

import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 16000
CLIP_SECONDS = 5
CLIP_SAMPLES = TARGET_SR * CLIP_SECONDS  # 80000


def process_one(args):
    src_path, dst_dir = args
    track_id = os.path.splitext(os.path.basename(src_path))[0]
    dst_path = os.path.join(dst_dir, track_id + ".wav")
    if os.path.exists(dst_path):
        return track_id, "skip"
    try:
        # decodifica mp3, pasa a mono y hace resampling a TARGET_SR
        y, _ = librosa.load(src_path, sr=TARGET_SR, mono=True, duration=CLIP_SECONDS)
    except Exception as e:
        return track_id, "error: {}".format(e)

    if y.shape[0] < CLIP_SAMPLES:
        # zero-padding para clips menores a 5s
        y = np.pad(y, (0, CLIP_SAMPLES - y.shape[0]))
    else:
        y = y[:CLIP_SAMPLES]

    # modelo utiliza int16 PCM, librosa regresa float en [-1, 1]
    y_int16 = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
    sf.write(dst_path, y_int16, TARGET_SR, subtype="PCM_16")
    return track_id, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="data/raw_audio/fma_small",
        help="Directorio que contiene folders FMA",
    )
    parser.add_argument(
        "--dst", default="data/clean_fma", help="Dirección destino para los .wavs"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Número de trabajadores para la tarea",
    )
    opts = parser.parse_args()

    os.makedirs(opts.dst, exist_ok=True)
    mp3_paths = sorted(glob.glob(os.path.join(opts.src, "*", "*.mp3")))
    print(
        f"Se encontraron {len(mp3_paths)} archivos mp3 en dirección {opts.src}"
    )
    if not mp3_paths:
        raise SystemExit("No se encontraron archivos mp3.")

    tasks = [(p, opts.dst) for p in mp3_paths]
    ok = skipped = errored = 0
    with Pool(opts.workers) as pool:
        for i, (track_id, status) in enumerate(
            pool.imap_unordered(process_one, tasks), 1
        ):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                errored += 1
                print(f"[{track_id}] {status}")
            if i % 200 == 0 or i == len(tasks):
                print(
                    f"{i}/{len(tasks)}, ok={ok}, saltados={skipped}, errores={errored}"
                )

    print(f"Terminado. Escribió {ok} .wavs en {opts.dst}")


if __name__ == "__main__":
    main()
