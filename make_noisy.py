"""Crea par noisy a partir de .wav limpio + audio de datset DEMAND.

Para cada .wav en --clear_dir empareja con un sonido aleatorio de --noises_dir
con SNR aleatorio dado por --snr_levels de segan.utils.Additive. El resultado se divide
80/10/10 en conjuntos de entrenamiento, validación y prueba. Output:

    <out_root>/
        clean_trainset/<id>.wav   (copy of the clean file)
        noisy_trainset/<id>.wav   (clean + noise at random SNR)
        clean_valset/<id>.wav
        noisy_valset/<id>.wav
        clean_testset/<id>.wav
        noisy_testset/<id>.wav

El script es resumible.
"""

import argparse
import glob
import os
import random
import shutil
from multiprocessing import Pool

import numpy as np
import scipy.io.wavfile as wavfile

from segan.utils import Additive

SR = 16000

_ADDITIVE = None  # populated per-worker via initializer


def _init_worker(noises_dir, snr_levels, base_seed):
    global _ADDITIVE
    _ADDITIVE = Additive(noises_dir, snr_levels=snr_levels)
    np.random.seed(base_seed + os.getpid())


def _process(args):
    clean_path, clean_out, noisy_out = args
    rate, wav_i16 = wavfile.read(clean_path)
    if rate != SR:
        return clean_path, "bad sr={}".format(rate)
    wav = wav_i16.astype(np.float32) / 32768.0
    noisy = _ADDITIVE(wav).numpy()
    noisy_i16 = np.clip(noisy * 32767.0, -32768, 32767).astype(np.int16)
    if not os.path.exists(clean_out):
        shutil.copyfile(clean_path, clean_out)
    wavfile.write(noisy_out, SR, noisy_i16)
    return clean_path, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", default="data/raw_audio/clean_fma")
    parser.add_argument("--noises_dir", default="data/raw_audio/musan_noise")
    parser.add_argument("--out_root", default="data")
    parser.add_argument("--snr_levels", type=int, nargs="+", default=[0, 5, 10, 15])
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1)
    )
    parser.add_argument("--seed", type=int, default=111)
    opts = parser.parse_args()

    clean_paths = sorted(glob.glob(os.path.join(opts.clean_dir, "*.wav")))
    if not clean_paths:
        raise SystemExit(f"No se encontraron archivos .wav en {opts.clean_dir}")

    rng = random.Random(opts.seed)
    rng.shuffle(clean_paths)
    n_valid = int(round(len(clean_paths) * opts.valid_ratio))
    n_test = int(round(len(clean_paths) * opts.test_ratio))
    valid_paths = clean_paths[:n_valid]
    test_paths = clean_paths[n_valid : n_valid + n_test]
    train_paths = clean_paths[n_valid + n_test :]
    total = len(train_paths) + len(valid_paths) + len(test_paths)
    print(
        f"Total: {total}, train: {len(train_paths)}, valid: {len(valid_paths)}, test: {len(test_paths)}"
    )

    dirs = {
        "train_clean": os.path.join(opts.out_root, "clean_trainset"),
        "train_noisy": os.path.join(opts.out_root, "noisy_trainset"),
        "valid_clean": os.path.join(opts.out_root, "clean_valset"),
        "valid_noisy": os.path.join(opts.out_root, "noisy_valset"),
        "test_clean": os.path.join(opts.out_root, "clean_testset"),
        "test_noisy": os.path.join(opts.out_root, "noisy_testset"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    tasks = []
    for p in train_paths:
        name = os.path.basename(p)
        tasks.append(
            (
                p,
                os.path.join(dirs["train_clean"], name),
                os.path.join(dirs["train_noisy"], name),
            )
        )
    for p in valid_paths:
        name = os.path.basename(p)
        tasks.append(
            (
                p,
                os.path.join(dirs["valid_clean"], name),
                os.path.join(dirs["valid_noisy"], name),
            )
        )
    for p in test_paths:
        name = os.path.basename(p)
        tasks.append(
            (
                p,
                os.path.join(dirs["test_clean"], name),
                os.path.join(dirs["test_noisy"], name),
            )
        )

    pending = [t for t in tasks if not os.path.exists(t[2])]
    print(f"Pares pendientes: {len(pending)} / {len(tasks)}")
    if not pending:
        print("No hay pares pendientes")
        return

    ok = errs = 0
    with Pool(
        opts.workers,
        initializer=_init_worker,
        initargs=(opts.noises_dir, opts.snr_levels, opts.seed),
    ) as pool:
        for i, (path, status) in enumerate(pool.imap_unordered(_process, pending), 1):
            if status == "ok":
                ok += 1
            else:
                errs += 1
                print(f"[{path}] {status}")
            if i % 200 == 0 or i == len(pending):
                print(f"{i}/{len(pending)}, ok={ok}, errores={errs}")

    print(f"Terminado. Pares escritos en {opts.out_root}")


if __name__ == "__main__":
    main()
