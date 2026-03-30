# ruff: noqa
# Source: kaggle kernels pull ttahara/birdclef-2026-hgnetv2-b0-baseline-inference
# Score: 0.884 LB
# Pulled: 2026-03-30

# # BirdCLEF+2026: HGNetV2-B0 Baseline [Inference]

# ## About
#
# This is a inference notebook.
#
# If you want to see the training process, visit the training notebook:
# https://www.kaggle.com/code/ttahara/birdclef-2026-hgnetv2-b0-baseline-training

# ## Preparation

# ### libraries

# --- Cell 4 ---
# !pip install --no-index \
#     -r /kaggle/input/notebooks/ttahara/birdclef-2026-download-wheels/requirements.txt \
#     --find-links=/kaggle/input/notebooks/ttahara/birdclef-2026-download-wheels/wheels

# --- Cell 5 ---
RANDOM_SEED = 1086

import os

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

# --- Cell 6 ---
import gc
import copy
import random
import typing as tp
from pathlib import Path

from time import time
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm, trange

import soundfile

import timm
import torch
import torchaudio
from torchvision.transforms import v2 as tvt_v2

from torch import nn

from joblib import load as jb_load, Parallel, delayed
import threading

import openvino as ov

# --- Cell 7 ---
ROOT = Path.cwd().parent

INPUT = ROOT / "input"
DATA = INPUT / "competitions" / "birdclef-2026"
TRAIN_AUDIO = DATA / "train_audio"
TRAIN_SS = DATA / "train_soundscapes"
TEST_SS = DATA / "test_soundscapes"

TRAINED_MODEL = INPUT / "notebooks" / "ttahara/birdclef-2026-hgnetv2-b0-baseline-training"

N_FOLDS = 4
N_CLASSES = 234

DEBUG = True
RANK_AVG = False

# --- Cell 8 ---
taxonomy = pd.read_csv(DATA / "taxonomy.csv")

CLASSES = taxonomy.primary_label.values.tolist()

label2idx = {label: idx for idx, label in enumerate(taxonomy.primary_label.values)}
idx2label = {idx: label for label, idx in label2idx.items()}


# --- Cell 9 ---
def set_random_seed(seed: int = 42, deterministic: bool = True):
    """Set seeds"""
    os.environ["PYTHONHASHSEED"] = str(seed)  # python
    random.seed(seed)  # python
    np.random.seed(seed)  # cpu
    torch.manual_seed(seed)  # cpu
    if torch.cuda.is_available():  # gpu
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic


set_random_seed(RANDOM_SEED)

# ### prepare file paths

# --- Cell 11 ---
sample_sub = pd.read_csv(DATA / "sample_submission.csv")
IS_TEST_ENV = len(sample_sub) > 10

if IS_TEST_ENV:
    # about 600 audio files in test environment
    test_ss_paths = []
    added = set()
    for row_id in sample_sub["row_id"].values:
        file_id = "_".join(row_id.split("_")[:-1])
        if file_id in added:
            continue
        added.add(file_id)
        test_ss_paths.append(TEST_SS / f"{file_id}.ogg")
else:
    if DEBUG:
        # debug by 600 audio files in train_soundscapes
        test_ss_paths = sorted(TRAIN_SS.iterdir())[:600]
    else:
        # fast submit
        test_ss_paths = sorted(TRAIN_SS.iterdir())[:10]

# --- Cell 12 ---
test_ss_segs = []
for p in test_ss_paths:
    for i in range(0, 60, 5):
        test_ss_segs.append(f"{p.stem}_{i + 5}")

# --- Cell 13 ---
for row_id in test_ss_segs[:24]:
    print(row_id)

# ### prepare log-mel-spectorogram transform and open vino models


# --- Cell 15 ---
class LogMelSpectrogramTransform(nn.Module):
    """"""

    def __init__(
        self, mel_spectrogram_params: tp.Dict, top_db: float, lms_shape=tp.Tuple[int, int]
    ):
        """"""
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spectrogram_params)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
        self.resize = tvt_v2.Resize(size=lms_shape)

    @torch.no_grad()
    def forward(self, wave):
        """
        wave: (B, sampling_rate * segment_sec)
        """
        mel_spec = self.mel_transform(wave)
        lms = self.db(mel_spec)  # shape: (B, n_mels, time)
        lms = self.resize(lms)  # shape: (B, *(lms_shape))

        batch_size = lms.shape[0]
        lms_flat = lms.reshape(batch_size, -1)
        lms_min = lms_flat.min(dim=1)[0][:, None, None]
        lms_max = lms_flat.max(dim=1)[0][:, None, None]
        lms = (lms - lms_min) / (lms_max - lms_min + 1e-7)

        return lms[:, None, :, :]


# --- Cell 16 ---
mel_spectrogram_params = dict(
    sample_rate=32_000,
    n_fft=2048,
    win_length=626,
    hop_length=313,
    f_min=20,
    n_mels=256,
    power=2.0,
    center=True,
    pad_mode="reflect",
    norm="slaney",
    mel_scale="htk",
)
top_db = 80
lms_shape = (256, 256)

lms_transform = LogMelSpectrogramTransform(
    mel_spectrogram_params, top_db=top_db, lms_shape=lms_shape
).eval()


# --- Cell 17 ---
def compile_ov_model(ov_model_path):
    compiled_model = ov.compile_model(
        str(ov_model_path),
        "CPU",
        {"PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_NUM_THREADS": 4, "NUM_STREAMS": 2},
    )
    return compiled_model


# --- Cell 18 ---
ov_model_list = []
for fold_id in range(N_FOLDS):
    compiled_model = compile_ov_model(str(TRAINED_MODEL / f"best_model_fold{fold_id}.xml"))
    ov_model_list.append(compiled_model)

# ## Inference

# ### create batchs

# --- Cell 21 ---
wave_batches = []
path_batches = []
tmp_wave_list = []
sample_rate = 32_000
max_sec = 60
duration = sample_rate * max_sec
batch_size = 12

for path in tqdm(test_ss_paths):
    with soundfile.SoundFile(path) as f:
        n_frames = f.frames

        if n_frames < duration:
            # 0-padding
            wave = np.zeros(duration, dtype="float32")
            wave[:n_frames] = f.read(dtype="float32")
        else:
            # head cropping
            wave = f.read(frames=duration, dtype="float32")

    for seg_sec in range(0, max_sec, 5):
        tmp_wave_list.append(wave[seg_sec * sample_rate : (seg_sec + 5) * sample_rate])

    if len(tmp_wave_list) == batch_size:
        batch = np.stack(tmp_wave_list, axis=0)
        wave_batches.append(batch)
        tmp_wave_list = []
else:
    if len(tmp_wave_list) > 0:
        batch = np.stack(tmp_wave_list, axis=0)
        wave_batches.append(batch)

# --- Cell 22 ---
print(len(wave_batches))

# --- Cell 23 ---
lms_batches = Parallel(n_jobs=4, verbose=1)(
    delayed(lms_transform)(torch.from_numpy(waves)) for waves in wave_batches
)

# --- Cell 24 ---
lms_batches = [np.ascontiguousarray(lms, dtype=np.float32) for lms in lms_batches]

del wave_batches
gc.collect()

# ## Inference


# --- Cell 26 ---
def async_infer_with_order(model, lms_batches, num_requests=4):
    """"""
    idx_batches = []
    tmp_idx = 0
    for b in lms_batches:
        b_size = len(b)
        idx_batches.append(np.arange(tmp_idx, tmp_idx + b_size))
        tmp_idx += b_size
    n_records = tmp_idx

    infer_queue = ov.AsyncInferQueue(model, num_requests)

    # array for predict result
    logit_arr = np.zeros((n_records, N_CLASSES), dtype=np.float32)

    start_time = time()

    def callback(request, userdata):
        input_idxs = userdata
        output = request.get_output_tensor().data
        logit_arr[input_idxs] = output

    infer_queue.set_callback(callback)
    input_name = compiled_model.inputs[0]

    for idxs, lms in zip(idx_batches, lms_batches):
        infer_queue.start_async({input_name: lms}, userdata=idxs)

    infer_queue.wait_all()

    print(f"... Done by {time() - start_time:.2f} sec")
    return logit_arr


# --- Cell 27 ---
def rank_normalize(x):
    r_x = np.zeros_like(x)
    for i in range(x.shape[1]):
        r_x_i = pd.Series(x[:, i]).rank(method="max")
        r_x[:, i] = r_x_i / r_x_i.shape[0]
    return r_x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --- Cell 28 ---
test_preds_arr = np.zeros((N_FOLDS, len(test_ss_segs), N_CLASSES))
test_preds_arr_rank = np.zeros((N_FOLDS, len(test_ss_segs), N_CLASSES))

for fold_id, ov_model in enumerate(ov_model_list):
    print(f"[fold: {fold_id}]")
    test_pred = async_infer_with_order(ov_model, lms_batches, 16)
    test_preds_arr[fold_id] = sigmoid(test_pred)
    test_preds_arr_rank[fold_id] = rank_normalize(test_pred)

# ## Make Submission

# --- Cell 30 ---
# fold averaging
if RANK_AVG:
    test_pred_avg = test_preds_arr_rank.mean(axis=0)
else:
    test_pred_avg = test_preds_arr.mean(axis=0)

sub_df = pd.DataFrame(
    test_pred_avg,
    columns=CLASSES,
    index=pd.Series(test_ss_segs, name="row_id"),
).reset_index()

display(sub_df.head())

# --- Cell 31 ---
sub_df = pd.merge(sample_sub[["row_id"]], sub_df, on="row_id", how="left")

sub_df.to_csv("submission.csv", index=False)
print(sub_df.shape)
display(sub_df.head())

# ## EOF
