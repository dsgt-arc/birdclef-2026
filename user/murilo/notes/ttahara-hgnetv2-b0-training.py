# ruff: noqa
# Source: kaggle kernels pull ttahara/birdclef-2026-hgnetv2-b0-baseline-training
# Score: 0.884 LB (with inference notebook)
# Pulled: 2026-03-30

# # BirdCLEF+2026: HGNetV2-B0 Baseline [Training]
#
# 【UPDATE1】 I added AttnetionSEDHead and CustomLoss for it.
# 【UPDATE2】 I renewed CustomLoss, which adopts logit-LSE.
# 【UPDATE3】 I replaced the classifier head with LSEHead and used BCEWithLogitsLoss.
#
# ## About
#
# This notebook shows you an example of training process.
#
# I used some techniques for fast audio loading:
# - converted `.ogg` files into `.wav` in advance
#     - https://www.kaggle.com/datasets/ttahara/birdclef2026-train-audio-wav-00
#     - https://www.kaggle.com/datasets/ttahara/birdclef2026-train-audio-wav-01
#     - https://www.kaggle.com/datasets/ttahara/birdclef2026-train-audio-wav-02
#     - https://www.kaggle.com/datasets/ttahara/birdclef2026-train-audio-wav-03
# - load **not** entire the `.wav`file but only **the necessary parts(5 sec)** of it using `soundfile` library
#
# I trained `HGNetV2-B0` by 4-fold cross validation.
# Each fold took a little less than 30 minutes, therefore the entire training process completed in less than 2 hours.
#
# ### settings
# #### data
# * input : `train_audio` and `train_soundscapes`
# * target: `primary_label` and `secondary_labels`
# * CV split:
#     * Multi-Label Stratifiled Group K-Fold(K=4)
#     * using file ids as group ids
# * LogMelSpectrogram:
#     ```python
#     mel_spectrogram_params = dict(
#         sample_rate= 32_000,
#         n_fft      = 2048,
#         win_length = 626,
#         hop_length = 313,
#         f_min      = 20,
#         n_mels     = 256,
#         power      = 2.0,
#         center     = True,
#         pad_mode   = "reflect",
#         norm       = "slaney",
#         mel_scale  = 'htk',
#     )
#     top_db = 80
#     lms_shape = (256, 256)
#     ```
# #### model
# * backbone: `hgnetv2_b0.ssld_stage2_ft_in1k` from `timm`
# * head    : LSEHead(head_dropout=0.5, temperature=1.0)
#
# #### training
# * max_epoch : 20
# * batch_size: 64
# * optimizer: AdamW(`lr`=5e-4, `weight_decay`=1e-4)
# * scheduler: OneCosineLR(`max_lr`=5e-4, `min_lr`=5e-6, `warmup_epoch`=5)
# * data augmentation: MixUp(`alpha`=1.0, `theta`=0.8)
# * use amp training: True
# * loss: `nn.BCEWithLogitsLoss`

# ## Preparation

# ### libraries

# --- Cell 3 ---
# !pip install openvino onnxscript

# --- Cell 4 ---
RANDOM_SEED = 1086

import os

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

# --- Cell 5 ---
import ast
import gc
import copy
import math
import random
import typing as tp
from pathlib import Path

from time import time
from functools import wraps
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from tqdm.notebook import tqdm, trange
from sklearn.metrics import roc_auc_score, log_loss

import wave
import soundfile

import timm
import torch
import torchaudio
from torchvision.transforms import v2 as tvt_v2

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW

import joblib
import openvino as ov

# --- Cell 6 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Cell 7 ---
ROOT = Path.cwd().parent

INPUT = ROOT / "input"
DATA = INPUT / "competitions" / "birdclef-2026"
TRAIN_AUDIO = DATA / "train_audio"
TRAIN_SS = DATA / "train_soundscapes"
TEST_SS = DATA / "test_soundscapes"

TRAIN_AUDIO_WAVS = [
    INPUT / "datasets" / f"ttahara/birdclef2026-train-audio-wav-{i:02}" for i in range(4)
]
PROC = ROOT / "processed_data"
TRAIN_SS_SPLIT = PROC / "train_soundscapes_split"
TRAIN_SS_SPLIT.mkdir(exist_ok=True, parents=True)

N_FOLDS = 4
N_CLASSES = 234


# --- Cell 8 ---
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

# ### load label file

# --- Cell 10 ---
train_labels = pd.read_csv(DATA / "train.csv")

train_ss_labels = pd.read_csv(DATA / "train_soundscapes_labels.csv")
train_ss_labels = train_ss_labels.drop_duplicates().reset_index(drop=True)

train_ss_labels["start"] = pd.to_datetime(train_ss_labels["start"], format="%H:%M:%S")
train_ss_labels["end"] = pd.to_datetime(train_ss_labels["end"], format="%H:%M:%S")
train_ss_labels["start_sec"] = (
    train_ss_labels["start"].dt.minute * 60 + train_ss_labels["start"].dt.second
)
train_ss_labels["end_sec"] = (
    train_ss_labels["end"].dt.minute * 60 + train_ss_labels["end"].dt.second
)

# --- Cell 11 ---
taxonomy = pd.read_csv(DATA / "taxonomy.csv")

CLASSES = taxonomy.primary_label.values.tolist()

label2idx = {label: idx for idx, label in enumerate(taxonomy.primary_label.values)}
idx2label = {idx: label for label, idx in label2idx.items()}

# ### split labeled audio files in train_soundscapes

# --- Cell 13 ---
train_ss_ai_list = []
train_ss_fn_list = []
train_ss_pl_list = []
sampling_rate = 32_000

tmp_in_fn, tmp_pl, tmp_start, tmp_end = train_ss_labels.loc[
    0, ["filename", "primary_label", "start_sec", "end_sec"]
].values
tmp_wave, _ = soundfile.read(TRAIN_SS / tmp_in_fn, dtype="float32")

for in_fn, pl, start, end in tqdm(
    train_ss_labels.loc[1:, ["filename", "primary_label", "start_sec", "end_sec"]].values
):
    if in_fn == tmp_in_fn and pl == tmp_pl:
        tmp_end = end
        continue

    wave_seg = tmp_wave[tmp_start * sampling_rate : tmp_end * sampling_rate]
    ai = tmp_in_fn.split(".")[0]
    out_fn = f"{ai}_{tmp_start}-{tmp_end}.wav"
    soundfile.write(
        str(TRAIN_SS_SPLIT / out_fn),
        wave_seg,
        samplerate=sampling_rate,
        format="wav",
        subtype="FLOAT",
    )

    train_ss_ai_list.append(ai)
    train_ss_fn_list.append(out_fn)
    train_ss_pl_list.append(tmp_pl)

    tmp_pl = pl
    tmp_start = start
    tmp_end = end

    if in_fn != tmp_in_fn:
        tmp_in_fn = in_fn
        tmp_wave, _ = soundfile.read(TRAIN_SS / tmp_in_fn, dtype="float32")

else:
    wave_seg = tmp_wave[tmp_start * sampling_rate : tmp_end * sampling_rate]
    ai = tmp_in_fn.split(".")[0]
    out_fn = f"{ai}_{start}-{end}.wav"
    soundfile.write(
        str(TRAIN_SS_SPLIT / out_fn),
        wave_seg,
        samplerate=sampling_rate,
        format="wav",
        subtype="FLOAT",
    )

    train_ss_ai_list.append(ai)
    train_ss_fn_list.append(out_fn)
    train_ss_pl_list.append(tmp_pl)

# ### prepare train label dataframe

# --- Cell 15 ---
train_ss_labels_merged = pd.DataFrame(
    {
        "audio_id": train_ss_ai_list,
        "filename": train_ss_fn_list,
        "primary_label": train_ss_pl_list,
        "labels": train_ss_pl_list,
    }
)
train_ss_labels_merged["file_path"] = [
    str(TRAIN_SS_SPLIT / fn) for fn in train_ss_labels_merged["filename"].values
]

# --- Cell 16 ---
print(train_ss_labels_merged.shape)

# --- Cell 17 ---
display(train_ss_labels.head(12))

# --- Cell 18 ---
display(train_ss_labels_merged.head(5))

# --- Cell 19 ---
pl2wp = {}
for i in range(4):
    wav_path = TRAIN_AUDIO_WAVS[i]
    for pl_dir in sorted(wav_path.iterdir()):
        pl2wp[pl_dir.name] = wav_path

# --- Cell 20 ---
train_labels_merged = pd.DataFrame()
train_labels_merged["audio_id"] = train_labels["filename"].str.split(".").explode().values[0::2]
train_labels_merged["filename"] = train_labels["filename"]

train_labels_merged["primary_label"] = train_labels["primary_label"]
train_labels_merged["labels"] = [
    ";".join([pl] + ast.literal_eval(sls))
    for pl, sls in train_labels[["primary_label", "secondary_labels"]].values
]

train_labels_merged["file_path"] = [
    str(pl2wp[pl] / f"{fn.split(".")[0]}.wav")
    for fn, pl in train_labels_merged[["filename", "primary_label"]].values
]

# --- Cell 21 ---
train_df = pd.concat(
    [
        train_labels_merged,
        train_ss_labels_merged,
    ],
    axis=0,
    ignore_index=True,
)

# --- Cell 22 ---
labels_arr = np.zeros((len(train_df), len(label2idx)), dtype=np.float32)

for idx, labels in enumerate(train_df["labels"].values):
    for l in labels.split(";"):
        labels_arr[idx, label2idx[l]] = 1

# --- Cell 23 ---
print(train_df.shape)
label_df = pd.DataFrame(labels_arr, columns=list(label2idx.keys()))
train_df = pd.concat([train_df, label_df], axis=1)
print(train_df.shape)

# ### split folds


# --- Cell 25 ---
class MultiLabelStratifiedGroupKFold:
    def __init__(self, n_splits: int, random_state: int):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, label_arr: np.array, gid_arr: np.array):
        """
        create multi-label stratified group kfold indexs.

        reference: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
        input:
            label_arr: numpy.ndarray, shape = (n_train, n_class)
                multi-label for each sample's index using multi-hot vectors
            gid_arr: numpy.array, shape = (n_train,)
                group id for each sample's index
        output:
            yield indexs array list for each fold's train and validation.
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        start_time = time()
        n_train, n_class = label_arr.shape
        gid_unique = sorted(set(gid_arr))
        n_group = len(gid_unique)

        # # aid_arr: (n_train,), indicates alternative id for group id.
        # # generally, group ids are not 0-index and continuous or not integer.
        gid2aid = dict(zip(gid_unique, range(n_group)))
        # aid2gid = dict(zip(range(n_group), gid_unique))
        aid_arr = np.vectorize(lambda x: gid2aid[x])(gid_arr)

        # # count labels by class
        cnts_by_class = label_arr.sum(axis=0)  # (n_class, )

        # # count labels by group id.
        col, row = np.array(sorted(enumerate(aid_arr), key=lambda x: x[1])).T
        cnts_by_group = (
            coo_matrix((np.ones(len(label_arr)), (row, col)))
            .dot(coo_matrix(label_arr))
            .toarray()
            .astype(int)
        )
        del col
        del row
        cnts_by_fold = np.zeros((self.n_splits, n_class), int)

        groups_by_fold = [[] for fid in range(self.n_splits)]
        group_and_cnts = list(enumerate(cnts_by_group))  # pair of aid and cnt by group
        np.random.shuffle(group_and_cnts)
        print("finished preparation", time() - start_time)
        for aid, cnt_by_g in sorted(group_and_cnts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for fid in range(self.n_splits):
                # # eval assignment.
                cnts_by_fold[fid] += cnt_by_g
                fold_eval = (cnts_by_fold / cnts_by_class).std(axis=0).mean()
                cnts_by_fold[fid] -= cnt_by_g

                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = fid

            cnts_by_fold[best_fold] += cnt_by_g
            groups_by_fold[best_fold].append(aid)
        print("finished assignment.", time() - start_time)

        gc.collect()
        idx_arr = np.arange(n_train)
        for fid in range(self.n_splits):
            val_groups = groups_by_fold[fid]

            val_indexs_bool = np.isin(aid_arr, val_groups)
            train_indexs = idx_arr[~val_indexs_bool]
            val_indexs = idx_arr[val_indexs_bool]

            print("[fold {}]".format(fid), end=" ")
            print(
                "n_group: (train, val) = ({}, {})".format(
                    n_group - len(val_groups), len(val_groups)
                ),
                end=" ",
            )
            print("n_sample: (train, val) = ({}, {})".format(len(train_indexs), len(val_indexs)))

            yield train_indexs, val_indexs


# --- Cell 26 ---
mlsgkf = MultiLabelStratifiedGroupKFold(n_splits=N_FOLDS, random_state=RANDOM_SEED)
train_val_splits = list(mlsgkf.split(labels_arr, train_df["audio_id"].values))

# --- Cell 27 ---
train_df.insert(5, "fold", -1)
for fold_id, (trn_idx, val_idx) in enumerate(train_val_splits):
    train_df.loc[val_idx, "fold"] = fold_id

# --- Cell 28 ---
for fold_id in range(N_FOLDS):
    print(f"[fold {fold_id}]")
    print(
        "train_audio      :",
        ((train_df["fold"] != fold_id) & train_df["filename"].str.contains("ogg")).sum(),
        ((train_df["fold"] == fold_id) & train_df["filename"].str.contains("ogg")).sum(),
    )
    print(
        "train_soundscapes:",
        ((train_df["fold"] != fold_id) & train_df["filename"].str.contains("wav")).sum(),
        ((train_df["fold"] == fold_id) & train_df["filename"].str.contains("wav")).sum(),
    )

# ## Training

# ### definition

# #### dataset


# --- Cell 32 ---
class BirdTrainDataset(Dataset):
    """"""

    def __init__(self, paths, labels, sampling_rate=32_000, segment_sec=5):
        """"""
        self.paths = paths
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.segment_sec = segment_sec

    def __len__(self):
        """"""
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        wave = self._load_audio_file(path)
        label = self.labels[idx].astype(np.float32)
        return {
            "wave": wave,
            "label": label,
        }

    def _load_audio_file(self, path):
        """"""
        duration = self.sampling_rate * self.segment_sec
        with soundfile.SoundFile(path) as f:
            n_frames = f.frames

            if n_frames < duration:
                # 0-padding
                wave = np.zeros(duration, dtype="float32")
                start = np.random.randint(duration - n_frames + 1)
                wave[start : start + n_frames] = f.read(dtype="float32")
            else:
                # random cropping
                start = np.random.randint(n_frames - duration + 1)
                f.seek(start)
                wave = f.read(frames=duration, dtype="float32")

        return wave


class BirdValidDataset(Dataset):
    """"""

    def __init__(self, paths, labels, sampling_rate=32_000, segment_sec=5):
        """"""
        self.paths = paths
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.segment_sec = segment_sec

    def __len__(self):
        """"""
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        wave = self._load_audio_file(path)
        label = self.labels[idx]

        return {
            "wave": wave,
            "label": label.astype(np.float32),
        }

    def _load_audio_file(self, path):
        """"""
        duration = self.sampling_rate * self.segment_sec

        with soundfile.SoundFile(path) as f:
            n_frames = f.frames

            if n_frames < duration:
                # 0-padding
                wave = np.zeros(duration, dtype="float32")
                wave[:n_frames] = f.read(dtype="float32")
            else:
                # head cropping
                wave = f.read(frames=duration, dtype="float32")

        return wave


# #### preprocessing


# --- Cell 34 ---
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


# --- Cell 35 ---
class MixUp(nn.Module):
    """mixup for Log Mel Spectrogram"""

    def __init__(self, alpha=0.5, theta=1):
        """"""
        super().__init__()
        self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)
        self.theta = theta

    def forward(self, lms, label):
        """
        lms  : (B, C, Freq, Time)
        label: (B, N_CLS)
        """
        batch_size = lms.shape[0]
        device = lms.device

        lambda_tensor = self.beta_dist.sample(sample_shape=(batch_size,)).to(device)
        lambda_tensor = torch.maximum(lambda_tensor, 1 - lambda_tensor).float()

        shuffle_idxs = torch.randperm(batch_size).to(device)

        lms_lambda = lambda_tensor[:, None, None, None]
        lms = lms_lambda * lms + (1 - lms_lambda) * lms[shuffle_idxs]

        label_lambda = lambda_tensor[..., None]
        label = label_lambda * label + (1 - label_lambda) * label[shuffle_idxs]
        label[label >= self.theta] = 1

        return lms, label


def dummy_mixup(lms, label):
    return lms, label


# #### model


# --- Cell 37 ---
class GeMPooling(nn.Module):
    """"""

    def __init__(self, init_p=3.0, mean_axis=(1, 2), eps=1e-6):
        """"""
        super().__init__()
        self.p = nn.Parameter(torch.tensor(init_p))
        self.mean_axis = mean_axis
        self.eps = eps

    def forward(self, h):
        """
        h: shape=(B, C, H, W)
        """
        p = self.p.clip(min=1.0)
        h = h.clip(min=self.eps).pow(self.p)
        h = h.mean(dim=self.mean_axis)
        h = h.pow(1.0 / self.p)
        return h


class AttnSEDHead(nn.Module):
    """"""

    def __init__(self, num_features, num_classes, dropout=0.1):
        """"""
        super().__init__()
        self.pre_fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.att_fc = nn.Linear(num_features, num_classes)
        self.cls_fc = nn.Linear(num_features, num_classes)

    def forward(self, h):
        """
        h: (B, C, Time)
        """
        h = h.permute(0, 2, 1)  # (B, Time, C)
        h = self.pre_fc(h)  # (B, Time, C)
        att_w = torch.tanh(self.att_fc(h))  # (B, Time, N_CLASSES)
        att_w = F.softmax(att_w, dim=1)  # (B, Time, N_CLASSES)
        timewise_logits = self.cls_fc(h)  # (B, Time, N_CLASSES)
        logits = (att_w * timewise_logits).sum(dim=1)  # (B, N_CLASSES)
        return logits, timewise_logits


class AttnSEDModel(nn.Module):
    """"""

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        drop_path_rate: float = 0.0,
        head_dropout: float = 0.1,
        num_classes: int = 234,
    ):
        """"""
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            global_pool="",
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        # Some backbone's num_features don't match its output
        dummy_input = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            dummy_output = self.backbone(dummy_input)
        num_features = dummy_output.shape[1]
        print(f"num_features: {self.backbone.num_features}, dummy_output's dim: {num_features}")
        self.gem_pool = GeMPooling(mean_axis=2)
        self.head = AttnSEDHead(num_features, num_classes, head_dropout)

    def forward_for_training(self, x):
        """"""
        h = self.backbone(x)  # (B, C, Freq, Time)
        h = self.gem_pool(h)  # (B, C, Time)
        logits, timewise_logits = self.head(h)  # (B, N_CLASSES), (B, N_CLASSES, Freq' * Time')
        return logits, timewise_logits

    def forward(self, x):
        """"""
        logits, _ = self.forward_for_training(x)

        return logits


# --- Cell 38 ---
class LSEPooling(nn.Module):
    """"""

    def __init__(self, pool_axis=-1, temperature: float = 1.0):
        """"""
        super().__init__()
        self.temperature = temperature
        self.pool_axis = pool_axis

    def forward(self, x):
        """"""
        y = self.temperature * (
            torch.logsumexp(x / self.temperature, axis=self.pool_axis)
            - math.log(x.shape[self.pool_axis])
        )
        return y


class LSEHead(nn.Module):
    """"""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        dropout: float = 0.2,
        lse_temperature: float = 1.0,
    ):
        """"""
        super().__init__()
        self.lse_pool = LSEPooling(pool_axis=1, temperature=lse_temperature)
        self.cls_fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, h):
        """
        h : (B, C, Freq, Time)
        """
        h = torch.mean(h, axis=2)  # (B, C, Time)
        h = h.transpose(1, 2)  # (B, Time, C)
        timewise_logits = self.cls_fc(h)  # (B, Time, N_CLS)
        logits = self.lse_pool(timewise_logits)  # (B, N_CLS)

        return logits


class LSEModel(nn.Module):
    """"""

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        drop_path_rate: float = 0.0,
        num_classes: int = 234,
        head_dropout: float = 0.2,
        lse_temperature: float = 1.0,
    ):
        """"""
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            global_pool="",
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        # Some backbone's num_features don't match its output
        dummy_input = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            dummy_output = self.backbone(dummy_input)
        num_features = dummy_output.shape[1]
        self.head = LSEHead(num_features, num_classes, head_dropout, lse_temperature)

    def forward(self, x):
        """"""
        h = self.backbone(x)  # (B, C, Freq, Time)
        logits = self.head(h)  # (B, N_CLS)
        return logits


# --- Cell 39 ---
class CustomBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """BCEWithLogitsLoss for SEDModel."""

    def __init__(self, timewise_weight: float = 0.5, temperature: float = 1.0, **kwargs):
        """"""
        super().__init__(**kwargs)
        self.timewise_weight = timewise_weight
        self.temperature = temperature

    def forward(self, logits, timewise_logits, label):
        """
        logits         : (B, N_CLASSES)
        timewise_logits: (B, Time, N_CLASSES)
        """
        loss = super().forward(logits, label)
        logits_timeaxis_lse = self.temperature * (
            torch.logsumexp(timewise_logits / self.temperature, axis=1)
            - math.log(timewise_logits.shape[1])
        )
        loss_timeaxis_lse = super().forward(logits_timeaxis_lse, label)

        return (1 - self.timewise_weight) * loss + self.timewise_weight * loss_timeaxis_lse


# #### utility


# --- Cell 41 ---
def to_device(
    tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
    device: torch.device,
    *args,
    **kwargs,
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


# --- Cell 42 ---
def get_data_loader(train_df, fold_id, device=None):
    trn_idxs = train_df.query("fold != @fold_id").index.values
    val_idxs = train_df.query("fold == @fold_id").index.values

    file_paths = train_df["file_path"].values.tolist()
    labels_arr = train_df[CLASSES].values

    trn_paths = [file_paths[idx] for idx in trn_idxs]
    trn_labels = [labels_arr[idx] for idx in trn_idxs]
    val_paths = [file_paths[idx] for idx in val_idxs]
    val_labels = [labels_arr[idx] for idx in val_idxs]

    trn_dataset = BirdTrainDataset(trn_paths, trn_labels)
    val_dataset = BirdValidDataset(val_paths, val_labels)

    trn_loader = DataLoader(
        trn_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=16,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=16,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    return trn_loader, val_loader


# ### run training


# --- Cell 44 ---
class CFG:
    max_epoch = 20
    warmup_epoch = 5
    batch_size = 64
    lr = 5e-04
    weight_decay = 1e-04

    # for model
    model_name = "hgnetv2_b0.ssld_stage2_ft_in1k"
    pretrained = True
    drop_path_rate = 0.0
    head_dropout = 0.5
    lse_temperature = 1.0  # for logit-LSE

    # for log mel spectrogram transform
    mel_spectrogram_params = dict(
        sample_rate=32_000,
        n_fft=2048,
        win_length=626,
        hop_length=313,
        f_min=20,
        # f_max=16_000,
        n_mels=256,
        power=2.0,
        center=True,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="htk",
    )
    lms_shape = (256, 256)
    top_db = 80.0

    # other
    mixup = dict(alpha=1.0, theta=0.8)
    use_amp = True
    use_dp = False


# --- Cell 45 ---
def train_one_fold(train_df, fold_id, device, output_dir=None):
    if output_dir == None:
        output_dir = Path.cwd()
    else:
        output_dir.mkdir(exist_ok=True)

    print(f"[training fold {fold_id}]")
    print(
        f"train: {len(train_df.query("fold != @fold_id"))}, valid: {len(train_df.query("fold == @fold_id"))}"
    )

    set_random_seed(RANDOM_SEED)
    trn_loader, val_loader = get_data_loader(train_df, fold_id, device)

    lms_transform = LogMelSpectrogramTransform(
        CFG.mel_spectrogram_params, CFG.top_db, CFG.lms_shape
    )
    lms_transform = lms_transform.eval().to(device)
    mixup = MixUp(**CFG.mixup)
    model = LSEModel(
        CFG.model_name,
        CFG.pretrained,
        drop_path_rate=CFG.drop_path_rate,
        num_classes=N_CLASSES,
        head_dropout=CFG.head_dropout,
        lse_temperature=CFG.lse_temperature,
    )
    model = model.to(device)

    num_devices = torch.cuda.device_count()
    if CFG.use_dp and num_devices > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    optimizer.zero_grad()
    scheduler = OneCycleLR(
        optimizer=optimizer,
        epochs=CFG.max_epoch,
        pct_start=CFG.warmup_epoch / CFG.max_epoch,
        max_lr=CFG.lr,
        div_factor=25,
        final_div_factor=4.0e-00,
        steps_per_epoch=len(trn_loader),
    )
    loss_func = nn.BCEWithLogitsLoss()
    grad_scaler = torch.GradScaler(enabled=CFG.use_amp)

    result_list = []
    best_val_score = 0.0

    for epoch in trange(CFG.max_epoch):
        epoch_start = time()
        trn_loss = 0.0
        tmp_mixup = mixup if epoch >= CFG.warmup_epoch else dummy_mixup
        model = model.train()
        for batch in trn_loader:
            batch = to_device(batch, device, non_blocking=True)
            wave, label = batch["wave"], batch["label"]
            lms = lms_transform(wave)
            lms, label = tmp_mixup(lms, label)
            with torch.autocast(device.type, enabled=grad_scaler.is_enabled()):
                logits = model(lms.detach())
                loss = loss_func(logits, label)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            trn_loss += loss.item()
            del batch, wave, label, lms, logits, loss

        trn_loss /= len(trn_loader)

        model = model.eval()
        logit_list = []
        label_list = []
        for batch in val_loader:
            lms = lms_transform(to_device(batch["wave"], device, non_blocking=True))
            with torch.no_grad() and torch.autocast(device.type, enabled=grad_scaler.is_enabled()):
                logits = model(lms.detach())

            logit_list.append(logits.detach().cpu())
            label_list.append(batch["label"].detach())
            del batch, lms, logits

        logits = torch.cat(logit_list, axis=0)
        labels = torch.cat(label_list, axis=0)
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()

        logits = logits.numpy()
        labels = labels.numpy()
        mask = labels.sum(axis=0) > 0

        val_score = roc_auc_score(labels[:, mask], logits[:, mask], average="macro")

        if val_score > best_val_score:
            best_val_score = val_score
            if CFG.use_dp and num_devices > 1:
                best_state = {k: v.detach().cpu() for k, v in model.module.state_dict().items()}
            else:
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_pred = logits

        epoch_end = time()
        result_list.append(
            [
                epoch,
                scheduler.get_last_lr()[0],
                trn_loss,
                val_loss,
                val_score,
                epoch_end - epoch_start,
            ]
        )
        print(
            "[epoch {}] lr={:.6f}, trn_loss={:.5f}, val_loss={:.5f}, val_score={:.5f}. elapsed={:.2f}".format(
                *result_list[-1]
            )
        )

    # save training process
    result_df = pd.DataFrame(
        result_list, columns=["epoch", "lr", "trn_loss", "val_loss", "val_score", "elapsed_time"]
    )
    result_df.to_csv(output_dir / f"result_df_fold{fold_id}.csv", index=False)
    # save best model
    torch.save(best_state, output_dir / f"best_model_fold{fold_id}.pt")
    # save val_pred by best model
    np.save(output_dir / f"best_val_pred_fold{fold_id}.npy", best_val_pred)


# --- Cell 46 ---
print(list(filter(lambda x: x[0][:2] != "__", CFG.__dict__.items())))
for fold_id in range(N_FOLDS):
    train_one_fold(train_df, fold_id, device)

# ## Calculate OOF Score


# --- Cell 48 ---
def rank_normalize(x):
    r_x = np.zeros_like(x)

    for i in range(x.shape[1]):
        r_x_i = pd.Series(x[:, i]).rank(method="max")
        r_x[:, i] = r_x_i / r_x_i.shape[0]

    return r_x


# ### calculate CV score of predictions by torch model (on GPU)

# --- Cell 50 ---
oof_pred_trn = np.zeros((len(train_df), N_CLASSES))
oof_pred_trn_rank = np.zeros((len(train_df), N_CLASSES))

for fold_id, (trn_idxs, val_idxs) in enumerate(train_val_splits):
    val_pred = np.load(f"best_val_pred_fold{fold_id}.npy")
    oof_pred_trn[val_idxs] = val_pred
    oof_pred_trn_rank[val_idxs] = rank_normalize(val_pred)

# --- Cell 51 ---
print("auc for raw pred :", roc_auc_score(labels_arr, oof_pred_trn))
print("auc for rank pred:", roc_auc_score(labels_arr, oof_pred_trn_rank))

# ## convert torch models to openvino models and save them


# --- Cell 53 ---
def convert_torch_to_onnx_to_ov(
    cfg,
    model_path: Path,
    out_dir: Path,
):
    """convert torch model to openvino model via onnx model"""
    torch_model = LSEModel(
        cfg.model_name,
        pretrained=False,
        drop_path_rate=cfg.drop_path_rate,
        num_classes=N_CLASSES,
        head_dropout=cfg.head_dropout,
        lse_temperature=CFG.lse_temperature,
    )
    torch_model.load_state_dict(torch.load(model_path))
    torch_model = torch_model.eval()

    dummy_input = torch.randn(64, 1, 256, 256, dtype=torch.float32)
    onnx_path = out_dir / f"{model_path.stem}.onnx"
    _ = torch.onnx.export(
        torch_model,
        (dummy_input,),
        str(onnx_path),
        opset_version=11,
        do_constant_folding=True,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    ov_model = ov.convert_model(onnx_path)
    onnx_path.unlink()
    ov.save_model(ov_model, out_dir / f"{model_path.stem}.xml")

    return ov_model


# --- Cell 54 ---
lms_transform = LogMelSpectrogramTransform(
    CFG.mel_spectrogram_params, top_db=CFG.top_db, lms_shape=CFG.lms_shape
).eval()

joblib.dump(lms_transform, "log_mel_spectrogram.jb")

# --- Cell 55 ---
out_dir = Path.cwd()
for fold_id in range(N_FOLDS):
    model_state_path = out_dir / f"best_model_fold{fold_id}.pt"
    ov_model = convert_torch_to_onnx_to_ov(CFG, model_state_path, out_dir)

# ### calculate score of predictions by openvino model on CPU

# --- Cell 57 ---
lms_transform = (
    LogMelSpectrogramTransform(
        CFG.mel_spectrogram_params, top_db=CFG.top_db, lms_shape=CFG.lms_shape
    )
    .eval()
    .to(device)
)

ov_model_list = []
for fold_id in range(N_FOLDS):
    ov_model_path = f"best_model_fold{fold_id}.xml"
    compiled_model = ov.compile_model(
        ov_model_path,
        "CPU",
        {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "INFERENCE_NUM_THREADS": 4,
            "NUM_STREAMS": 2,
        },
    )
    ov_model_list.append(compiled_model)

gc.collect()


# --- Cell 58 ---
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


# --- Cell 59 ---
oof_pred_ov = np.zeros((len(train_df), N_CLASSES), dtype="float32")
oof_pred_ov_rank = np.zeros((len(train_df), N_CLASSES), dtype="float32")

CFG.batch_size = 4
for fold_id, (_, val_idxs) in enumerate(train_val_splits):
    _, val_loader = get_data_loader(train_df, fold_id)
    ov_model = ov_model_list[fold_id]
    lms_batches = []
    for batch in tqdm(val_loader):
        lms = lms_transform(to_device(batch["wave"], device))
        lms_batches.append(lms.cpu().numpy())

    logits = async_infer_with_order(ov_model, lms_batches, 16)

    oof_pred_ov[val_idxs] = logits
    oof_pred_ov_rank[val_idxs] = rank_normalize(oof_pred_ov[val_idxs])

# --- Cell 60 ---
print("auc for raw pred :", roc_auc_score(labels_arr, oof_pred_ov))
print("auc for rank pred:", roc_auc_score(labels_arr, oof_pred_ov_rank))

# ## EOF
