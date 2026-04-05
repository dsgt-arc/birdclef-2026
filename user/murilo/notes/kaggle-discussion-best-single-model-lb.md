# Kaggle Discussion: What is your best single model LB score?

- **URL**: https://www.kaggle.com/competitions/birdclef-2026/discussion/683791
- **Posted**: 2026-03-21 (Antoine Masq)
- **Captured**: 2026-04-05
- **24 comments**

---

## Original Post

**Antoine Masq** (706th) — Posted 14 days ago

> Hi everyone,
>
> As most of us are probably still improving our baseline models, I was curious to
> see what your best LB score is for a single model without using the unlabeled data.
>
> So far, my best single model performance is 0.889 with an EfficientNetv2_b0 backbone

---

## Key Competitor Scores

| Person | Rank | LB | Details |
|--------|------|-----|---------|
| **Boredom** | 7th | **0.922** single/single-fold | EfficientNet, "easily 925+" |
| **Salman Ahmed** | 448th | **0.921** solo, **0.923** w/ priors | 2 SED ensemble, CE+sigmoid, waveform MixUp |
| **Ali Ozan Memetoglu** | 2nd | **0.925** single model | No unlabeled data, no details |
| **hengck23** | 175th | **0.896** CNN only | Long input wave, augmentation, no Perch, no PP |
| **Tom** | 8th | ~similar to hengck23 | "slightly different approach which fits on 4x5sec" |
| **EliKal** | 464th | **0.909** single (EffNet-B0) | Distillation (KLD) + 12ep finetune, time-shift TTA |
| **yukiZ** | 71st | **0.904** (fold 0 only, 5-fold SKF) | Raw waveform augmentations |
| Antoine Masq | 706th | 0.889 | EfficientNetV2-B0 |
| Don Mathis | 1007th | 0.880 | CNN + logit-LSE, BCE, 5s clips |

---

## Comments (chronological)

### Salman Ahmed (448th) — Posted 14 days ago

> My best single model is SED Efficientnet B0 (Trained on 10 sec). Mixup on Raw
> Waves, and CrossEntropyLoss on clipwise and framewise max. Trained for 20+ epochs.
> This single model scores 0.908
>
> I have a ensemble of these 2 models and it scores 0.914
>
> No other augmentation. No unlabeled data.

### Ali Ozan Memetoglu (2nd) — Posted 14 days ago

> Mine is at 0.925 without using the unlabeled data. I couldn't make good use of
> it, and it might not even be that useful in this competition. But I might be wrong.

### Tom (8th) — Posted 14 days ago

> I think the floor of top scores would be very high in the end

### hengck23 (175th) — Posted 14 days ago

> "and it might not even be that useful in this competition. But I might be wrong."
>
> it should be very useful! in fact winning solution is how you make use of it

### yukiZ (71st) — Posted 14 days ago

> I believe our interest lies in whether there is a correlation between CV and LB,
> so I will share my experimental results. However, due to the highly skewed
> distribution of the data, the validation strategy will likely differ depending
> on the participant.
>
> The following experimental results show the results of training/submitting using
> only Fold0 with 5Fold SKF, assuming future Blending of the BB of the SED Model
> with B0 (approximately 5-7M parameters) or Alternatively, it uses a backbone of
> around 15M, similar to v2s.
>
> | MyExpNo | AUC (LocalValidation, Fold1/5) | LB (Fold1/5) |
> |---------|-------------------------------|--------------|
> | #107 | 0.9774 | 0.904 |
> | #105 | 0.9756 | 0.902 |
> | #042v4 | 0.9768 | 0.898 |
> | #042 | 0.9744 | 0.888 |
> | #041 | 0.9737 | 0.885 |
> | #040 | 0.9722 | 0.878 |
>
> [2026/03/27 Upd. Add Exp "#042v4","#105","#107"]

### Amit Aharoni (701st) — Posted 13 days ago

> Thanks for sharing. Based on the AUC score, is it safe to assume that your local
> validation is using train_audio and not train_soundscape?

### yukiZ (71st) — Posted 12 days ago

> Yes, your way of thinking is correct.

### Boredom (7th) — Posted 14 days ago

> Single model / Single Fold: 0.922 without using the unlabeled/extra data

### vion (979th) — Posted 10 days ago

> Amazing score considering that 0.940 is the maximum achievable with 206 of 234
> classes. Congrats!

### Murilo Gustineli (347th) — Posted 6 days ago

> My best single SED model is 0.86 LB with EfficientNet-B0 - and I've been
> struggling to break past that despite extensive ablations.
>
> My setup: EfficientNet-B0 SED (ImageNet init), GeM + attention head, n_fft=4096,
> n_mels=224, AmplitudeToDB norm, soft CE dual loss (clipwise + framewise max), 10s
> windows, spectrogram MixUp (alpha=0.5), 90/10 split, 50 epochs with CosineAnnealing.
> Trained on focal + labeled soundscapes (234 classes).
>
> I've run 15+ ablations (BCE, ASL, LSEHead, HGNetV2-B0, waveform MixUp, weight
> decay, StochasticDepth) and can't break past 0.86 - offline val improvements
> consistently fail to translate to LB gains.
>
> I also reproduced ttahara's HGNetV2-B0 recipe (LSEHead, BCE, OneCycleLR,
> per-sample MixUp, 5s windows, 20 epochs) and got 0.872 val/auroc offline, but
> only 0.855 LB on EfficientNet with the same recipe. The val-to-LB disconnect
> has been consistent across all ablations - improvements in offline val rarely
> translate to LB gains.
>
> Curious what's driving the gap to 0.9+ for other single EfficientNet-B0 SEDs.
> A few things we haven't explored:
> - NS-JFT pretrained weights (tf_efficientnet_b0.ns_jft_in1k) vs standard ImageNet init
> - 3-channel mel input (repeat mono to match pretrained stem) vs 1-channel with weight averaging
> - Longer context windows (15-20s) for capturing insect/amphibian textures
> - Training on focal only vs focal + soundscapes
>
> Would love to hear from folks who've broken 0.9+

### Salman Ahmed (448th) — Posted 5 days ago

> In my experiments, more epochs make the model too confident on it's predictions
> and that hurt the AUC (Evaluation metric).
>
> In my experiments, SED simple Attn head's framewise max is better than lse head.
>
> I mixup waves rather than Spectrograms.
>
> For now, Efficientnet B0 didn't help.
>
> Try to listen to a audio from train soundscape and then look at the attentionhead's
> output. That might help.

### hengck23 (175th) — Posted 5 days ago

> it is better to think of baseline
> 1) use perch embedding to get cv and public lb
>    just train a linear or MLP probe on clip and evaluate on soundscape ss
>    (in addition to clip). this give you an idea on effect of domain shift.
>
> 2) repeat (1) but using your model. train on clip only
>    read perch paper and see how many train samples they use and compare that
>    with kaggle data. this gives you an idea of how num and variety of data
>    affects results.
>
> from these two, it would be enough to see if your results is reasonable or no
> and other kagger top results is due to tricks, post processing, better
> modelling, better data, etc

### yukiZ (71st) — Posted 5 days ago

> Hi! Regarding the contribution of CV/LB in my experiment, data augmentation was
> effective.
>
> I believe many publicly available notebooks include Mixup. On the other hand,
> data augmentation on raw waveforms is not seen very often, but it has been
> frequently used in solutions from past BirdCLEF competitions. For example,
> "Pink Noise" and "Time Stretch".

### Murilo Gustineli (347th) — Posted 6 days ago

> Is it safe to assume that you're only training on train_audio or also using the
> train_soundscapes to achieve the 0.908 LB score?

### Salman Ahmed (448th) — Posted 5 days ago

> Yup. Train audio + label soundscapes.

### Jiacheng Ma (642nd) — Posted 5 days ago

> What are label soundscapes and unlabel soundscapes? Thanks.

### Salman Ahmed (448th) — Posted 5 days ago

> train_soundscapes_labels.csv are labeled soundscapes.

### Salman Ahmed (448th) — Posted 2 days ago (UPDATED)

> Alright, Finally have 2 SED models with ensemble of LB 0.923. and solo around
> 0.921 Simple training with mixup on waves. 25 epochs. CE Loss but Sigmoid at
> Infer Trained with train_audio and labelled soundscapes only. Without priors it
> achieves 0.917 and with priors -> 0.923. Nothing fancy, simple model training
> setup. Another thing I noticed, for melspecs, MBConvs from EfficientNets are
> not learning that well, compared to other architectures.
>
> WITHOUT any Perch KD

### Boredom (7th) — Posted 2 days ago

> Interesting findings. In your experiments, did EfficientNet underperform
> compared to other architectures? Also, may I ask what you mean by "Priors"
> here? In my experiments, a single EfficientNet model can easily achieve an LB
> score above 925+.

### Salman Ahmed (448th) — Posted 2 days ago

> Yes, for me Efficientnet NS (with MBConv Architecture) were not converging fast
> and scored much lower on LB, compared to other models. Not sure why. By Priors
> I mean, prior table of species based on site and hours.

### To be happy (920th) — Posted today

> How is 'CrossEntropyLoss on clipwise and framewise max' used in this
> competition? I'm having trouble visualizing the process.

### Jiacheng Ma (642nd) — Posted 5 days ago

> You only use train_audio? no train_soundscape?

### Boredom (7th) — Posted 2 days ago

> train_audio + train_soundscape (w/ label), w/o pseudo label

### Boredom (7th) — Posted 2 days ago

> I guess without using unlabeled data, the highest LB (leaderboard score) we can
> achieve is around 0.94-0.945.

### !!!!! (912th) — Posted 2 days ago

> May I ask you use Perch or just timm-based model? For me, I get low score (0.88)
> using pretrained timm model.

### EliKal (464th) — Posted today ⭐ NEW

> EfficientNet-B0 – 0.909
> EfficientNet-B3 – 0.906
> DINOv3-small – 0.906
> ConvNeXt-small – 0.907
> All models were trained using the same pipeline for 12 epochs (after 10 epochs
> of distillation with KLD loss), with mixup applied to spectrograms. After
> reading the comments, I'm considering replacing this with mixup on raw waveforms.
> Inference was performed using only time-shift TTA.

### Don Mathis (1007th) — Posted 3 days ago

> Mine is only 0.880, fine-tuning an imagenet-pretrained CNN backbone that takes
> spectrogram input. The backbone outputs low-temporal resolution BCT (T=10,
> C=1024), that I then map to 234 and pool with logit-LSE to get clip logits,
> trained with BCE loss. The .880 isn't really replicable, more like .875 only.
> Trained on train_audio and the labeled soundscapes. Using 5-second clips.
>
> I tried lots of mixing schemes with the unlabeled soundscapes but that didn't
> transfer to the LB for me. I can't imagine how it's possible to get something
> like .922 using the same data I'm using.
>
> I also trained a 4-layer residual mamba model from scratch (no pretrained network
> at all), but that topped out at .772.

### hengck23 (175th) — Posted 3 days ago

> "The .880 isn't really replicable, more like .875 only." this is due to small
> sample size in some class. bird clef is domain and long tail problem. the issue
> can be solved by better validation and training methods. Hence the potential of
> timm and "train from pretrained CNN/scratch" can be higher than you think.
>
> although "train from pretrained CNN/scratch" can achieve good results, using
> perchv2 may be faster and easier.

### hengck23 (175th) — Posted 1 day ago

> how to use CNN only (no perch2, no post processing) to get 0.896 lb
> context and augmentation is the key. here we cheat by using CNN and long input
> wave to capture temporal information
>
> hint: how to get even better results: multi-conext head

**Attached image** (`notes/figures/hengck23-cnn-birdclef.png`):

Architecture: ONE backbone pass on 15s spectrogram, then SPLIT temporal features
into 3 chunks at the head level. Each chunk gets its own LSEHead prediction.

- Single 15s spectrogram → HGNetV2-B0 backbone → feature map (B, C, F', T')
- Split T' into 3 chunks of T'/3 → apply LSEHead to each → 3 predictions
- All 3 supervised during training, each with its own 5s label
- "conv does have location information" — backbone's receptive field spans
  across windows, so center 5s features are informed by flanking context
- LB 0.876 (5sec baseline) → **LB 0.896 (3×5sec) = +0.020 gain**, 4-fold, no PP
- Wave mix augmentation specially designed for 3×5sec input
- Training: 35k focal clips + 66 labeled soundscapes (same data as us)

**Key difference from Tattaka's approach**: hengck23 does NOT aggregate attention
over 15s. He keeps LSEHead and applies it per-chunk. The multi-context comes from
the backbone seeing 15s, not from the head aggregating over 15s.

### Tom (8th) — Posted 1 day ago

> This is similar to my current approach, but I made a slightly different approach
> which fits on 4x5sec.

### hengck23 (175th) — Posted 1 day ago

> I checked your Claude report. He actually knows this trick the very first day.
> That is impressive

---

## Key Takeaways

1. **0.920+ is achievable with single SED models** — no Perch, no pseudo-labels, no unlabeled data
2. **Salman's recipe evolution**: 0.908 (EfficientNet-B0, CE, waveform MixUp) → 0.921 (non-EfficientNet, same recipe + priors)
3. **Salman explicitly moved away from EfficientNet**: "MBConvs from EfficientNets are not learning that well"
4. **Boredom disagrees**: says EfficientNet can "easily" reach 925+ — recipe matters more than backbone
5. **Priors add +0.006** per Salman (0.917→0.923), we measured +0.003-0.004
6. **Waveform augmentations** (pink noise, time stretch) recommended by yukiZ from past winning solutions
7. **CE loss + sigmoid inference** is Salman's approach — but uses AttHead, not LSEHead
8. **Offline val/auroc is unreliable**: yukiZ's data shows correlation but with significant variance
9. **Maximum achievable without unlabeled data**: ~0.94-0.945 per Boredom (9th place)
10. **Salman says "Attn head's framewise max is better than LSE head"** — we use LSEHead, he uses AttHead
11. **hengck23: "context and augmentation is the key"** — 0.896 LB with CNN only (no Perch, no PP) using **long input wave** to capture temporal info. Hints at **multi-context head** for even better results.
12. **Tom (8th) confirms similar approach** — "fits on 4x5sec" (20s total context via multiple 5s windows)
13. **Multi-context (hengck23)** = ONE backbone pass on 15s, then SPLIT temporal features into 3 chunks, apply LSEHead per-chunk → 3 predictions. The backbone's convolutional receptive field provides cross-window context, but each prediction is still from a 5s chunk. NOT the same as Tattaka's train_period/infer_period crop (which aggregates attention over all 15s). hengck23 uses HGNetV2-B0 + LSE head + 4-fold → **+0.020 gain** (0.876→0.896).
14. **Our v5i (10s single window) got 0.880 LB (worse than v5j's 0.906 at 5s)** — different from multi-context because v5i pools over the full 10s. hengck23's approach preserves 5s pooling resolution per chunk while adding backbone-level context.
15. **Our v7 Phase D (AttHeadMultiContext) failed badly** (−0.054) because we changed the head AND aggregated over all 15s. The correct approach is: keep LSEHead, just feed longer input and split at head level.
16. **EliKal: Knowledge distillation (KLD loss) is a new lever** — 10 epochs distillation + 12 epochs finetune achieves 0.909 with EfficientNet-B0. This is notable because (a) it matches our best v5j (0.906) using a backbone we found weaker (EffNet-B0), (b) multiple architectures (EffNet-B0/B3, DINOv3-small, ConvNeXt-small) all land at 0.906-0.909 with the same pipeline, suggesting the distillation teacher is the key ingredient, not the student backbone. Uses spectrogram MixUp and time-shift TTA at inference.
17. **Time-shift TTA** — EliKal uses "only time-shift TTA" at inference. We have no TTA at all. This could be a low-effort inference-time improvement.
