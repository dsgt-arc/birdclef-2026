# Kaggle Discussion: What is your best single model LB score?

- **URL**: https://www.kaggle.com/competitions/birdclef-2026/discussion/683791
- **Posted**: 2026-03-21 (Antoine Masq)
- **Captured**: 2026-04-04
- **22 comments**

---

## Original Post

**Antoine Masq** (796th) — Posted 13 days ago

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
| **Boredom** | 9th | **0.922** single/single-fold | EfficientNet, "easily 925+" |
| **Salman Ahmed** | 425th | **0.921** solo, **0.923** w/ priors | 2 SED ensemble, CE+sigmoid, waveform MixUp |
| **Ali Ozan Memetoglu** | 2nd | **0.925** single model | No unlabeled data, no details |
| **hengck23** | 154th | **0.896** CNN only | Long input wave, augmentation, no Perch, no PP |
| **Tom** | 6th | ~similar to hengck23 | "slightly different approach which fits on 4x5sec" |
| **yukiZ** | 65th | **0.904** (fold 0 only, 5-fold SKF) | Raw waveform augmentations |
| Antoine Masq | 802nd | 0.889 | EfficientNetV2-B0 |
| Don Mathis | 988th | 0.880 | CNN + logit-LSE, BCE, 5s clips |

---

## Comments (chronological)

### Salman Ahmed (403rd) — Posted 13 days ago

> My best single model is SED Efficientnet B0 (Trained on 10 sec). Mixup on Raw
> Waves, and CrossEntropyLoss on clipwise and framewise max. Trained for 20+ epochs.
> This single model scores 0.908
>
> I have a ensemble of these 2 models and it scores 0.914
>
> No other augmentation. No unlabeled data.

### Ali Ozan Memetoglu (2nd) — Posted 13 days ago

> Mine is at 0.925 without using the unlabeled data. I couldn't make good use of
> it, and it might not even be that useful in this competition. But I might be wrong.

### Tom (7th) — Posted 13 days ago

> I think the floor of top scores would be very high in the end

### hengck23 (144th) — Posted 12 days ago

> "and it might not even be that useful in this competition. But I might be wrong."
>
> it should be very useful! in fact winning solution is how you make use of it

### yukiZ (262nd) — Posted 12 days ago

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

### Amit Aharoni (655th) — Posted 11 days ago

> Thanks for sharing. Based on the AUC score, is it safe to assume that your local
> validation is using train_audio and not train_soundscape?

### yukiZ (262nd) — Posted 11 days ago

> Yes, your way of thinking is correct.

### Boredom (9th) — Posted 12 days ago

> Single model / Single Fold: 0.922 without using the unlabeled/extra data

### vion (953rd) — Posted 8 days ago

> Amazing score considering that 0.940 is the maximum achievable with 206 of 234
> classes. Congrats!

### Murilo Gustineli (313th) — Posted 4 days ago

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

### Salman Ahmed (403rd) — Posted 4 days ago

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

### hengck23 (144th) — Posted 4 days ago

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

### yukiZ (262nd) — Posted 4 days ago

> Hi! Regarding the contribution of CV/LB in my experiment, data augmentation was
> effective.
>
> I believe many publicly available notebooks include Mixup. On the other hand,
> data augmentation on raw waveforms is not seen very often, but it has been
> frequently used in solutions from past BirdCLEF competitions. For example,
> "Pink Noise" and "Time Stretch".

### Murilo Gustineli (313th) — Posted 4 days ago

> Is it safe to assume that you're only training on train_audio or also using the
> train_soundscapes to achieve the 0.908 LB score?

### Salman Ahmed (403rd) — Posted 4 days ago

> Yup. Train audio + label soundscapes.

### Jiacheng Ma (596th) — Posted 4 days ago

> What are label soundscapes and unlabel soundscapes? Thanks.

### Salman Ahmed (403rd) — Posted 3 days ago

> train_soundscapes_labels.csv are labeled soundscapes.

### Salman Ahmed (425th) — Posted 1 day ago (UPDATED)

> Alright, Finally have 2 SED models with ensemble of LB 0.923. and solo around
> 0.921 Simple training with mixup on waves. 25 epochs. CE Loss but Sigmoid at
> Infer Trained with train_audio and labelled soundscapes only. Without priors it
> achieves 0.917 and with priors -> 0.923. Nothing fancy, simple model training
> setup. Another thing I noticed, for melspecs, MBConvs from EfficientNets are
> not learning that well, compared to other architectures.
>
> WITHOUT any Perch KD

### Boredom (9th) — Posted 1 day ago

> Interesting findings. In your experiments, did EfficientNet underperform
> compared to other architectures? Also, may I ask what you mean by "Priors"
> here? In my experiments, a single EfficientNet model can easily achieve an LB
> score above 925+.

### Salman Ahmed (425th) — Posted 1 day ago

> Yes, for me Efficientnet NS (with MBConv Architecture) were not converging fast
> and scored much lower on LB, compared to other models. Not sure why. By Priors
> I mean, prior table of species based on site and hours.

### Jiacheng Ma (621st) — Posted 4 days ago

> You only use train_audio? no train_soundscape?

### Boredom (9th) — Posted 1 day ago

> train_audio + train_soundscape (w/ label), w/o pseudo label

### Boredom (9th) — Posted 1 day ago

> I guess without using unlabeled data, the highest LB (leaderboard score) we can
> achieve is around 0.94-0.945.

### !!!!! (956th) — Posted 1 day ago

> May I ask you use Perch or just timm-based model? For me, I get low score (0.88)
> using pretrained timm model.

### Don Mathis (988th) — Posted 2 days ago

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

### hengck23 (154th) — Posted 2 days ago

> "The .880 isn't really replicable, more like .875 only." this is due to small
> sample size in some class. bird clef is domain and long tail problem. the issue
> can be solved by better validation and training methods. Hence the potential of
> timm and "train from pretrained CNN/scratch" can be higher than you think.
>
> although "train from pretrained CNN/scratch" can achieve good results, using
> perchv2 may be faster and easier.

### hengck23 (154th) — Posted 9 hours ago ⭐ NEW

> how to use CNN only (no perch2, no post processing) to get 0.896 lb
> context and augmentation is the key. here we cheat by using CNN and long input
> wave to capture temporal information
>
> hint: how to get even better results: multi-conext head

**Attached image shows:**
- 3×5sec approach: 3 adjacent 5s windows as longer input wave (15s total),
  providing 5s before and 5s after the center 5s window for temporal context
- Wave mix augmentation designed specifically for the 3×5sec multi-window input
- HGNetV2-B0 + LSE head with 4-folds
- 3 input windows → 3 output predictions (all supervised)
- This matches BirdCLEF 2023 5th place (Tattaka) train_period=15s / infer_period=5s

### Tom (6th) — Posted 9 hours ago ⭐ NEW

> This is similar to my current approach, but I made a slightly different approach
> which fits on 4x5sec.

### hengck23 (154th) — Posted 8 hours ago ⭐ NEW

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
12. **Tom (6th) confirms similar approach** — "fits on 4x5sec" (20s total context via multiple 5s windows)
13. **Multi-context head** = train on long windows (3×5s=15s or 4×5s=20s), predict on center 5s only. Backbone processes full spectrogram, head crops to center at inference. This is the BirdCLEF 2023 5th place (Tattaka) `train_period/infer_period` mechanism. hengck23 uses HGNetV2-B0 + LSE head (our exact v5j setup) — the multi-context is the key difference, not the architecture.
14. **Our v5i (10s single window) got 0.880 LB (worse than v5j's 0.906 at 5s)** — but multi-context is different: it preserves 5s resolution per window while adding temporal context, rather than processing one long spectrogram at reduced temporal density.
