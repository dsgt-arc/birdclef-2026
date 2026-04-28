# Kaggle Discussion: What is your best single model LB score?

- **URL**: https://www.kaggle.com/competitions/birdclef-2026/discussion/683791
- **Posted**: 2026-03-21 (Antoine Masq)
- **Captured**: 2026-04-28 (updated)
- **55+ comments** (2 marked "This comment has been deleted." in latest pull)

---

## Original Post

**Antoine Masq** (396th) — Posted 18 days ago

> Hi everyone,
>
> As most of us are probably still improving our baseline models, I was curious to
> see what your best LB score is for a single model without using the unlabeled data.
>
> So far, my best single model performance is 0.889 with an EfficientNetv2_b0 backbone

---

## Key Competitor Scores

Ranks reflect latest pull (2026-04-28); some have shifted since prior captures.
"DM" = D.M. = Don Mathis (same person; Kaggle username changed).

| Person | Rank | LB | Details |
|--------|------|-----|---------|
| **Starry** | **2nd** | — | (asked Boredom about 0.950+ unlabeled flow) |
| **Boredom** | **3rd** | **0.950+** w/ unlabeled, **0.938-0.939** w/o | Non-Perch, timm+SED. ~11min inference. 1st iter on unlabeled. **No TTA**; "simple PP" worth ~+0.004 (corrected from 0.03-0.04 typo). |
| **Tom** | 10th | ~similar to hengck23 | "slightly different approach which fits on 4x5sec" |
| **Ali Ozan Memetoglu** | 13th | **0.941** single model | Uses BirdCLEF 2025 pretrained checkpoint (XC-filtered, 2-stage). Pretrain → no folds. |
| **Salman Ahmed** | **19th** | **0.937** solo v2B0, **0.932** ens of 4 | **Modeling breakthrough** — attention investigation. B0_NS now matching v2B0. Uses PP/TTA. **NEW**: 20s segments, ~50-60min inference. |
| **tennogh** | 62nd | — | ~0.02 diff between backbones; KD reduces diversity (cos sim 0.5→0.67) |
| **Anil Ozturk** | 67th | 0.879 | GKF on sites OOF=0.608 (extreme CV/LB disconnect) |
| **Jake** | **102nd** | **0.915 w/o unlabeled, 0.922 w/ unlabeled** | NEW (2026-04-28). Solo model; small ~0.007 lift from unlabeled. |
| **Jack** | 126th | — | Confirms Perch KD causes backbone stability |
| **yukiZ** | 173rd | **0.904** (fold 0 only, 5-fold SKF) | Raw waveform augmentations |
| **Tucker Arrants** | **248th** | **0.93+ NEW** w/ PP (5-fold ensemble), **0.922** w/o PP | **NEW (2026-04-28)**: Perch-distilled B0 + 5s clips + BCE + waveform augs + site/hour priors + Gaussian smooth (5 folds). Also: 0.925 w/o deep learning (frozen Perch + clustering + metadata), 0.91 w/ Perch+GRU. |
| **Aic** | 266th | — | Asks about LB stability; soundscape AUC closely aligned with LB |
| **OpPrime** | 353rd | 0.826 5-fold / 0.835 fold0 | EfficientNetB0, 10ep/fold, 5s clips |
| **Murilo Gustineli** | 484th | **0.906** (HGNet+LSEHead) | No KD, no Perch in SED |
| **Gaurav Rawat** | 493rd | **0.926** LB | GroupK on site CV, shows poor CV-LB correlation |
| **hengck23** | 706th | **0.896** CNN only | Long input wave, multi-context head, no Perch, no PP |
| **!!!!!** | 851st | 0.88 | Pretrained timm |
| **EliKal** | 1072nd | **0.909** single (EffNet-B0) | Perch KD (KLD loss) + 12ep finetune, time-shift TTA |
| **MengYe** | 209th | — | Asks Salman about CV/LB correlation |
| **D.M. (Don Mathis)** | 1691st | 0.880 (~0.875 reproducible) | CNN + logit-LSE, BCE, 5s clips. **Pain point**: ".921 with mixup on waves + train_audio + labelled SS — I get .865 doing same." |
| **XuKong Ji** | 1383rd | **0.899** single fold | EffNet-B0+LSEHead+BCE, 5s. 5-fold=0.889 (worse!) |
| Antoine Masq | 46th | 0.889 | EfficientNetV2-B0 |

---

## Comments (chronological, oldest first)

### Salman Ahmed (562nd) — Posted 18 days ago

> My best single model is SED Efficientnet B0 (Trained on 10 sec). Mixup on Raw
> Waves, and CrossEntropyLoss on clipwise and framewise max. Trained for 20+ epochs.
> This single model scores 0.908
>
> I have a ensemble of these 2 models and it scores 0.914
>
> No other augmentation. No unlabeled data.

### Ali Ozan Memetoglu (4th) — Posted 18 days ago

> Mine is at 0.925 without using the unlabeled data. I couldn't make good use of
> it, and it might not even be that useful in this competition. But I might be wrong.

### Tom (6th) — Posted 18 days ago

> I think the floor of top scores would be very high in the end

### hengck23 (261st) — Posted 18 days ago

> "and it might not even be that useful in this competition. But I might be wrong."
>
> it should be very useful! in fact winning solution is how you make use of it

### yukiZ (45th) — Posted 18 days ago

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

### Amit Aharoni (798th) — Posted 17 days ago

> Thanks for sharing. Based on the AUC score, is it safe to assume that your local
> validation is using train_audio and not train_soundscape?

### yukiZ (45th) — Posted 16 days ago

> Yes, your way of thinking is correct.

### Boredom (1st) — Posted 18 days ago

> Single model / Single Fold: 0.922 without using the unlabeled/extra data

### vion (1101st) — Posted 14 days ago

> Amazing score considering that 0.940 is the maximum achievable with 206 of 234
> classes. Congrats!

### Murilo Gustineli (111th) — Posted 10 days ago

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

### Murilo Gustineli (111th) — Posted 10 days ago

> Is it safe to assume that you're only training on train_audio or also using the
> train_soundscapes to achieve the 0.908 LB score?

### Salman Ahmed (562nd) — Posted 9 days ago

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

### Salman Ahmed (562nd) — Posted 9 days ago

> Yup. Train audio + label soundscapes.

### hengck23 (261st) — Posted 9 days ago

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

### yukiZ (45th) — Posted 9 days ago

> Hi! Regarding the contribution of CV/LB in my experiment, data augmentation was
> effective.
>
> I believe many publicly available notebooks include Mixup. On the other hand,
> data augmentation on raw waveforms is not seen very often, but it has been
> frequently used in solutions from past BirdCLEF competitions. For example,
> "Pink Noise" and "Time Stretch".

### Jiacheng Ma (771st) — Posted 9 days ago

> What are label soundscapes and unlabel soundscapes? Thanks.

### Salman Ahmed (562nd) — Posted 9 days ago

> train_soundscapes_labels.csv are labeled soundscapes.

### Don Mathis (1132nd) — Posted 7 days ago

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

### hengck23 (261st) — Posted 7 days ago

> "The .880 isn't really replicable, more like .875 only." this is due to small
> sample size in some class. bird clef is domain and long tail problem. the issue
> can be solved by better validation and training methods. Hence the potential of
> timm and "train from pretrained CNN/scratch" can be higher than you think.
>
> although "train from pretrained CNN/scratch" can achieve good results, using
> perchv2 may be faster and easier.

### Salman Ahmed (562nd) — Posted 6 days ago (UPDATED)

> Alright, Finally have 2 SED models with ensemble of LB 0.923. and solo around
> 0.921 Simple training with mixup on waves. 25 epochs. CE Loss but Sigmoid at
> Infer Trained with train_audio and labelled soundscapes only. Without priors it
> achieves 0.917 and with priors -> 0.923. Nothing fancy, simple model training
> setup. Another thing I noticed, for melspecs, MBConvs from EfficientNets are
> not learning that well, compared to other architectures.
>
> WITHOUT any Perch KD

### Boredom (1st) — Posted 6 days ago

> Interesting findings. In your experiments, did EfficientNet underperform
> compared to other architectures? Also, may I ask what you mean by "Priors"
> here? In my experiments, a single EfficientNet model can easily achieve an LB
> score above 925+.

### Salman Ahmed (562nd) — Posted 6 days ago

> Yes, for me Efficientnet NS (with MBConv Architecture) were not converging fast
> and scored much lower on LB, compared to other models. Not sure why. By Priors
> I mean, prior table of species based on site and hours.

### Boredom (1st) — Posted 6 days ago

> train_audio + train_soundscape (w/ label), w/o pseudo label

### Boredom (1st) — Posted 6 days ago

> I guess without using unlabeled data, the highest LB (leaderboard score) we can
> achieve is around 0.94-0.945.

### !!!!! (755th) — Posted 6 days ago

> May I ask you use Perch or just timm-based model? For me, I get low score (0.88)
> using pretrained timm model.

### hengck23 (261st) — Posted 5 days ago

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

### Tom (6th) — Posted 5 days ago

> This is similar to my current approach, but I made a slightly different approach
> which fits on 4x5sec.

### hengck23 (261st) — Posted 5 days ago

> I checked your Claude report. He actually knows this trick the very first day.
> That is impressive

### Don Mathis (1132nd) — Posted 4 days ago

> I sort of see the overall goal, but this is too sketchy for me. I don't know how
> many context_nets there are or what clip length they've been trained on, nor what
> the '3' is in the shape of prob3. I think the bottom loop is sliding width-4
> windows within a 12 timestep clip and averaging the last 2 dimensions of prob3 in
> each of the 12 cells. But then I still don't know what the 3 dimension is, nor
> what 'probs' is at the end.

### Murilo Gustineli (111th) — Posted 4 days ago

> Great score with a single EfficientNet! I'm struggling to get EfficientNet-B0 to
> go beyond 0.855 LB, even when trying multiple different heads and loss functions.
> Could you share what loss function you're using (BCE vs CE) and whether you're
> using an attention-based head or something simpler? Trying to understand what
> unlocks 925+ on EfficientNet specifically

### Murilo Gustineli (111th) — Posted 4 days ago

> Congrats on the 0.921! My best single model so far got 0.906 LB with
> HGNetV2-B0 + LSEHead + BCE. I tried switching to AttHead (with tanh attention,
> sigmoid-space clipwise, dual 50/50 supervision) but it got worse LB results
> both with BCE and CE loss. Two quick questions if you don't mind: (1) what
> backbone did you switch to from EfficientNet? and (2) are you using 5s or
> longer windows for training? Thanks!

### hengck23 (261st) — Posted 4 days ago

> i suppose your 0.906 comes from distillation. Comparing with your previous result,
> you would note that the improvement comes from distillation and not data.
> What to distill? logit? embedding or spatial embedding or all?
>
> if the results is from distillation, then maybe modeling is less important.
>
> if you look at public notebook which is just based on perch2 embedding, they are:
> lb 0.930 = (perch2 + proxy class + temporal + PCA) + prior + state space + perclass normalisation ….
>
> you can estimate
> lb 0.92+ = (perch2 + proxy class + temporal + PCA)
>
> this means to get lb 0.92+ from timm, you need to add temporal

### Murilo Gustineli (111th) — Posted 4 days ago

> I'm not using knowledge distillation at all actually. The 0.906 is just
> HGNetV2-B0 trained from ImageNet init on focal + labeled soundscapes with BCE
> loss. No Perch at all.
>
> Really interesting point about temporal. I've tried your 3x5s multi-context
> approach (backbone sees 15s, split into 3 chunks, LSEHead per-chunk) and got
> best-ever offline val metrics, but it regressed on LB (0.896 with 4×5s avg, vs
> 0.906 baseline). What exactly do you mean by "add temporal"? is it the
> multi-context you showed, or something else like temporal embeddings or sequence
> modeling?

### EliKal (579th) — Posted 4 days ago

> EfficientNet-B0 – 0.909
> EfficientNet-B3 – 0.906
> DINOv3-small – 0.906
> ConvNeXt-small – 0.907
> All models were trained using the same pipeline for 12 epochs (after 10 epochs
> of distillation with KLD loss), with mixup applied to spectrograms. After
> reading the comments, I'm considering replacing this with mixup on raw waveforms.
> Inference was performed using only time-shift TTA.

### tennogh (23rd) — Posted 4 days ago

> Interesting that you have so little difference between backbones. With my pipeline
> I have up to .02 difference between model families so far.

### Jack (40th) — Posted 1 day ago

> It's the distillation (assuming he's distilling perch) causing that stability. I
> experience the same swings as you with different backbones

### EliKal (579th) — Posted 1 day ago

> Correct. Perch embeddings for all the architectures. Did you solve this stability
> "issue"?

### tennogh (23rd) — Posted 1 day ago

> Not sure if this is an issue, I guess it could make the models more similar and
> ensembling less beneficial. I'll check when I have a stable KD pipeline.

### To be happy (1045th) — Posted 4 days ago

> How is 'CrossEntropyLoss on clipwise and framewise max' used in this
> competition? I'm having trouble visualizing the process.

### Aic (597th) — Posted 1 day ago

> May I ask, do you experience any score fluctuations in your experiments due to
> different numbers of epochs or changes in other hyperparameters?…it's been giving
> me a headache lately.

### Gaurav Rawat (411th) — Posted 2 days ago ⭐ NEW (CV/LB DATA)

> - 0.82 cv lb 908
> - 0.94 cv lb 913
> - 0.97 cv lb 926
> - 0.87 cv lb 916 :/
> - 0.942 cv lb 915

### Aic (597th) — Posted 2 days ago

> May I ask about your CV strategy? Mine is quite unreliable, even though my
> SS_AUC is closely aligned with the LB.

### Gaurav Rawat (411th) — Posted 2 days ago

> pretty simple groupk on site

### Anil Ozturk (1028th) — Posted 4 hours ago ⭐ NEW (CRITICAL CV DATA)

> Soundscape-val-only GKF on sites OOF: 0.60775 LB: 0.879
>
> lol

### Tucker Arrants (966th) — Posted 19 minutes ago ⭐ NEW (CV INSIGHT)

> The GKF is overly "harsh" for this competition since there is overlap between
> the train and test sites and GKF does not allow the model to take advantage of
> this. But without GKF, the model does not learn to generalize to unseen sites -
> its a tricky problem.
>
> You can try this: train on everything but have a joint validation set - one that
> measures within-site generalizations and one that measures site-level
> generalizations. You can then set up model checkpoints on this combined metric
> and monitor them to see if you model is overfitting to site-specific patterns or
> if it is learning general features without exploting site specific patterns.

### hengck23 (261st) — Posted 13 hours ago ⭐ NEW (DIAGNOSTICS POST)

> why some people get lb0.87+ and why get as high as 0.91+ for simple CNN model?
> (exclude effects, pseudo-labels, extreme tricks … we focus on base model performance)
>
> the reason are:
> - long tail (some class has few train samples)
> - domain shift. focal audio: one species, high snr soundscape: multiple species,
>   low snr (background noise, bird far from mic)
>
> how to solve:
> - long tail: pretrained model, sampling, more information per sample
>   (e.g. 10,15 sec instead of 5 sec), regularization
> - domain shift: augmentation (others: using unlabelled, external data)
>
> what should you monitor
> - auc: sensitive to few samples
> - bce: can skews towards low sample (bce hacking … like reward hacking)
> - logit std: monitor this as a "proxy to logit distribution"
>
> how a robust model would look like?
> - agreement for train and validation
> - lower logit std
> - low bce
> - high auc (split into auc for normal, rare, problematic classes etc)
> - can train for very long epoch (if augmentation is good). "Good grad descent at high lr"
> - low auc variance in different folds
> - flat loss valley for low sample class
>
> how an overfit model would look like?
> - disagreement for train and validation
> - high train logit std (growing as training iteration progress)
> - low train bce
> - high train auc
>
> finally, cut and paste this post into your favourite AI llm and ask for comment.
> If useful, add to the skills in agent.

### hengck23 (364th) — Posted 5 days ago

> how to learn to be an expert in modelling. distill ChatGPT knowledge to yourself.
>
> instead of asking chatgpt to "improve the model", …
> Pick a point from the above post, e.g. "flat loss valley for low sample class"
> ask chatgpt:
> - how to measure flatness. how to prove current model methods is flat or not
> - suggest methods to improve flatness
> - implement and carry experiments
> - verify flatness is improved and lead to better train,validation and public LB.
> - if it does not, ask chatgpt why and most importantly how to prove?
>
> "turning insights into measurable, testable pipelines"

### Salman Ahmed (17th) — Posted 2 days ago ⭐ NEW (MAJOR UPDATE)

> As of April 11. Updated a couple of things in my train settings.
>
> Trained on 20 sec with Train Audio + Labelled Soundscapes.
>
> No KD. No Claude for now. Just wanted to solve it / write it myself
> after a longtime relying on claude. [Just for the fun of it].
>
> No Prior or additional features.
>
> Ensemble of 4 scores 0.932.
>
> Individual scores around 0.922.
>
> 10 seconds / 15 seconds scores almost the same for me / sometimes better
> [Didn't include those in my ensemble].
>
> Will play with some cool attention for next week.

### Salman Ahmed (17th) — Posted a day ago

> xD. Yes it's trained on 20 seconds audio segments. Not the split and max logits.

### Salman Ahmed (17th) — Posted 12 hours ago

> Yes. Solo B0 was around 0.922.

### Salman Ahmed (17th) — Posted 7 hours ago ⭐ NEW (EXTERNAL DATA)

> The data I am using is used in solutions for previous competitions.
> [...] Background Noise (nocall in 2020, 2021 comp + rainforest +
> environment sound + nocall in freefield1010, warblrb, birdvox)

### Salman Ahmed (17th) — Posted 7 hours ago

> btw, I think I get same LB without any other data than this competition's.
> But I think using that previous competition's data, might be helpful to
> make my model generalize better.

### Boredom (1st) — Posted 8 hours ago ⭐ NEW (SINGLE MODEL 0.947+)

> update: Single model: 0.947+ about 13min

### Boredom (1st) — Posted 8 hours ago

> Yes, absolutely. I would be very surprised if someone could achieve above
> 0.945 with a single model without using unlabeled data😭. My best single
> model without unlabeled data is around 0.937-0.938.

### Boredom (1st) — Posted 7 hours ago

> non-Perch, maybe Perch can do better.

### Ali Ozan Memetoglu (3rd) — Posted 7 hours ago ⭐ NEW

> My single best model is 0.941 but I use last year's pretrained
> checkpoint so I believe it is possible :D

### Kurise (4th) — Posted 5 hours ago

> Not sure whether I'm right, using last year's checkpoint just like
> 2 stage training(pretrain on 25, retrain on 26), but there is no extra
> computation cost. So that we can get a model trained on both 25 and 26 data.

### XuKong Ji (1057th) — Posted 8 hours ago ⭐ NEW (5-FOLD REGRESSION)

> I have a single fold(one of 5folds) model(effecient b0+LSEHead + BCE)
> score 0.899 with 5s input/time smooth/time shift tta(shift 0s and 2.5s).
> I got something weird:
> - 5fold ensemble score 0.889;
> - After increasing input to 10s, score drop to 0.86(1fold)…

### Tucker Arrants (311th) — Posted a day ago ⭐ NEW

> Frozen Perch embeddings + learned pool + GRU
> 5 folds, 3x time-shift TTA
> 0.91 public leaderboard

### tennogh (48th) — Posted 18 days ago (UPDATED)

> Edit: with KD and 4 different backbones, I still have a ~0.02 difference
> between the worst and the best (the ranking is a little bit different too).
> KD also reduces diversity to an extent (cosine similarity ~0.5->~0.67),
> to the point where is might be worth ensembling KD and non-KD models.

### Mahog (1300th) — Posted 8 days ago

> Have you managed to fix this? I'm having the same issue, even a seed change
> shifts LB by 0.005 :V

### goevian (486th) — Posted 9 days ago

> may i ask, what do you mean by "multi-context head"?

### Salman Ahmed (13th) — Posted 3 days ago ⭐ NEW (MAJOR: 0.937 SINGLE MODEL)

> Finally. Single model LB 0.937 without using Unlabeled data. Recently started
> paying attention to the modeling part in this competition, Apparently, it was
> not that important in previous competitions, but in this one, for me it is one
> of the most important part. I have been investigating attention values on some
> examples with highest loss during training, and minimum loss during training,
> and now it's much better compared to previous models I had.

### Leon (1336th) — Posted 3 days ago

> Wow that is amazing! May I ask is it still efficientnet b0 ns?

### Salman Ahmed (13th) — Posted 3 days ago

> xD LB 0.937 is with Efficientnet v2B0 because it's faster to train compared
> to others. Will try other variants (B0 NS / mixnet / nfnet etc) soon.

### Seeing Times (1530th) — Posted 3 days ago

> an amazing score, May I ask did you change sed head?

### Leon (1336th) — Posted 3 days ago

> I think you mentioned before b0 MBConv doesn't learn well, and ebv2 is using
> Fused-MBConv, might be reason why its better (not sure). and may I also ask
> if your b0 model (0.922) is with any post processing (I assume only model soup)?

### Salman Ahmed (13th) — Posted 3 days ago

> xD No worries, For that previous model architecture / training strategy MBConvs
> were not learning, but now B0_NS is matching or performing even better than v2B0.
> Yeah, I am not using Model Soup this time, simple model checkpoint.

### Shiro (1501st) — Posted 2 days ago

> Impressive results, with efficientnetb1 I am not able to reach that results
> with single models, trying different architectures. May I ask if you include
> postprocessing in the inference step or is it just a vanilla prediction on
> 5 seconds?

### Salman Ahmed (13th) — Posted 2 days ago

> Hi @ludovick Yes, I definitely rely on some post processing / TTA.

### Ali Ozan Memetoglu (6th) — Posted 2 days ago

> Thank you :)) Are you using the soundscapes too or only this?

### Salman Ahmed (13th) — Posted 2 days ago ⭐ NEW (SOUNDSCAPE TRAINING FAILURE)

> Just Train Audios. Just started experimenting with Soundscapes, and for now,
> adding Soundscapes reduce my LB to around 0.85.

### Salman Ahmed (13th) — Posted 2 days ago

> As I was typing this, i remember something, I think there is a bug in
> Soundscapes training. I'll fix it soon.

### Ali Ozan Memetoglu (6th) — Posted a day ago

> No, I mean for the background noise I was only using soundscapes so I'm
> thinking of combining them with the dataset you provided

### Salman Ahmed (13th) — Posted 12 hours ago

> Ah I see. No I don't use the SS as background noise, because there are
> species in those SS recordings.

### MengYe (171st) — Posted 5 days ago

> does your cv correlate with lb?

### Aic (178th) — Posted 15 days ago

> May I ask, do you experience any score fluctuations in your experiments due to
> different numbers of epochs or changes in other hyperparameters?...it's been
> giving me a headache lately.

### OpPrime (242nd) — Posted 11 days ago

> I got 0.826 on EfficientNetB0 with 5-fold training, 0.835 on fold 0, 10 epochs
> per fold, feeding in 5 seconds clips only. Following Salman's post below, I then
> got a 0.801 from SED HGNetV2 feeding it 20s of mel with a temporal attention head.

### EliKal (928th) — Posted 11 days ago

> I'd start by focusing on your 5s models. You can get 0.915+ with an ensemble
> of just two models. For 20s inputs, I'd try adding segment-wise labels from
> the labeled soundscapes.

### Tucker Arrants (164th) — Posted 12 days ago (UPDATED)

> Frozen Perch embeddings + learned pool + GRU
> 5 folds, 3x time-shift TTA
> 0.91 public leaderboard
>
> Edit: I was able to score 0.925 without any deep learning, just some simple
> clustering on the frozen perch embeddings and metadata engineering. The learned
> pool + GRU was total over kill.

### Ali Ozan Memetoglu (6th) — Posted 10 days ago ⭐ NEW (XC PRETRAIN DETAILS)

> They used the entire Xeno-Canto dataset by filtering it and fixing a bug during
> pretraining, but I didn't use anything beyond the checkpoint.

### Cody_Null — Posted 9 days ago

> Im not super excited about the idea of doing all these models 5 fold on the
> entire XC dataset but we gotta do what we gotta do

### Ali Ozan Memetoglu (6th) — Posted 8 days ago

> it is pretraining, so there are no folds

### Boredom (2nd) — Posted 3 days ago ⭐ NEW (SINGLE MODEL 0.950+)

> Update: Single model LB 0.950+ (11 minutes). After reading Salman's latest
> post, I realized I might have been constrained by my experience from past
> competitions, which may have limited the optimal single-model performance of
> this pipeline to around 0.953-0.955. I will soon build the next pipeline from
> scratch for ensembling, hoping to push even higher.

### Starry (1st) — Posted 17 hours ago

> Hi Boredom, it is an amazing result, may I ask if you use unlabeled
> train_soundscapes in your 0.950+ model. And if so, what the score with only
> train_audio + labeled train_soundscapes?

### Boredom (2nd) — Posted 10 hours ago

> Yep! this is the result of the first iteration on unlabeled data. Maybe one or
> two more iterations are possible. A single model using only labeled data achieves
> an LB of 0.938-0.939. I've never been able to break 0.94 with a single model.
> Compared to others with solid work, there might still be a significant gap in
> potential.

### Gaurav Rawat (370th) — Posted 16 hours ago

> nice SED or vision model?

### Boredom (2nd) — Posted 10 hours ago

> timm backbone+SED

---

## Comments — 2026-04-28 update (NEW since 04-23 capture)

### Tucker Arrants (248th) — Posted ~1 day ago ⭐ NEW (FIRST 0.93+ NON-PERCH-FREE PUBLIC RECIPE)

> Finally broke 0.93 with a single model only using focal recordings and labeled
> soundscapes.
>
> Perch distilled B0 trained on 5s clips. Simple BCE loss, waveform augmentations,
> and moderate post processing (site/hour priors + Gaussian smooth)
>
> Without any post processing it is 0.922

### Salman Ahmed (19th) — Posted ~1 day ago (asks Tucker)

> HI @tuckerarrants Did you use site/hour priors from labeled soundscapes only?
> Also is this a single model or 5 folds of the same model?

### Tucker Arrants (248th) — Posted ~1 day ago ⭐ NEW (TUCKER 0.93+ DETAILS)

> Labeled soundscapes only for the site/hour priors, for now…
>
> This is 5 folds of the same model, simple KFold on filenames and MultiLabelStrat
> on focal labels.

### Jake (102nd) — Posted ~1 day ago ⭐ NEW (0.915 SOLO)

> Best solo model 0.915 w/o using any of the unlabelled soundscape data.
> 0.922 using the unlabelled soundscape data.

### Boredom (3rd) — Posted ~4 days ago ⭐ NEW (NO TTA, PP CORRECTION)

> [Re: 0.950+ inference] no tta, 950+ lb include some simple pp, these can
> improve about 0.03-0.04
>
> [Then corrected:] sorry, 0.004. haha 😂

### D.M. (Don Mathis, 1691st) — Posted ~3 days ago ⭐ NEW (KEY PAIN POINT)

> How can cv correlate with LB? The training data are "clean" clips not embedded
> in soundscapes. Mixup of clean clips with unlabeled soundscapes doesn't work
> because there is unlabeled signal in the soundscapes. Mixup with labeled
> soundscapes helps got from .80 to .86 but there's not enough of them to go
> further. I'm stumped.

### D.M. (1691st) — Posted ~3 days ago ⭐ NEW (CAN'T REPRODUCE SALMAN'S 0.921)

> [Re: Salman 0.921] "Simple training with mixup on waves. Trained with
> train_audio and labelled soundscapes only." No clue how you are doing that.
> I get .865 when I do that :/

### Mahog (1466th) — Posted ~14 days ago (under Salman 20s reply, MISSED IN 04-23 CAPTURE)

> Does training on longer audio segments increase training time by a lot for
> you? 20 second segments should in theory increase training time by 4x right?

### Salman Ahmed (19th) — Posted ~17 days ago ⭐ NEW (RUNTIME)

> Around 50 to 60 minutes I think.

### Jiacheng Ma (1273rd) — Posted ~11 days ago

> Great！It's still efficientnet_b0 or efficientnet_v2b0?

### lhwcv (1796th) — Posted ~15 days ago

> @salmanahmedtamu Hi Salman, are you still on Discord?

### Salman Ahmed (19th) — Posted ~15 days ago

> Hi @lihaoweicvch, Great to see you in this year's competition as well. Yes,
> but not too active there. 😅

### Shiro (1652nd) — Posted ~2 days ago

> [To Salman] thanks, do you know your LB on single model without postprocessing/TTA?

(no answer recorded)

### Shiro (1652nd) — Posted ~4 days ago (to Boredom)

> Impressive! does it include TTA/postprocessing during inference? Is there a big
> gap between with/without those postprocessing in term of LB score?

### MengYe (209th) — Posted ~10 days ago

> [To Salman] does your cv correlate with lb?

(Salman did not directly answer)

### Tucker Arrants (248th) — Posted ~19 days ago ⭐ NEW (UPDATED DUAL-VAL FRAMING)

Updated from earlier "joint validation" hint to a fully-specified proposal:

> GKF is overly harsh for this competition because the test set has site overlap
> with training. GKF models can't learn site-level species priors that would
> help at test time — they're forced to hold out entire sites during training,
> so they never see the "this site has these species" signal that exists in
> reality. But removing GKF entirely is also risky: the model might memorize
> site-specific artifacts that don't transfer, and you'd have no way to detect
> that the model relies on site memorization versus learning general features.
>
> But you can try **dual validation on a single training run**: train on
> everything, but hold out two disjoint val sets. **Val-A** holds out specific
> recordings from seen sites (tests the regime: new recordings at seen sites).
> **Val-B** holds out an entire site (tests whether the model has any ability
> to generalize to unseen sites as a hedge against test sets that include
> unseen sites). Combine them into a weighted metric for early stopping,
> weighted toward Val-A because it matches the test regime more directly.

### "This comment has been deleted." × 2 (preserved markers)

Two comments in the latest pull are marked deleted by author (no recoverable
content). Locations:
- Between hengck23 ("I checked your Claude report... very first day") and D.M.
  ("I sort of see the overall goal..."). Likely an interim back-and-forth.
- Under Salman's 0.908 single-model post. Original author + content unknown.

---

## Proven 0.93+ Recipes (synthesis, 2026-04-28)

Filtering everything down to **single-model recipes that have publicly hit ≥0.93 LB**:

### 1. Salman Ahmed — 0.937 v2B0 SED (no KD, no unlabeled)

| Component | Value |
|---|---|
| Backbone | EfficientNetV2-B0 (Fused-MBConv) — also B0_NS now matches |
| Head | "simple Attn head's framewise max is better than LSE head" |
| Loss | CE on clipwise + framewise max; **sigmoid at inference** |
| Window | 20s training segments (not split+max). 10s/15s "almost the same / sometimes better" |
| Mixup | Wave mixup |
| Epochs | 25; "more epochs make the model too confident" |
| Data | train_audio + labeled soundscapes (Salman says SS adds ~0 to LB; he sometimes only uses train_audio) |
| KD | None |
| External | BG noise from prior comps (nocall 2020/21, freefield1010, etc.) |
| PP / TTA | "I definitely rely on some post processing / TTA" |
| Soundscape-as-BGN | Does NOT use SS as BG noise (species in them) |
| Inference time | ~50-60 min |
| Breakthrough | "investigating attention values on highest/lowest loss examples" — modeling > tricks |

### 2. Tucker Arrants — 0.93+ Perch-distilled B0 SED (5-fold ensemble)

| Component | Value |
|---|---|
| Backbone | EfficientNet-B0 with **Perch distillation** |
| Window | 5s clips |
| Loss | BCE |
| Mixup | Waveform augmentations (no specifics; "simple") |
| Folds | **5-fold ensemble** (KFold on filenames + MultiLabelStrat on focal labels) |
| Data | Focal + labeled soundscapes only (no unlabeled) |
| PP | Site/hour priors (from labeled SS only) + Gaussian smooth — "moderate" |
| Without PP | 0.922 |
| With PP | 0.93+ |

### 3. Boredom — 0.938-0.939 timm+SED (no unlabeled), 0.950+ (with unlabeled, 1 iter)

| Component | Value |
|---|---|
| Backbone | timm + SED, non-Perch |
| Inference | ~11 min total; **no TTA** |
| PP | "simple PP" worth ~+0.004 (he initially said 0.03-0.04, corrected to 0.004) |
| Unlabeled | 1 iteration of pseudo-labels → 0.938-0.939 → 0.950+ |
| Pipeline ceiling | 0.953-0.955 estimated |

### 4. Ali Ozan Memetoglu — 0.941 (BirdCLEF 2025 pretrained checkpoint)

| Component | Value |
|---|---|
| Approach | 2-stage: pretrain on filtered Xeno-Canto (entire 2025 checkpoint), then fine-tune on 2026 |
| Pretrain | "It is pretraining, so there are no folds" — XC + bug fix |
| Fine-tune | Standard on 2026 data; only used the existing checkpoint, not the XC dataset |
| Architecture | Not specified in the thread |

### What 0.93+ recipes have in common

1. **Either** Perch KD (Tucker, EliKal etc. land 0.91+ with KD as the pipeline floor) **OR** in-domain pretrain (Ali Ozan via 2025 ckpt) **OR** modeling refinement on attention (Salman, "investigating attention values" → +0.015 LB jump 0.922→0.937)
2. **Simple post-processing**: site/hour priors + Gaussian smoothing (Tucker, Boredom). PP alone worth +0.005-0.010.
3. **5-fold ensemble OR a single-model breakthrough** — Tucker uses 5-fold; Salman/Boredom/Ali Ozan get there single-model
4. **Either CE or BCE** (Salman: CE+sigmoid-infer; Tucker: BCE)
5. **5s OR 20s window** — both work; 5s is dominant (Tucker, Boredom, EliKal). 10s is the "death zone" (XuKong Ji, OpPrime)
6. **NOT**: special head architecture, exotic loss (ASL, etc.), 3-channel mel, 15s multi-context as the silver bullet

### Where our 0.906 stalls vs 0.93+

| Lever they have / we don't | Estimated LB lift |
|---|---:|
| Perch knowledge distillation (KLD loss on Perch embeddings during training) | +0.005-0.020 |
| Site/hour priors at inference (from labeled SS) | +0.005-0.010 |
| 4 or 5-fold ensemble (proper KFold + ensembling) | +0.005-0.010 |
| 2025-pretrained checkpoint (Ali Ozan style, on filtered XC) | +0.030 (Ali Ozan 0.941 vs us 0.906) |
| Attention-quality investigation (Salman) | +0.015 (Salman 0.922→0.937) |

### Occam's-razor priorities for us

The simplest path to 0.93+ for a non-Perch SED today is **(5-fold ensemble) + (site/hour priors) + (Gaussian smooth)**. Tucker explicitly states this gets him from 0.922 → 0.93+. Our v5j-4fold-pp2 already exists at **LB 0.917** (per MODEL-RUNS), which is ≥0.917 from the same pattern. The +0.013 gap to Tucker's 0.93+ is most likely **Perch distillation** during training.

This argues exp 030's mixup-recipe lever is small relative to the unexplored levers (Perch KD, fold ensembling at scale, simple PP). The recipe-level deltas from exp 030 sit inside the seed-noise band on a single fold.

---

## Key Takeaways

### Top Scores & Unlabeled Data (Updated 2026-04-23)

1. **Boredom: 0.950+ single model WITH unlabeled data** — non-Perch, timm+SED. ~11min inference.
   First iteration on unlabeled data. Without unlabeled: 0.938-0.939. Now 2nd place (Starry is 1st).
   Says pipeline ceiling may be 0.953-0.955. Plans to build new pipeline from scratch for ensembling.
2. **Ali Ozan: 0.941 single model using BirdCLEF 2025 pretrained checkpoint** — 2-stage
   training. The checkpoint used XC dataset (filtered) for pretraining. No folds needed for pretrain.
3. **Salman: 0.937 solo EfficientNetV2-B0** — **major jump from 0.922 to 0.937 via "modeling"**.
   Investigated attention values on highest/lowest loss examples during training. B0_NS now
   matching or better than v2B0. Uses PP/TTA. Simple checkpoint (no model soup).
4. **Unlabeled data is THE differentiator for 0.94+** — Boredom: "never been able to break 0.94
   with a single model" using labeled data only. 1 iteration of unlabeled → 0.950+.

### Salman's 0.922→0.937 Breakthrough (NEW 2026-04-23)

5. **"Modeling" is the key, not just data/augmentation** — Salman says "recently started paying
   attention to the modeling part... for me it is one of the most important parts." This is a
   departure from previous advice ("nothing fancy, simple model training setup").
6. **Attention investigation on loss extremes** — "investigating attention values on some examples
   with highest loss during training, and minimum loss during training, and now it's much better."
   This suggests he debugged/tuned the attention head by examining what it attends to on hard
   vs easy examples, not just changing hyperparameters.
7. **EfficientNetV2-B0 (Fused-MBConv)** — 0.937 achieved with v2B0. Previously said MBConvs
   "not learning well" but that was architecture/training-strategy-specific. With current strategy,
   B0_NS is "matching or performing even better than v2B0."
8. **No model soup** — simple checkpoint selection, not weight averaging.
9. **Post-processing + TTA** — "I definitely rely on some post processing / TTA." Not vanilla.
10. **Soundscape training currently hurts Salman** — "adding Soundscapes reduce my LB to around
    0.85" but he suspects a bug. Does NOT use soundscapes as background noise because "there are
    species in those SS recordings."

### Architecture & Training

11. **Salman was on EfficientNet-B0 at 0.922, now EfficientNetV2-B0 at 0.937** — v2B0 is "faster
    to train." Plans to try B0 NS / mixnet / nfnet variants.
12. **20sec full-audio training** (not split+max) was Salman's 0.922 approach. 10s/15s "almost
    the same / sometimes better" — he only uses 20sec in his ensemble.
13. **CE loss + sigmoid inference** remains Salman's loss approach
14. **Waveform augmentations** (pink noise, time stretch) recommended by yukiZ
15. **Offline val/auroc is unreliable**: yukiZ's data shows correlation but with significant variance
16. **Seed sensitivity**: Mahog reports seed changes shift LB by 0.005

### 5-Fold Regression

17. **XuKong Ji: 5-fold ensemble (0.889) WORSE than single fold (0.899)** — with EffNet-B0 +
    LSEHead + BCE. Also confirms 10s input drops to 0.86 (matches our v5i experience).
    This validates that naive ensembling can hurt, and 5s is the right window size.

### CV/LB Correlation

18. **Gaurav Rawat's CV/LB data (GroupKFold on site)** shows poor correlation:
   - 0.82 CV → 0.908 LB
   - 0.94 CV → 0.913 LB
   - 0.97 CV → 0.926 LB
   - 0.87 CV → 0.916 LB (worse CV, better LB than 0.82→0.908)
   - 0.942 CV → 0.915 LB
   The CV values are all over the place relative to LB ranking. Not monotonic.

19. **Anil Ozturk: soundscape-val-only GKF on sites** — OOF 0.608, LB 0.879. The soundscape-only
    CV score is extremely low and not predictive of LB at all.

20. **Tucker Arrants explains why**: GKF is "overly harsh" because **train and test sites overlap**.
    GKF forces the model to generalize to *unseen* sites, but the test set includes *seen* sites too.
    Proposes **joint validation**: one metric for within-site generalization + one for site-level
    generalization, with combined checkpoint selection.

### Multi-Context & Architecture

21. **hengck23: "context and augmentation is the key"** — 0.896 LB with CNN only (no Perch, no PP) using **long input wave** to capture temporal info. Hints at **multi-context head** for even better results.
22. **Tom (4th) confirms similar approach** — "fits on 4x5sec" (20s total context via multiple 5s windows)
23. **Multi-context (hengck23)** = ONE backbone pass on 15s, then SPLIT temporal features into 3 chunks, apply LSEHead per-chunk → 3 predictions. The backbone's convolutional receptive field provides cross-window context, but each prediction is still from a 5s chunk. NOT the same as Tattaka's train_period/infer_period crop (which aggregates attention over all 15s). hengck23 uses HGNetV2-B0 + LSE head + 4-fold → **+0.020 gain** (0.876→0.896).
24. **Our v5i (10s single window) got 0.880 LB (worse than v5j's 0.906 at 5s)** — different from multi-context because v5i pools over the full 10s. hengck23's approach preserves 5s pooling resolution per chunk while adding backbone-level context.
25. **Our v7 Phase D (AttHeadMultiContext) failed badly** (−0.054) because we changed the head AND aggregated over all 15s. The correct approach is: keep LSEHead, just feed longer input and split at head level.

### Knowledge Distillation & Perch KD

26. **EliKal: KLD loss distillation from Perch** — 10 epochs distillation + 12 epochs finetune achieves 0.909 with EfficientNet-B0. Multiple architectures (EffNet-B0/B3, DINOv3-small, ConvNeXt-small) all land at 0.906-0.909 with same pipeline.
27. **EliKal confirms using Perch embeddings** for all architectures' distillation.
28. **Jack (94th) confirms**: Perch KD causes backbone stability — less difference between backbones. Without KD, he sees the same swings as tennogh.
29. **tennogh (48th)**: Up to 0.02 diff between model families w/o KD. With KD: cosine similarity
    increases 0.5→0.67. "Might be worth ensembling KD and non-KD models."

### Perch Without Deep Learning (NEW 2026-04-23)

30. **Tucker Arrants: 0.925 without deep learning** — "simple clustering on the frozen perch
    embeddings and metadata engineering." The learned pool + GRU (0.91) was "total over kill."
    This is remarkable — frozen Perch + simple methods beats many trained SED models.

### hengck23's Diagnostic Framework

31. **Root causes for 0.87+ vs 0.91+**: long tail + domain shift (focal=high SNR single species, soundscape=low SNR multi-species)
32. **Key metric to monitor**: logit std as "proxy to logit distribution" — overfit models show high train logit std growing over training
33. **Robust model traits**: low logit std, train-val agreement, can train for very long epochs, flat loss valley for low sample classes, low AUC variance across folds
34. **Overfit model traits**: train-val disagreement, high growing logit std, low train BCE + high train AUC

### Salman's Head Preference

35. **Salman says "Attn head's framewise max is better than LSE head"** — we use LSEHead, he uses AttHead
36. **hengck23 suggests our 0.906 may benefit from "temporal"** — but we tried multi-context and it regressed. The gap may be elsewhere.

### External Data & 2-Stage Training

37. **Salman uses background noise from prior competitions** — nocall/BG noise from BirdCLEF 2020/2021
    + rainforest + environment sound + freefield1010/warblrb/birdvox. Says "same LB without it" but
    "helpful to generalize better." Does NOT use soundscapes as BG noise ("there are species in those").
38. **Ali Ozan: BirdCLEF 2025 pretrained checkpoint → 0.941** — effectively 2-stage training on
    2025+2026 data. The original checkpoint was pretrained on entire Xeno-Canto (filtered).
    Ali Ozan didn't use anything beyond the checkpoint. "It is pretraining, so there are no folds."
39. **Boredom confirms 0.938-0.939 w/o unlabeled, 0.950+ with** — first iteration of pseudo-labels
    on unlabeled data. "Maybe one or two more iterations are possible." Uses timm backbone + SED.
