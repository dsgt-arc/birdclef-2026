# Kaggle Discussion: What is your best single model LB score?

- **URL**: https://www.kaggle.com/competitions/birdclef-2026/discussion/683791
- **Posted**: 2026-03-21 (Antoine Masq)
- **Captured**: 2026-04-13 (updated)
- **46+ comments**

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

| Person | Rank | LB | Details |
|--------|------|-----|---------|
| **Boredom** | **1st** | **0.947+** w/ unlabeled, **0.937-0.938** w/o | Non-Perch, non-KD. ~13min inference. |
| **Ali Ozan Memetoglu** | 3rd | **0.941** single model | Uses BirdCLEF 2025 pretrained checkpoint (2-stage) |
| **Kurise** | 4th | — | Notes 2025 checkpoint = 2-stage training on 25+26 data |
| **Tom** | 7th | ~similar to hengck23 | "slightly different approach which fits on 4x5sec" |
| **Salman Ahmed** | 17th | **0.922** solo B0, **0.932** ens of 4 | 20sec training, no KD, no priors. Uses BG noise external data |
| **tennogh** | 31st | — | ~0.02 diff between backbones; KD reduces diversity (cos sim 0.5→0.67) |
| **Jack** | 50th | — | Confirms Perch KD causes backbone stability |
| **yukiZ** | 57th | **0.904** (fold 0 only, 5-fold SKF) | Raw waveform augmentations |
| **Murilo Gustineli** | 141st | **0.906** (HGNet+LSEHead) | No KD, no Perch in SED |
| **Tucker Arrants** | 311th | **0.91** | Frozen Perch + learned pool + GRU, 5-fold, 3x TTA |
| **hengck23** | 364th | **0.896** CNN only | Long input wave, multi-context head, no Perch, no PP |
| **Gaurav Rawat** | 514th | **0.926** LB | GroupK on site CV, shows poor CV-LB correlation |
| **EliKal** | 682nd | **0.909** single (EffNet-B0) | Perch KD (KLD loss) + 12ep finetune, time-shift TTA |
| **XuKong Ji** | 1057th | **0.899** single fold | EffNet-B0+LSEHead+BCE, 5s. 5-fold=0.889 (worse!) |
| Antoine Masq | 501st | 0.889 | EfficientNetV2-B0 |
| Don Mathis | 1237th | 0.880 | CNN + logit-LSE, BCE, 5s clips |

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

### tennogh (31st) — Posted 5 days ago (UPDATED)

> Edit: with KD and 4 different backbones, I still have a ~0.02 difference
> between the worst and the best (the ranking is a little bit different too).
> KD also reduces diversity to an extent (cosine similarity ~0.5->~0.67),
> to the point where is might be worth ensembling KD and non-KD models.

---

## Key Takeaways

### Top Scores & Unlabeled Data (Updated 2026-04-13)

1. **Boredom: 0.947+ single model WITH unlabeled data** — non-Perch, ~13min inference.
   Without unlabeled: 0.937-0.938. Confirms unlabeled data adds ~0.010.
   Previous ceiling estimate (0.94-0.945 w/o unlabeled) was accurate.
2. **Ali Ozan: 0.941 single model using BirdCLEF 2025 pretrained checkpoint** — 2-stage
   training (pretrain on 2025, retrain on 2026). No extra compute, just leverages prior year data.
3. **Salman: 0.922 solo EfficientNet-B0 on 20sec** — ensemble of 4 = 0.932. Dropped max-trick
   in favor of full 20sec training. Still no KD, no priors. Uses background noise from prior
   competition solutions as external data (but says same LB without it).
4. **Unlabeled data is THE differentiator for 0.94+** — Boredom struggles to break 0.94 w/o it.
   Ali Ozan uses 2025 data as proxy. Our Phase C waveform mixing is the right direction.

### Architecture & Training

5. **Salman still on EfficientNet-B0** — clarified "Solo B0 was around 0.922". Previously
   said MBConvs "not learning well" but apparently still using B0 with updated settings.
6. **20sec full-audio training** (not split+max) is Salman's current approach. 10s/15s "almost
   the same / sometimes better" — he only uses 20sec in his ensemble.
7. **CE loss + sigmoid inference** remains Salman's approach with AttHead
8. **Waveform augmentations** (pink noise, time stretch) recommended by yukiZ
9. **Offline val/auroc is unreliable**: yukiZ's data shows correlation but with significant variance

### 5-Fold Regression (NEW — validates our observation)

10. **XuKong Ji: 5-fold ensemble (0.889) WORSE than single fold (0.899)** — with EffNet-B0 +
    LSEHead + BCE. Also confirms 10s input drops to 0.86 (matches our v5i experience).
    This validates that naive ensembling can hurt, and 5s is the right window size.

### CV/LB Correlation

11. **Gaurav Rawat's CV/LB data (GroupKFold on site)** shows poor correlation:
   - 0.82 CV → 0.908 LB
   - 0.94 CV → 0.913 LB
   - 0.97 CV → 0.926 LB
   - 0.87 CV → 0.916 LB (worse CV, better LB than 0.82→0.908)
   - 0.942 CV → 0.915 LB
   The CV values are all over the place relative to LB ranking. Not monotonic.

12. **Anil Ozturk: soundscape-val-only GKF on sites** — OOF 0.608, LB 0.879. The soundscape-only
    CV score is extremely low and not predictive of LB at all.

13. **Tucker Arrants explains why**: GKF is "overly harsh" because **train and test sites overlap**.
    GKF forces the model to generalize to *unseen* sites, but the test set includes *seen* sites too.
    Proposes **joint validation**: one metric for within-site generalization + one for site-level
    generalization, with combined checkpoint selection.

### Multi-Context & Architecture

14. **hengck23: "context and augmentation is the key"** — 0.896 LB with CNN only (no Perch, no PP) using **long input wave** to capture temporal info. Hints at **multi-context head** for even better results.
15. **Tom (7th) confirms similar approach** — "fits on 4x5sec" (20s total context via multiple 5s windows)
16. **Multi-context (hengck23)** = ONE backbone pass on 15s, then SPLIT temporal features into 3 chunks, apply LSEHead per-chunk → 3 predictions. The backbone's convolutional receptive field provides cross-window context, but each prediction is still from a 5s chunk. NOT the same as Tattaka's train_period/infer_period crop (which aggregates attention over all 15s). hengck23 uses HGNetV2-B0 + LSE head + 4-fold → **+0.020 gain** (0.876→0.896).
17. **Our v5i (10s single window) got 0.880 LB (worse than v5j's 0.906 at 5s)** — different from multi-context because v5i pools over the full 10s. hengck23's approach preserves 5s pooling resolution per chunk while adding backbone-level context.
18. **Our v7 Phase D (AttHeadMultiContext) failed badly** (−0.054) because we changed the head AND aggregated over all 15s. The correct approach is: keep LSEHead, just feed longer input and split at head level.

### Knowledge Distillation & Perch KD

19. **EliKal: KLD loss distillation from Perch** — 10 epochs distillation + 12 epochs finetune achieves 0.909 with EfficientNet-B0. Multiple architectures (EffNet-B0/B3, DINOv3-small, ConvNeXt-small) all land at 0.906-0.909 with same pipeline.
20. **EliKal confirms using Perch embeddings** for all architectures' distillation.
21. **Jack (50th) confirms**: Perch KD causes backbone stability — less difference between backbones. Without KD, he sees the same swings as tennogh.
22. **tennogh (31st)**: Up to 0.02 diff between model families w/o KD. With KD: cosine similarity
    increases 0.5→0.67. "Might be worth ensembling KD and non-KD models."

### hengck23's Diagnostic Framework (NEW)

23. **Root causes for 0.87+ vs 0.91+**: long tail + domain shift (focal=high SNR single species, soundscape=low SNR multi-species)
24. **Key metric to monitor**: logit std as "proxy to logit distribution" — overfit models show high train logit std growing over training
25. **Robust model traits**: low logit std, train-val agreement, can train for very long epochs, flat loss valley for low sample classes, low AUC variance across folds
26. **Overfit model traits**: train-val disagreement, high growing logit std, low train BCE + high train AUC

### Salman's Head Preference

27. **Salman says "Attn head's framewise max is better than LSE head"** — we use LSEHead, he uses AttHead
28. **hengck23 suggests our 0.906 may benefit from "temporal"** — but we tried multi-context and it regressed. The gap may be elsewhere.

### External Data & 2-Stage Training (NEW 2026-04-13)

29. **Salman uses background noise from prior competitions** — nocall/BG noise from BirdCLEF 2020/2021
    + rainforest + environment sound + freefield1010/warblrb/birdvox. Says "same LB without it" but
    "helpful to generalize better."
30. **Ali Ozan: BirdCLEF 2025 pretrained checkpoint → 0.941** — effectively 2-stage training on
    2025+2026 data with zero extra compute. Kurise (4th) confirms this interpretation.
31. **Boredom confirms 0.937-0.938 w/o unlabeled, 0.947+ with** — the ~0.010 gap from unlabeled
    data is now confirmed by the #1 competitor. Our Phase C waveform mixing targets exactly this gap.
