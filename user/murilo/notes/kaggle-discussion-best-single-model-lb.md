# Kaggle Discussion: What is your best single model LB score?

- **URL**: https://www.kaggle.com/competitions/birdclef-2026/discussion/683791
- **Posted**: 2026-03-21 (Antoine Masq)
- **Topic author**: Antoine Masq (33rd at 2026-05-27 capture; was 49th at 2026-05-21)
- **Comment count at capture**: 65
- **Captured**: 2026-05-27 (prior pulls 2026-04-23, 2026-04-28, 2026-05-09, 2026-05-21)
- **Companion thread**: [`kaggle-discussion-non-perch-limit.md`](kaggle-discussion-non-perch-limit.md) — "What's the limit without any Perch involved?"
- **Companion Perch-stack note**: [`2026-05-21-nina2025-eos6-protossm-stack-analysis.md`](2026-05-21-nina2025-eos6-protossm-stack-analysis.md)

Notation in the transcript below:
- `↳` indicates one level of nesting (reply); `↳↳` two levels; `↳↳↳` three levels; etc.
- ⭐ marks comments referenced by the synthesis sections at the bottom of this file.
- `"This comment has been deleted."` is preserved verbatim wherever Kaggle displays it.

---

## Original post

**Antoine Masq (49th, Topic Author) — Posted 2 months ago** ⭐

> Hi everyone,
>
> As most of us are probably still improving our baseline models, I was curious to see what your best LB score is for a single model without using the unlabeled data.
>
> So far, my best single model performance is 0.889 with an EfficientNetv2_b0 backbone

---

## Key competitor scores (at-a-glance, 2026-05-27)

Synthesis table — *see comment transcript below for source quotes*. Ranks are as of 2026-05-27.

| Person | Rank | LB | Details |
|---|---|---|---|
| **Arunodhayan** | **7th** | **0.946** | No Perch, no details (cross-ref: [`kaggle-discussion-non-perch-limit.md`](kaggle-discussion-non-perch-limit.md)) |
| **Boredom** | **2nd** (was 5th) | **0.950+** w/ unlabeled, **0.938-0.939** w/o | Non-Perch, timm+SED, ~11min inference. 1st iter on unlabeled. **No TTA**; "simple PP" worth ~+0.004 (corrected from 0.03-0.04 typo). 2026-05-27 update: confirms **"doesn't work for more than one pass"** in pseudo iteration thread. |
| **Ali Ozan Memetoglu** | **7th** (was 6th) | **0.941** single | Uses Sydorskyi 2nd-place 2025 published checkpoint directly ([dataset](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models)). 1028 attention dim. ONNX 5-fold ~30 min total. 2026-05-27 update: recommends **"trim probability"** (cell-level pseudo prob below threshold zeroed) — also says to **"increase the trim probability"** when pseudo fails. |
| **Antoine Masq** | **33rd** (was 49th, Topic Author) | **0.943** single, **0.948** ensemble | EffNetV2-B0 + XC pretrain + own-teacher PL + custom ASL + sliding-window TTA + smoothing. 10 min inference. Trajectory: 0.889 → 0.936 → 0.943 (cross-ref: [`kaggle-discussion-non-perch-limit.md`](kaggle-discussion-non-perch-limit.md)). 2026-05-27 update: warns against ensembling models trained on the **same teachers** (too correlated). |
| **Salman Ahmed** | **43rd** (was 39th) | **0.937** solo v2B0 w/o PL, **0.939** w/o PL, **0.944** w/ PL, **0.952** ensemble | Modeling breakthrough — attention investigation on hardest/easiest loss examples. B0_NS now matching v2B0. Uses PP/TTA. 20s segments, ~50-60 min inference. 2026-05-27 update: PL lift +0.005 (0.939 → 0.944); second PL iteration doesn't help; teacher is a 0.944 ensemble. |
| **Anil Ozturk** | **6th** (was 2nd) | **0.940** w/ "very weird customized pseudo" | Two trajectory points: supervised 0.931 → pseudo 0.940; supervised 0.934 → pseudo 0.939. **"Doesn't work for more than one pass."** GKF on sites OOF=0.608 (extreme CV/LB disconnect). Recommends "cell level any pseudo probability below it gets zeroed". |
| **Jack** | **17th** (was 21st) | **0.933** 5-folds no pseudo / no distill, **0.934** w/ pseudo (1 fold) | No TTA / no PP. "Pseudo not lifting solos but allowing ensembling of solos to get a much better score in same compute." Confirms Sydorskyi backbone yoink + own head with melspec match. |
| **Tucker Arrants** | **685th** (was 196th) | **0.946** best after 6 rounds of pseudo, virtually zero ensemble benefit | "6 rounds of pseudo, best model 0.946, but virtually zero ensemble benefit 😆". Earlier 0.93+ recipe (Perch-distilled B0, 5-fold + priors + smoothing) holds. Also: 0.925 w/o deep learning (frozen Perch + clustering + metadata). |
| **Zejun_** | 15th | — | "totally same, only first student is smart" — independent confirmation of pseudo-iteration plateau. |
| **Joseph** | 317th | **0.949** single, **0.947** ensemble | Solo > ensemble case — opposite of Jack's pattern. |
| **Jiacheng Ma** | **25th** | — | Tried 2nd-place 2024 pseudo recipe, didn't work down to stage 1. Asks Tucker about adding priors in logit space. Reports same pattern as Jack: solo + pseudo + ensemble not lifting much. |
| **MarkDjadchenko** | **109th** | 0.899 (clustering only) | Asks Tucker about clustering recipe. Also: "Maybe you are not training backbone during the pseudo-only distillation" — to Tucker re: 6-round plateau. |
| **To be happy** | 1149th | — | Rank-ensembling teacher boosts LB but distorts probabilities → hard to use for PL targets. |
| **BNBU Anpeng Yuan** | 918th | **0.910** 4-fold HGNetv2 no distill / no pseudo | New data point: HGNetv2 4-fold matches our 0.906 ballpark exactly without KD / PL. |
| **Starry** | 3rd | — | Asked Boredom about 0.950+ unlabeled flow |
| **Anil Ozturk** | 2nd | 0.879 | GKF on sites OOF=0.608 (extreme CV/LB disconnect) |
| **Tom** | 14th | ~similar to hengck23 | "slightly different approach which fits on 4x5sec" |
| **yukiZ** | 17th | 0.904 (single fold, exp #107) | Long table of CV/LB pairs across experiments |
| **Jack** | 21st | not disclosed in this thread; "didn't manage to see the increase" | Also yoinked Sydorskyi's backbone + own head; matched melspec; −0.012 LB swap to that melspec. Confirms Perch KD causes backbone stability |
| **tennogh** | 33rd | — | ~0.02 diff between backbones; KD reduces diversity (cos sim 0.5→0.67) |
| **Jake** | 34th | **0.915 w/o unlabeled, 0.922 w/ unlabeled** | Solo model; small ~0.007 lift from unlabeled |
| **MarkDjadchenko** | 74th | 0.899 (clustering only) | Asks Tucker about clustering recipe |
| **Kurise** | 115th | — | Discusses Sydorskyi-checkpoint reuse |
| **!!!!!** | 122nd | 0.88 | Pretrained timm; later reports the same |
| **Tucker Arrants** | **196th** | **0.93+** w/ PP (5-fold ensemble), **0.922** w/o PP | Perch-distilled B0 + 5s clips + BCE + waveform augs + site/hour priors + Gaussian smooth (5 folds). Quantified Perch KD lift at **~+0.02 LB**. Also: 0.925 w/o deep learning (frozen Perch + clustering + metadata), 0.91 w/ Perch+GRU. |
| **OpPrime** | 655th | 0.835 (fold 0) / 0.826 (5-fold) | Notebook + weights shared |
| **Aic** | 837th | — | Asks about LB stability |
| **Gaurav Rawat** | 1065th | 0.908-0.926 | GroupK on site CV; shows poor CV-LB correlation |
| **goevian** | 1075th | — | Asks about multi-context head |
| **Jiacheng Ma** | 1332nd | — | Asks Tucker how to add priors (logit space?) |
| **Amit Aharoni** | 1531st | — | Asks yukiZ about val data |
| **Seeing Times** | 1490th | — | Asks Salman about SED head |
| **XuKong Ji** | 1609th | **0.899** single fold | EffNet-B0+LSEHead+BCE, 5s. 5-fold=0.889 (worse!) |
| **Murilo Gustineli** (us) | 1715th | **0.906** (HGNet+LSEHead) | No KD, no Perch in SED |
| **hengck23** | 1813th | **0.896** CNN only | Long input wave, multi-context head, no Perch, no PP |
| **EliKal** | 2044th | **0.909** single (EffNet-B0) | Perch KD (KLD loss) + 12ep finetune, time-shift TTA. Multiple backbones cluster 0.906-0.909 |
| **chenwenqiang_001** | 2073rd | 0.899 student | Own 0.886 teacher → student +0.013 from own-teacher distillation |
| **Shiro** | 2192nd | — | Asks Salman about no-PP/TTA score |
| **vion** | 2220th | — | 0.940 ceiling math |
| **Mahog** | 2364th | — | Reports ±0.005 LB shift on seed change |
| **Leon** | 2394th | — | Working to match Salman's 0.922 single B0 |
| **D.M. (Don Mathis)** | 2573rd | 0.880 (~0.875 reproducible) | CNN + logit-LSE, BCE, 5s clips. **Pain point**: ".921 with mixup on waves + train_audio + labelled SS — I get .865 doing same." |
| **lhwcv** | 2687th | — | Discord ping to Salman |

---

## Full comment transcript (Kaggle thread display order)

### Thread 1 — Salman Ahmed's April 11 update (LB 0.932 ensemble / 0.922 solo)

**Salman Ahmed (39th) — Posted a month ago** ⭐ MAJOR (20s training, no KD, ensemble 0.932 / solo 0.922)

> As of April 11. Updated a couple of things in my train settings.
>
> Trained on 20 sec with Train Audio + Labelled Soundscapes.
>
> No KD. No Claude for now. Just wanted to solve it / write it myself after a longtime relying on claude. [Just for the fun of it].
>
> No Prior or additional features.
>
> Ensemble of 4 scores 0.932.
>
> Individual scores around 0.922.
>
> 10 seconds / 15 seconds scores almost the same for me / sometimes better [Didn't include those in my ensemble].
>
> Will play with some cool attention for next week.

**↳ Jack (21st) — Posted a month ago**

> Impressive stuff!

**↳ Antoine Masq (49th, Topic Author) — Posted a month ago**

> So this score is still without using the unlabeled soundscapes ? If so, that's super impressive!

**↳ Gaurav Rawat (1065th) — Posted a month ago**

> Awesome dunno getting single models similar but infer too slow for me :/

**↳ XuKong Ji (1609th) — Posted a month ago**

> That's impressive! If you don't mind, could you please share some details about the data processing pipeline? Specifically, do you directly convert the entire 15-second audio clip into a mel spectrogram and feed it into the model for prediction and training? Or do you split the 15-second spectrogram into three 5-second segments, convert each segment into a mel spectrogram separately, perform model inference on each one individually, and then aggregate the three results?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (20s NOT split+max)

> xD. Yes it's trained on 20 seconds audio segments. Not the split and max logits.

**↳↳↳ Mahog (2364th) — Posted a month ago**

> Does training on longer audio segments increase training time by a lot for you? 20 second segments should in theory increase training time by 4x right?

**↳ !!!!! (122nd) — Posted a month ago**

> If you don't mind me asking, what was the runtime for your submission?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (50-60 min runtime)

> Around 50 to 60 minutes I think.

**↳ Leon (2394th) — Posted a month ago**

> Awesome! Is it still the efficientnetb0 backbone?

**↳↳ Salman Ahmed (39th) — Posted a month ago**

> Yes. Solo B0 was around 0.922.

**↳↳↳ Jiacheng Ma (1332nd) — Posted a month ago**

> Great！It's still efficientnet_b0 or efficientnet_v2b0?

**↳ Starry (3rd) — Posted a month ago**

> Hi Salman~ Nice to see you in BirdClef again. I wonder do you use external data in your current pipeline?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (BG noise from past comps)

> The data I am using is used in solutions for previous competitions.
>
> I have a question for host @stefankahl ; Do we need to share that again? If yes, then where? and by what time / day?
>
> for example; no call / Background noise was used by 2nd place solution in 2023 [Background Noise(nocall in 2020, 2021 comp + rainforest + environment sound + nocall in freefield1010, warblrb, birdvox)] https://www.kaggle.com/datasets/honglihang/background-noise

**↳↳↳ Salman Ahmed (39th) — Posted a month ago**

> btw, I think I get same LB without any other data than this competition's. But I think using that previous competition's data, might be helpful to make my model generalize better.

**↳↳↳ lhwcv (2687th) — Posted a month ago**

> @salmanahmedtamu Hi Salman, are you still on Discord?

**↳↳↳↳ Salman Ahmed (39th) — Posted a month ago**

> Hi @lihaoweicvch, Great to see you in this year's competition as well. Yes, but not too active there. 😅

**↳ Ali Ozan Memetoglu (6th) — Posted a month ago**

> Thank you :)) Are you using the soundscapes too or only this?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (SS HURTS HIS LB — possible bug)

> Just Train Audios. Just started experimenting with Soundscapes, and for now, adding Soundscapes reduce my LB to around 0.85.

**↳↳↳ Salman Ahmed (39th) — Posted a month ago**

> As I was typing this, i remember something, I think there is a bug in Soundscapes training. 😂 I'll fix it soon.

**↳↳ Ali Ozan Memetoglu (6th) — Posted a month ago**

> No, I mean for the background noise I was only using soundscapes so I'm thinking of combining them with the dataset you provided

**↳↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (does NOT use SS as BG noise)

> Ah I see. No I don't use the SS as background noise, because there are species in those SS recordings.

**↳ MengYe (558th) — Posted a month ago**

> does your cv correlate with lb?

---

### Thread 2 — Salman Ahmed's 0.937 single-model breakthrough ("modeling matters")

**Salman Ahmed (39th) — Posted a month ago** ⭐⭐ (0.937 SINGLE — attention investigation)

> Finally. Single model LB 0.937 without using Unlabeled data. Recently started paying attention to the modeling part in this competition, Apparently, it was not that important in previous competitions, but in this one, for me it is one of the most important part. I have been investigating attention values on some examples with highest loss during training, and minimum loss during training, and now it's much better compared to previous models I had.

**↳ Leon (2394th) — Posted a month ago**

> Wow that is amazing! May I ask is it still efficientnet b0 ns?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (EffNetV2-B0 is the 0.937 backbone)

> xD LB 0.937 is with Efficientnet v2B0 because it's faster to train compared to others. Will try other variants (B0 NS / mixnet / nfnet etc) soon.

**↳↳↳ Seeing Times (1490th) — Posted a month ago**

> an amazing score, May I ask did you change sed head?

**↳ Leon (2394th) — Posted a month ago**

> hh sorry for keeping asking this. I am working on making my b0 single model to match your score xD. I think you mentioned before b0 MBConv doesn't learn well, and ebv2 is using Fused-MBConv, might be reason why its better (not sure). and may I also ask if your b0 model (0.922) is with any post processing (I assume only model soup)?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (no model soup; B0_NS now matches v2B0)

> xD No worries, For that previous model architecture / training strategy MBConvs were not learning, but now B0_NS is matching or performing even better than v2B0. Yeah, I am not using Model Soup this time, simple model checkpoint.

**↳ Shiro (2192nd) — Posted a month ago**

> Impressive results, with efficientnetb1 I am not able to reach that results with single models, trying different architectures. May I ask if you include postprocessing in the inference step or is it just a vanilla prediction on 5 seconds?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (relies on PP/TTA)

> Hi @ludovick Yes, I definitely rely on some post processing / TTA.

**↳ D.M. (2573rd) — Posted a month ago**

> How can cv correlate with LB? The training data are "clean" clips not embedded in soundscapes. Mixup of clean clips with unlabeled soundscapes doesn't work because there is unlabeled signal in the soundscapes. Mixup with labeled soundscapes helps got from .80 to .86 but there's not enough of them to go further. I'm stumped.

**↳ Shiro (2192nd) — Posted a month ago**

> thanks, do you know your LB on single model without postprocessing/TTA?

**↳ XuKong Ji (1609th) — Posted 22 days ago**

> Hi Salman, May I ask your single model means 5fold average or just 1fold model?

---

### Thread 3 — hengck23's diagnostic framework

**hengck23 (1813th) — Posted a month ago** ⭐⭐ (diagnostic framework — failure modes + monitoring)

> why some people get lb0.87+ and why get as high as 0.91+ for simple CNN model? (exclude effects, pseudo-labels, extreme tricks … we focus on base model performance)
>
> the reason are:
> - long tail (some class has few train samples)
> - domain shift. focal audio: one species, high snr soundscape: multiple species, low snr (background noise, bird far from mic)
>
> how to solve:
> - long tail: pretrained model, sampling, more information per sample (e.g. 10,15 sec instead of 5 sec), regularization
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
> - high auc (split into acu for normal, rare, problematic classes etc)
> - can train for every long epoch (if augmentation is good)." Good grad descent at high lr"
> - low auc variance in different folds
> - flat loss valley for low sample class
>
> how a overfit model would look like?
> - disagreement for train and validation
> - high train logit std (growing as training iteration pogress)
> - low train bce
> - kigh train auc
>
> finally, cut and paste this post into you favourite AI llm and ask for comment. If useful, add to the skills in agent.

**hengck23 (1813th) — Posted a month ago** ⭐ (methodology: "turning insights into measurable, testable pipelines")

> how to learn to be an expert in modelling. distill ChatGPT knowledge to yourself.
>
> instead of asking chaptgpt to "improve the model", …
> Pick a point from the above post, e.g. "flat loss valley for low sample class"
> ask chagpt:
> - how to measure flatness. how to prove current model methods is flat or not
> - suggest methods to improve flatness
> - implement and carry experiments
> - verify flatness is improved and lead to better train,validation and public LB.
> - if it does not, ask chatgpt why and most importantly how to prove?
>
> "turning insights into measurable, testable pipelines"

---

### Thread 4 — Tucker Arrants's Perch+GRU and Perch-only-clustering

**Tucker Arrants (196th) — Posted a month ago** ⭐ (frozen Perch + clustering only → 0.925 no DL)

> Frozen Perch embeddings + learned pool + GRU
>
> 5 folds, 3x time-shift TTA
>
> 0.91 public leaderboard
>
> Edit: I was able to score 0.925 without any deep learning, just some simple clustering on the frozen perch embeddings and metadata engineering. The learned pool + GRU was total over kill.

**↳ MarkDjadchenko (74th) — Posted 11 days ago**

> Could you tell me more about the clustering? Which clustering algorithm did you use? The best LB score I managed to achieve with clustering was 0.899.

---

### Thread 5 — hengck23's CNN-only 0.896 with multi-context head

**hengck23 (1813th) — Posted 2 months ago** ⭐⭐ (CNN-only 0.896 LB with multi-context)

> how to use CNN only (no perch2, no post processing) to get 0.896 lb
> context and augmentation is the key. here we cheat by using CNN and long input wave to capture temporal information
>
> hint: how to get even better results: multi-conext head

*(Attached image: ASCII pseudo-code for multi-context head. Reproducing from prior captures: single 15s spectrogram → HGNetV2-B0 backbone → feature map (B, C, F', T') → split T' into 3 chunks of T'/3 → LSEHead per-chunk → 3 predictions, all supervised during training with their own 5s label. LB 0.876 → 0.896 (+0.020), 4-fold, no PP. Wave mix augmentation specially designed for 3×5sec input. Training: 35k focal clips + 66 labeled soundscapes. Image vendored at `notes/figures/hengck23-cnn-birdclef.png`.)*

**↳ Tom (14th) — Posted 2 months ago**

> This is similar to my current approach, but I made a slightly different approach which fits on 4x5sec.

**↳↳ hengck23 (1813th) — Posted 2 months ago**

> I checked your Claude report. He actually knows this trick the very first day. That is impressive

**↳↳↳ This comment has been deleted.**

**↳↳↳ D.M. (2573rd) — Posted 2 months ago**

> I sort of see the overall goal, but this is too sketchy for me. I don't know how many context_nets there are or what clip length they've been trained on, nor what the '3' is in the shape of prob3. I think the bottom loop is sliding width-4 windows within a 12 timestep clip and averaging the last 2 dimensions of prob3 in each of the 12 cells. But then I still don't know what the 3 dimension is, nor what 'probs' is at the end.

**↳ goevian (1075th) — Posted a month ago**

> may i ask, what do you mean by"multi-context head"?

---

### Thread 6 — Salman Ahmed's 0.921 solo / 0.923 with priors (no Perch KD)

**Salman Ahmed (39th) — Posted 2 months ago** ⭐ (0.921 solo / 0.917 no-priors / 0.923 with priors, WITHOUT Perch KD)

> Alright, Finally have 2 SED models with ensemble of LB 0.923. and solo around 0.921 Simple training with mixup on waves. 25 epochs. CE Loss but Sigmoid at Infer Trained with train_audio and labelled soundscapes only. Without priors it achieves 0.917 and with priors -> 0.923. Nothing fancy, simple model training setup. Another thing I noticed, for melspecs, MBConvs from EfficientNets are not learning that well, compared to other architectures.
>
> WITHOUT any Perch KD

**↳ Boredom (5th) — Posted 2 months ago** ⭐ (EffNet works for Boredom at 0.925+)

> Interesting findings. In your experiments, did EfficientNet underperform compared to other architectures? Also, may I ask what you mean by "Priors" here? In my experiments, a single EfficientNet model can easily achieve an LB score above 925+.

**↳↳ Salman Ahmed (39th) — Posted 2 months ago** ⭐ (priors = species×site×hour)

> Yes, for me Efficientnet NS (with MBConv Architecture) were not converging fast and scored much lower on LB, compared to other models. Not sure why. By Priors I mean, prior table of species based on site and hours.

**↳ Murilo Gustineli (1715th) — Posted 2 months ago**

> Great score with a single EfficientNet! I'm struggling to get EfficientNet-B0 to go beyond 0.855 LB, even when trying multiple different heads and loss functions. Could you share what loss function you're using (BCE vs CE) and whether you're using an attention-based head or something simpler? Trying to understand what unlocks 925+ on EfficientNet specifically

**↳ Murilo Gustineli (1715th) — Posted 2 months ago**

> Congrats on the 0.921! My best single model so far got 0.906 LB with HGNetV2-B0 + LSEHead + BCE. I tried switching to AttHead (with tanh attention, sigmoid-space clipwise, dual 50/50 supervision) but it got worse LB results both with BCE and CE loss. Two quick questions if you don't mind: (1) what backbone did you switch to from EfficientNet? and (2) are you using 5s or longer windows for training? Thanks!

**↳↳ hengck23 (1813th) — Posted 2 months ago** ⭐ (claims 0.906 must come from distillation — WRONG re us, but interesting framing)

> i suppose your 0.906 comes from distillation. Comparing with your previous result, you would note that the improvement comes from distillation and not data.
> What to distill? logit? embedding or spatial embedding or all?
>
> if the results is from distillation, then maybe modeling is less impotant.
>
> if you look at public notebook which is just based on perch2 emebdding, they are :
> lb 0.930 = (perch2 + proxy class + temporal + PCA) + prior + state space + perclass normalisation ….
>
> you can estimate
> lb 0.92+ = (perch2 + proxy class + temporal + PCA)
>
> this means to get lb 0.92+ from timm, you need to add temporal

**↳↳↳ Murilo Gustineli (1715th) — Posted 2 months ago** ⭐ (clarifies: no KD, no Perch; multi-context regressed)

> I'm not using knowledge distillation at all actually. The 0.906 is just HGNetV2-B0 trained from ImageNet init on focal + labeled soundscapes with BCE loss. No Perch at all.
>
> Really interesting point about temporal. I've tried your 3x5s multi-context approach (backbone sees 15s, split into 3 chunks, LSEHead per-chunk) and got best-ever offline val metrics, but it regressed on LB (0.896 with 4×5s avg, vs 0.906 baseline). What exactly do you mean by "add temporal"? is it the multi-context you showed, or something else like temporal embeddings or sequence modeling?

**↳ Aic (837th) — Posted a month ago**

> May I ask, do you experience any score fluctuations in your experiments due to different numbers of epochs or changes in other hyperparameters?…it's been giving me a headache lately.

**↳↳ Mahog (2364th) — Posted a month ago** ⭐ (seed sensitivity)

> Have you managed to fix this? I'm having the same issue, even a seed change shifts LB by 0.005 :V

**↳ D.M. (2573rd) — Posted a month ago** ⭐ (CAN'T REPRODUCE SALMAN 0.921)

> 0.921 Simple training with mixup on waves. Trained with train_audio and labelled soundscapes only. No clue how you are doing that. I get .865 when I do that :/

---

### Thread 7 — Murilo Gustineli's 0.86 ablations + calls for help

**Murilo Gustineli (1715th) — Posted 2 months ago** ⭐ (our public ask)

> My best single SED model is 0.86 LB with EfficientNet-B0 - and I've been struggling to break past that despite extensive ablations.
>
> My setup: EfficientNet-B0 SED (ImageNet init), GeM + attention head, n_fft=4096, n_mels=224, AmplitudeToDB norm, soft CE dual loss (clipwise + framewise max), 10s windows, spectrogram MixUp (α=0.5), 90/10 split, 50 epochs with CosineAnnealing. Trained on focal + labeled soundscapes (234 classes).
>
> I've run 15+ ablations (BCE, ASL, LSEHead, HGNetV2-B0, waveform MixUp, weight decay, StochasticDepth) and can't break past 0.86 - offline val improvements consistently fail to translate to LB gains.
>
> I also reproduced ttahara's HGNetV2-B0 recipe (LSEHead, BCE, OneCycleLR, per-sample MixUp, 5s windows, 20 epochs) and got 0.872 val/auroc offline, but only 0.855 LB on EfficientNet with the same recipe. The val-to-LB disconnect has been consistent across all ablations - improvements in offline val rarely translate to LB gains.
>
> Curious what's driving the gap to 0.9+ for other single EfficientNet-B0 SEDs. A few things we haven't explored:
>
> - NS-JFT pretrained weights (tf_efficientnet_b0.ns_jft_in1k) vs standard ImageNet init
> - 3-channel mel input (repeat mono to match pretrained stem) vs 1-channel with weight averaging
> - Longer context windows (15-20s) for capturing insect/amphibian textures
> - Training on focal only vs focal + soundscapes - currently training on focal + soundscapes
>
> Would love to hear from folks who've broken 0.9+ @aliozanmemetoglu @hengck23 @tom99763 @cudacoding @salmanahmedtamu @hideyukizushi what was the single biggest change that got you there? Any pointers would be appreciated!

**↳ Salman Ahmed (39th) — Posted 2 months ago** ⭐ (key advice to us)

> In my experiments, more epochs make the model too confident on it's predictions and that hurt the AUC (Evaluation metric).
>
> In my experiments, SED simple Attn head's framewise max is better than lse head.
>
> I mixup waves rather than Spectrograms.
>
> For now, Efficientnet B0 didn't help.
>
> Try to listen to a audio from train soundscape and then look at the attentionhead's output. That might help.

**↳ hengck23 (1813th) — Posted 2 months ago** ⭐ (Perch-baseline framing)

> it is better to think of baseline
> 1) use perch embedding to get cv and public lb
>
> just train a linear or MLP probe on clip and evaluate on soundscape ss (in addition to clip). this give you an idea on effect of domain shift.
>
> 2) repeat (1) but using your model. train on clip only
>
> read perch paper and see how many train samples they use and compare that with kaggle data. this gives you an idea of how num and variety of data affects results.
>
> from these two, it would be enough to see if your results is reasonable or no and other kagger top results is due to tricks, post processing, better modelling, better data, etc

**↳ yukiZ (17th) — Posted 2 months ago** ⭐ (waveform augmentations: Pink Noise + Time Stretch)

> Hi! Regarding the contribution of CV/LB in my experiment, data augmentation was effective.
>
> I believe many publicly available notebooks include Mixup. On the other hand, data augmentation on raw waveforms is not seen very often, but it has been frequently used in solutions from past BirdCLEF competitions. For example, "Pink Noise" and "Time Stretch".

---

### Thread 8 — Salman Ahmed's earliest 0.908 (10s, CE, raw wave mixup)

**Salman Ahmed (39th) — Posted 2 months ago** ⭐ (original 0.908 baseline recipe)

> My best single model is SED Efficientnet B0 (Trained on 10 sec). Mixup on Raw Waves, and CrossEntropyLoss on clipwise and framewise max. Trained for 20+ epochs. This single model scores 0.908
>
> I have a ensemble of these 2 models and it scores 0.914
>
> No other augmentation. No unlabeled data.

**↳ This comment has been deleted.**

**↳ Murilo Gustineli (1715th) — Posted 2 months ago**

> Is it safe to assume that you're only training on train_audio or also using the train_soundscapes to achieve the 0.908 LB score?

**↳↳ Salman Ahmed (39th) — Posted 2 months ago**

> Yup. Train audio + label soundscapes.

**↳ Jiacheng Ma (1332nd) — Posted 2 months ago**

> What are label soundscapes and unlabel soundscapes? Thanks.

**↳↳ Salman Ahmed (39th) — Posted 2 months ago**

> train_soundscapes_labels.csv are labeled soundscapes.

**↳ This comment has been deleted.**

---

### Thread 9 — yukiZ's experiment table (CV/LB pairs)

**yukiZ (17th) — Posted 2 months ago** ⭐ (CV/LB correlation table)

> I believe our interest lies in whether there is a correlation between CV and LB, so I will share my experimental results. However, due to the highly skewed distribution of the data, the validation strategy will likely differ depending on the participant.
>
> The following experimental results show the results of training/submitting using only Fold0 with 5Fold SKF, assuming future Blending of the BB of the SED Model with B0 (approximately 5-7M parameters) or Alternatively, it uses a backbone of around 15M, similar to v2s.
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

**↳ Amit Aharoni (1531st) — Posted 2 months ago**

> Thanks for sharing Based on the AUC score, is it safe to assume that your local validation is using train_audio and not train_soundscape?

**↳↳ yukiZ (17th) — Posted 2 months ago**

> Yes, your way of thinking is correct.

---

### Thread 10 — Boredom's progression 0.922 → 0.947+ → 0.950+

**Boredom (5th) — Posted 2 months ago** ⭐ (0.922 single-fold, no unlabeled)

> Single model / Single Fold: 0.922 without using the unlabeled/extra data

**↳ vion (2220th) — Posted 2 months ago**

> Amazing score considering that 0.940 is the maximum achievable with 206 of 234 classes. Congrats!

**↳ Jiacheng Ma (1332nd) — Posted 2 months ago**

> You only use train_audio? no train_soundscape?

**↳↳ Boredom (5th) — Posted 2 months ago**

> train_audio + train_soundscape (w/ label)，w/o pseduo label

**↳ Boredom (5th) — Posted 2 months ago** ⭐ (estimates 0.94-0.945 ceiling w/o unlabeled)

> I guess without using unlabeled data, the highest LB (leaderboard score) we can achieve is around 0.94-0.945.

**↳ !!!!! (122nd) — Posted 2 months ago**

> May I ask you use Perch or just timm-based model? For me, I get low score (0.88) using pretrained timm model.

**↳ Boredom (5th) — Posted a month ago** ⭐ (0.947+ update, 13 min)

> update: Single model: 0.947+ about 13min

**↳↳ Kurise (115th) — Posted a month ago**

> Amazing results, did you used the unlabeled data?

**↳↳↳ Boredom (5th) — Posted a month ago** ⭐ (confirms unlabeled critical for >0.945)

> Yes, absolutely. I would be very surprised if someone could achieve above 0.945 with a single model without using unlabeled data😭. My best single model without unlabeled data is around 0.937-0.938.

**↳↳↳↳ Aic (837th) — Posted a month ago**

> Really impressive! Was your best single model based on Perch distillation, or was it a non-Perch model?

**↳↳↳↳↳ Boredom (5th) — Posted a month ago** ⭐ (NON-PERCH)

> non-Perch, maybe Perch can do better.

**↳↳↳↳↳ Ali Ozan Memetoglu (6th) — Posted a month ago** ⭐⭐ (0.941 SINGLE w/ 2025 PRETRAIN)

> My single best model is 0.941 but I use last year's pretrained checkpoint so I believe it is possible :D

**↳↳↳↳↳↳ Aic (837th) — Posted a month ago**

> Impressive stuff!

**↳↳↳↳↳↳ Boredom (5th) — Posted a month ago**

> 😱Did you use extra data?I've been struggling for a long time to break 0.94 without using unlabeled data..

**↳↳↳↳↳↳ Kurise (115th) — Posted a month ago**

> You did what I want to do tomorrow! May I ask how much will it improve by using last year's checkpoint?

**↳↳↳↳↳↳ Kurise (115th) — Posted a month ago**

> Not sure whether I'm right, using last year's checkpoint just like 2 stage training(pretrain on 25, retrain on 26), but there is no extra computation cost. So that we can get a model trained on both 25 and 26 data.

**↳↳↳↳↳↳↳ Ali Ozan Memetoglu (6th) — Posted a month ago** ⭐ (Sydorskyi XC pretrain)

> They used the entire Xeno-Canto dataset by filtering it and fixing a bug during pretraining, but I didn't use anything beyond the checkpoint.

**↳↳↳↳↳↳↳↳ Cody_Null (310th) — Posted a month ago**

> Im not super excited about the idea of doing all these models 5 fold on the entire XC dataset but we gotta do what we gotta do 😂

**↳↳↳↳↳↳↳↳↳ Ali Ozan Memetoglu (6th) — Posted a month ago**

> it is pretraining, so there are no folds

**↳↳↳↳↳↳ Jack (21st) — Posted 20 days ago** ⭐ (Sydorskyi backbone yoink CONFIRMED)

> You just yoinked a backbone from 2nd place and trained another head on it, yeah? They mentioned doing that with the full xc data backbone I believe

**↳↳↳↳↳↳↳ Ali Ozan Memetoglu (6th) — Posted 20 days ago** ⭐ (PUBLISHED CHECKPOINT URL)

> yes https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models

**↳↳↳↳↳↳↳↳ Jack (21st) — Posted 18 days ago**

> Did you match their melspec? My first experiment's results went well but my notebook runtime is pretty much your team's name smh

**↳↳↳↳↳↳↳↳↳ Ali Ozan Memetoglu (6th) — Posted 18 days ago** ⭐ (ATTN DIM 1028 + ONNX RUNTIME)

> Yes, 1028 is the attention dimension too. When I convert them to ONNX, all 5 folds only take half an hour

**↳↳↳↳↳↳↳↳↳↳ Jack (21st) — Posted 17 days ago** ⭐ (MELSPEC MISMATCH = −0.012 LB)

> Is your 0.941 with tta/postprocessing/priors? Swapping to match their melspec lowered my score 0.012.. the runtime is a LOT faster though hahaha

**↳ Boredom (5th) — Posted a month ago** ⭐⭐ (0.950+ SINGLE w/ 1-iter PL)

> Update: Single model LB 0.950+ (11 minutes). After reading Salman's latest post, I realized I might have been constrained by my experience from past competitions, which may have limited the optimal single-model performance of this pipeline to around 0.953–0.955. I will soon build the next pipeline from scratch for ensembling, hoping to push even higher.

**↳↳ Starry (3rd) — Posted a month ago**

> Hi Boredom, it is an amazing result, may I ask if you use unlabeled train_soundscapes in your 0.950+ model. And if so, what the score with only train_audio + labeled train_soundscapes ?

**↳↳↳ Boredom (5th) — Posted a month ago** ⭐ (1-iter PL on unlabeled; 0.938-0.939 w/o)

> Yep！this is the result of the first iteration on unlabeled data. Maybe one or two more iterations are possible. A single model using only labeled data achieves an LB of 0.938–0.939. I've never been able to break 0.94 with a single model. Compared to others with solid work, there might still be a significant gap in potential.

**↳↳ Shiro (2192nd) — Posted a month ago**

> Impressive! does it include TTA/postprocessing during inference? Is there a big gap between with/without those postprocessing in term of LB score?

**↳↳↳ Boredom (5th) — Posted a month ago** ⭐ (no TTA, simple PP)

> no tta，950+ lb include some simple pp，these can improve about 0.03-0.04

**↳↳↳↳ Kurise (115th) — Posted a month ago**

> Are you sure? I don't know how "simple" pp can enhance 0.04😨

**↳↳↳↳↳ Boredom (5th) — Posted a month ago** ⭐ (PP correction to +0.004)

> sorry，0.004。haha😂

**↳ Cody_Null (310th) — Posted 22 days ago**

> @cudacoding what sort of cv are you getting for your single model .95+ LB? If you are feeling extra generous I would also be super curious about your cv method? I was really excited about this competition because I saw they included a validation set this year, but there has proven to be a significant gap, curious if that still exists at the top end, and to what degree. I am sure the data is tough to manage but I really wish we simply had a collection of data we want to build a model for and we split files across train, public LB, and private LB so there is clear correlation for all of them but it must not be as simple as it seems.

**↳ Gaurav Rawat (1065th) — Posted a month ago**

> nice SED or vision model ?

**↳↳ Boredom (5th) — Posted a month ago** ⭐ (timm + SED)

> timm backbone+SED

---

### Thread 11 — Ali Ozan Memetoglu's initial 0.925 (no unlabeled)

**Ali Ozan Memetoglu (6th) — Posted 2 months ago** ⭐ (0.925 baseline, no unlabeled)

> Mine is at 0.925 without using the unlabeled data. I couldn't make good use of it, and it might not even be that useful in this competition. But I might be wrong.

**↳ Tom (14th) — Posted 2 months ago**

> I think the floor of top scores would be very high in the end

**↳ hengck23 (1813th) — Posted 2 months ago** ⭐ (winning solution = unlabeled)

> "nd it might not even be that useful in this competition. But I might be wrong."
>
> it should be very useful! in fact winng solution is how you make use of it

---

### Thread 12 — Jake's 0.915 / 0.922 (small unlabeled lift)

**Jake (34th) — Posted 25 days ago** ⭐ (small unlabeled lift +0.007)

> Best solo model 0.915 w/o using any of the unlabelled soundscape data. 0.922 using the unlabelled soundscape data.

**↳ BNBU Anpeng Yuan (491st) — Posted 18 days ago**

> May i ask how to choose the number of unlabelled filels?

---

### Thread 13 — Tucker Arrants's 0.93+ Perch-distilled B0 5-fold

**Tucker Arrants (196th) — Posted 25 days ago** ⭐⭐ (0.93+ recipe + 0.922 no-PP, KD lift +0.02)

> Finally broke 0.93 with a single model only using focal recordings and labeled soundscapes.
>
> Perch distilled B0 trained on 5s clips. Simple BCE loss, waveform augmentations, and moderate post processing (site/hour priors + Gaussian smooth)
>
> Without any post processing it is 0.922

**↳ Salman Ahmed (39th) — Posted 25 days ago**

> HI @tuckerarrants Did you use site/hour priors from labeled soundscapes only? Also is this a single model or 5 folds of the same model?

**↳↳ Tucker Arrants (196th) — Posted 25 days ago** ⭐ (5-fold KFold filenames + MultiLabelStrat)

> Labeled soundscapes only for the site/hour priors, for now…
>
> This is 5 folds of the same model, simple KFold on filenames and MultiLabelStrat on focal labels.

**↳↳↳ Jiacheng Ma (1332nd) — Posted 14 days ago**

> Hi, How do you add "priors" to your model? Transfer to logits space?

**↳ chenwenqiang_001 (2073rd) — Posted 22 days ago**

> what is the public score without distilling.the public score is 0.732 without using distilling on your Perch distilled B0.

**↳↳ Tucker Arrants (196th) — Posted 20 days ago** ⭐⭐ (PERCH KD LIFT QUANTIFIED AT ~+0.02)

> It is about 0.02 less than with distillation, so around 0.89, definitely much higher than 0.732. A simple linear probe on frozen perch features will get you in the mid 0.8s, so something is wrong if you are not above that.

**↳↳↳ chenwenqiang_001 (2073rd) — Posted 14 days ago** ⭐ (own-teacher KD = +0.013)

> appreciate your reply.I once used a single mode of 0.886 as the teacher and trained a student mode the same as the teacher.It increased 0.013.

---

### Thread 14 — XuKong Ji's 5-fold regression + 10s drop

**XuKong Ji (1609th) — Posted a month ago** ⭐ (5-fold WORSE than 1-fold; 10s drops 0.86)

> I have a single fold(one of 5folds) model(effecient b0+LSEHead + BCE) score 0.899 with 5s input/time smooth/time shift tta(shift 0s and 2.5s). I got something weird:
>
> - 5fold ensemble score 0.889;
> - After incresing input to 10s, score drop to 0.86(1fold)…

---

### Thread 15 — OpPrime's 0.826 + EliKal's advice

**OpPrime (655th) — Posted a month ago**

> I got 0.826 on EfficientNetB0 with 5-fold training, 0.835 on fold 0, 10 epochs per fold, feeding in 5 seconds clips only. Notebook and weights shared now.
>
> Following Salman's post below, I then got a 0.801 from SED HGNetV2 feeding it 20s of mel with a temporal attention head.
>
> I am now tracing how and where I am going wrong as my numbers don't match with others.

**↳ EliKal (2044th) — Posted a month ago** ⭐ (0.915+ ensemble of 2 from 5s models)

> I'd start by focusing on your 5s models. You can get 0.915+ with an ensemble of just two models. For 20s inputs, I'd try adding segment-wise labels from the labeled soundscapes. That said, you'll probably need to adjust the backbone or increase the resolution, since a larger context usually means a larger model. It might be worth trying 10s inputs first with the same backbone and resolution, just to see how things behave. My 10s model (EfficientNet-B0 as a starting point) isn't ready yet, I hope my intuition will work.

---

### Thread 16 — Gaurav Rawat's CV/LB data + GKF discussion

**Gaurav Rawat (1065th) — Posted a month ago** ⭐ (CV/LB data — poor monotonicity)

> - 0.82 cv lb 908
> - 0.94 cv lb 913
> - 0.97 cv lb 926
> - 0.87 cv lb 916 :/
> - 0.942 cv lb 915
> - SED 0.87 - 0.75 :(
>
> Weird but

**↳ Aic (837th) — Posted a month ago**

> May I ask about your CV strategy? Mine is quite unreliable, even though my SS_AUC is closely aligned with the LB.

**↳↳ Gaurav Rawat (1065th) — Posted a month ago**

> pretty simple groupk on site

**↳↳↳ EliKal (2044th) — Posted a month ago**

> Did you validate on xc, inat and ss data together?

**↳↳↳↳ Gaurav Rawat (1065th) — Posted a month ago**

> No, I validate strictly on Soundscapes (SS) using GroupKFold by Site as of now .

**↳↳↳↳↳ EliKal (2044th) — Posted a month ago**

> Interesting. When I tried to group the Soundscapes by site I had many missing species in the validation set. Your cv scores are OOF scores? It would make sense much more if they are

**↳↳↳↳↳ Anil Ozturk (2nd) — Posted a month ago** ⭐ (GKF site OOF=0.608 / LB=0.879 EXTREME DISCONNECT)

> Soundscape-val-only GKF on sites OOF: 0.60775 LB: 0.879
>
> lol

**↳↳↳↳↳↳ Tucker Arrants (196th) — Posted a month ago** ⭐ (DUAL-VAL PROPOSAL)

> GKF is overly harsh for this competition because the test set has site overlap with training. GKF models can't learn site-level species priors that would help at test time — they're forced to hold out entire sites during training, so they never see the "this site has these species" signal that exists in reality. But removing GKF entirely is also risky: the model might memorize site-specific artifacts that don't transfer, and you'd have no way to detect that the model relies on site memorization versus learning general features.
>
> But you can try dual validation on a single training run: train on everything, but hold out two disjoint val sets. Val-A holds out specific recordings from seen sites (tests the regime: new recordings at seen sites). Val-B holds out an entire site (tests whether the model has any ability to generalize to unseen sites as a hedge against test sets that include unseen sites). Combine them into a weighted metric for early stopping, weighted toward Val-A because it matches the test regime more directly.

---

### Thread 17 — EliKal's 4-backbone Perch-KD cluster (0.906-0.909)

**EliKal (2044th) — Posted 2 months ago** ⭐⭐ (4 backbones at 0.906-0.909 with KLD-on-Perch)

> EfficientNet-B0 – 0.909 EfficientNet-B3 – 0.906 DINOv3-small – 0.906 ConvNeXt-small – 0.907 All models were trained using the same pipeline for 12 epochs (after 10 epochs of distillation with KLD loss), with mixup applied to spectrograms. After reading the comments, I'm considering replacing this with mixup on raw waveforms. Inference was performed using only time-shift TTA.

**↳ tennogh (33rd) — Posted 2 months ago**

> Interesting that you have so little difference between backbones. With my pipeline I have up to .02 difference between model families so far.

**↳↳ Jack (21st) — Posted a month ago** ⭐ (KD causes backbone stability)

> It's the distillation (assuming he's distilling perch) causing that stability. I experience the same swings as you with different backbones

**↳↳↳ EliKal (2044th) — Posted a month ago**

> Correct. Perch embeddings for all the architectures. Did you solve this stability "issue"?

**↳↳↳↳ tennogh (33rd) — Posted a month ago** ⭐ (KD reduces diversity: cos sim 0.5→0.67)

> Not sure if this is an issue, I guess it could make the models more similar and ensembling less beneficial. I'll check when I have a stable KD pipeline.
>
> Edit: with KD and 4 different backbones, I still have a ~0.02 difference between the worst and the best (the ranking is a little bit different too). KD also reduces diversity to an extent (cosine similarity ~0.5->~0.67), to the point where is might be worth ensembling KD and non-KD models.

---

### Thread 18 — D.M. (Don Mathis)'s 0.880 + replicability pain

**D.M. (2573rd) — Posted 2 months ago** ⭐ (0.880 not replicable; mamba-from-scratch ceiling 0.772)

> Mine is only 0.880, fine-tuning an imagenet-pretrained CNN backbone that takes spectrogram input. The backbone outputs low-temporal resolution BCT (T=10, C=1024), that I then map to 234 and pool with logit-LSE to get clip logits, trained with BCE loss. The .880 isn't really replicable, more like .875 only. Trained on train_audio and the labeled soundscapes. Using 5-second clips.
>
> I tried lots of mixing schemes with the unlabeled soundscapes but that didn't transfer to the LB for me. I can't imagine how it's possible to get something like .922 using the same data I'm using.
>
> I also trained a 4-layer residual mamba model from scratch (no pretrained network at all), but that topped out at .772.

**↳ hengck23 (1813th) — Posted 2 months ago** ⭐ (long-tail framing)

> "The .880 isn't really replicable, more like .875 only." this is due to small sample size in some class. bird clef is domain and long tail problem. the issue can be solved by better validation and training methods. Hence the potential of timm and "train from pretained CNN/scratch" can be higher than you think.
>
> although "train from pretained CNN/scratch" can achieve good results, using perchv2 may be faster and easier.

---

### Thread 19 — BNBU Anpeng Yuan's 0.910 4-fold HGNetv2 (no KD, no PL)

**BNBU Anpeng Yuan (918th) — Posted a day ago** ⭐ (HGNetv2 4-fold = 0.910 baseline; matches our ballpark)

> No distill, no pseudo label. 4 Folds = 0.910 with HGNetv2

---

### Thread 20 — Jack's "pseudo doesn't help solos" cluster (2026-05-26/27, VERY RECENT)

**Jack (17th) — Posted a day ago** ⭐⭐ (0.933 5-fold no PL / 0.934 single fold w/ PL — "huge +0.001 from PP")

> I've got 0.933 with 5-folds and no pseudo or distill, 0.934 with pseudo but one fold. No TTA or postprocessing on either.
>
> What increases are people getting from postprocessing? I've tried several methods and have achieved a HUGE +0.001. Feels like a waste of time with stronger models.
>
> edit: asked about postprocessing, thread became about pseudo lol

**↳ Jiacheng Ma (25th) — Posted a day ago**

> My pseudo is not up😭.

**↳↳ Jack (17th) — Posted a day ago** ⭐ (PL enables stronger ensembling, not solo lift)

> Tbh, I haven't seen much gain either from pseudo on solos, but it's allowed me to ensemble those solos and get a much better score in the same amount of time as that 5-fold model.

**↳↳↳ Jiacheng Ma (25th) — Posted a day ago**

> My situation is just like yours.😅

**↳↳ Anil Ozturk (6th) — Posted a day ago** ⭐⭐ (CUSTOMIZED PL: 931→940, 934→939; one pass max)

> yeah kinda same
>
> - supervised 931 - a very weird customized pseudo 940
> - supervised 934 - same very weird customized pseudo 939
>
> and it doesn't work for more than one pass. 😅

**↳↳↳ Jiacheng Ma (25th) — Posted a day ago**

> Could you share your mean "very weird customized pseudo"😜haha ? I tried the 2th place in the past year, but it don't work, even down to stage1.

**↳↳↳↳ Ali Ozan Memetoglu (7th) — Posted a day ago** ⭐⭐ (TRIM PROBABILITY recommendation)

> Try to increase the trim probability

**↳↳↳↳↳ Jiacheng Ma (25th) — Posted a day ago**

> Thank you very much. Could you clarify what you mean by 'trim probability'? Are you suggesting to change the sampling ratio between pseudo-labeled and focal samples? 😀

**↳↳↳↳↳↳ Ali Ozan Memetoglu (7th) — Posted a day ago** ⭐⭐ (TRIM = CELL-LEVEL ZERO BELOW THRESHOLD)

> Cell level any pseudo probability below it gets zeroed

**↳↳↳↳↳↳↳ Jiacheng Ma (25th) — Posted a day ago** ⭐ (Jiacheng tried 0.1 threshold, didn't work for him)

> Thank you for the clarification. 😁So you mean: before training, for each pseudo-labeled soft label vector, set any probability below a threshold (e.g., 0.1) to zero.（just like the 2nd place solution） Yes — I already do this when generating the pseudo labels. However, even with this, my self-training still does not work well.😭

**↳↳↳ Jack (17th) — Posted a day ago**

> Man, if I could get 0.940/0.939 from my pseudomodels I'd be straight chillin rn. I guess I should keep working on that hahaha. You doing any postprocessing?

**↳↳↳ Boredom (2nd) — Posted a day ago** ⭐ (CONFIRMS one-pass plateau)

> it doesn't work for more than one pass. +1 🤣

**↳↳↳↳ Jack (17th) — Posted a day ago**

> Same for you?

**↳↳↳↳↳ Zejun_ (15th) — Posted 8 hours ago** ⭐ (3RD CONFIRMATION: "only first student is smart")

> totally same, only first student is smart.

**↳↳↳ Salman Ahmed (43rd) — Posted a day ago** ⭐⭐ (0.939 solo no PL → 0.944 w/ PL → 0.952 ensemble; PL = +0.005)

> My single model without pseudo is 0.939 and with pseudo its 0.944 Ensemble goes to around 0.952 Second iteration of pseudo doesn't work for me. 😶😐

**↳↳↳↳ Jack (17th) — Posted a day ago**

> 0.952 is nice. What was the teacher score for getting 0.944?

**↳↳↳↳↳ Salman Ahmed (43rd) — Posted a day ago** ⭐ (TEACHER WAS 0.944 ENSEMBLE)

> Teacher is an ensemble that scores 0.944 without pseudo.

**↳↳↳ Tucker Arrants (685th) — Posted a day ago** ⭐⭐ (6 ROUNDS PL → 0.946 BUT ZERO ENSEMBLE BENEFIT)

> I have the opposite problem - 6 rounds of pseudo, best model 0.946, but virtually zero ensemble benefit 😆

**↳↳↳↳ MarkDjadchenko (109th) — Posted 21 hours ago** ⭐ (training-backbone-during-PL hypothesis)

> Maybe you are not training backbone during the pseudo-only distillation

**↳↳↳↳ Antoine Masq (33rd, Topic Author) — Posted 12 hours ago** ⭐ (SAME-TEACHER ENSEMBLE TRAP)

> Do you ensemble models that were trained with the same teachers ? If yes, it might be because they are too correlated with each other

**↳↳↳↳ To be happy (1149th) — Posted 6 hours ago** ⭐ (RANK ENSEMBLE BREAKS PL TARGETS)

> My teacher model is an ensemble. Because the models are quite diverse, I need to use rank ensembling to boost the LB, otherwise the score drops. However, when generating pseudo-labels, I noticed that the submission after rank ensembling doesn't represent true probabilities. So it seems I can only use hard labels, and I'm not sure how I should generate soft labels instead. Has anyone encountered a similar situation? I'd love to get some advice from the top competitors here.

**↳↳↳ Joseph (317th) — Posted a day ago** ⭐ (SOLO 0.949 > ENSEMBLE 0.947 — opposite of Jack's pattern)

> wow, your single model is 940 but ensemble to top 5? my best single model for now is 949 but 947 after ensemble…

---

### Thread 21 — Nikita's PP-package disclosure + Anil's full progression (2026-05-28, VERY RECENT)

**Nikita Babych (1st) — Posted 5 hours ago** ⭐⭐⭐ (CURRENT LB LEADER: 0.935 no-PL+partial-PP / 0.955 PL+full-PP; 2-seed identical-arch ensemble for stability)

> Usually, I don't submit a single model but rather an ensemble of 2 seeds (I don't think it boosts the score much, but it makes results more stable):
>
> - Without pseudo-labels and with partial pp: ~0.935
> - With pseudo-labels and the full pp package: ~0.955

**↳ Jack (16th) — Posted 5 hours ago** ⭐ (PP-VARIANCE: "most things hurt my score")

> Full pp package? I am clearly not doing enough postprocessing with my models. Most things I've tried have only hurt my score though. Impressive stuff again man. Seeing our base scores are about the same is motivating but maaaaaaaaaaan I am hitting a wall 😆

**↳ Boredom (4th) — Posted 5 hours ago** ⭐⭐ (4-MODEL ENSEMBLE + PL + FULL PP = 0.954; matches Nikita's band)

> My 4 models with pseudo-labels and the full pp package: 0.954. By the way, does your "ensemble of 2 seeds" mean two identical models but with different seeds?

**↳↳ Nikita Babych (1st) — Posted 4 hours ago** ⭐ (2-SEED = IDENTICAL ARCH, DIFFERENT SEEDS)

> Yes, identical

**↳↳↳ Jack (16th) — Posted 44 minutes ago** ⭐ (SOUPING vs ENSEMBLING — open question)

> Do you not see improvement from souping the different seeds? Or is the ensemble just better?

**↳↳↳↳ Anil Ozturk (6th) — Posted 10 minutes ago** ⭐⭐⭐ (FIRST FULLY-DISCLOSED PROGRESSION: Supervised 0.931 → +Pseudo 0.943 → +PP 0.944; PP delta = +0.001 ONLY)

> I didn't try seed ensembling yet, still submitting soup of a single CV model set.
>
> Last status:
> - Supervised: ~0.931
> - Pseudo: ~0.943
> - Pseudo + PP: ~0.944
>
> I am running out of tricks 😅

---

## Synthesis: proven 0.93+ recipes (2026-05-21)

Single-model recipes that have publicly hit ≥0.93 LB.

### 1. Antoine Masq — 0.943 single, 0.948 ensemble (XC pretrain + own-teacher PL + custom ASL)

| Component | Value |
|---|---|
| Backbone | EfficientNetV2-B0 only |
| Pretrain | Xeno-Canto (own pretrain) |
| Fine-tune data | 2026 train_audio + labeled SS + **pseudo-labeled unlabeled SS** (own teacher) |
| Loss | Custom asymmetric loss (ASL variant; params undisclosed) |
| TTA | Sliding window + smoothing patterns |
| PL iteration | At least 2 teacher generations (0.936 → 0.943 from teacher upgrade) |
| Inference | 10 min |
| LB trajectory | 0.889 → 0.936 → 0.943 |
| Source | "non-Perch limit" thread, cross-ref [`kaggle-discussion-non-perch-limit.md`](kaggle-discussion-non-perch-limit.md) |

### 2. Arunodhayan — 0.946 single (no details)

7th place. Posted only "my single model LB is 0.946" with no further explanation. Strongest non-Perch single-model data point disclosed.

### 3. Salman Ahmed — 0.937 v2B0 SED (no KD, no unlabeled)

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

### 4. Tucker Arrants — 0.93+ Perch-distilled B0 SED (5-fold ensemble)

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
| **KD lift quantified** | **~+0.02 LB** — without distillation ≈0.89, with distillation ≈0.91 (Tucker) |
| Comparison: own-teacher KD | chenwenqiang_001: 0.886 self-teacher → student +0.013 |

### 5. Boredom — 0.938-0.939 timm+SED (no unlabeled), 0.950+ (with unlabeled, 1 iter)

| Component | Value |
|---|---|
| Backbone | timm + SED, non-Perch |
| Inference | ~11 min total; **no TTA** |
| PP | "simple PP" worth ~+0.004 (corrected from 0.03-0.04 typo) |
| Unlabeled | 1 iteration of pseudo-labels → 0.938-0.939 → 0.950+ |
| Pipeline ceiling | 0.953-0.955 estimated |

### 6. Ali Ozan Memetoglu — 0.941 (Sydorskyi 2025 published checkpoint + own head)

| Component | Value |
|---|---|
| Approach | "Yoink" Sydorskyi's published 2025 backbone (819K-recording XC pretrain + bug fix) + train new classification head on 2026 |
| Pretrain corpus | Filtered Xeno-Canto (entire 2025 checkpoint, no folds) — Sydorskyi did this; Ali Ozan didn't redo it |
| Backbone source | Public dataset `vladimirsydor/bird-clef-2025-all-pretrained-models` (EfficientNetV2-S + ECA-NFNet-L0) |
| Attention dim | **1028** (Ali Ozan: "1028 is the attention dimension too") |
| Inference | ONNX, 5 folds, ~30 min total |
| Melspec | Must match Sydorskyi's exact mel pipeline (Jack: −0.012 LB swing on mismatch) |
| 2026 training | Standard fine-tune on 2026 data with the new head |

### What 0.93+ recipes have in common

1. **At least one of**: in-domain pretrain (Antoine XC, Ali Ozan Sydorskyi-yoink), Perch KD (Tucker, EliKal cluster), modeling refinement on attention (Salman 0.922→0.937).
2. **Simple post-processing**: site/hour priors + Gaussian smoothing (Tucker, Boredom). PP alone worth +0.004 to +0.010.
3. **5-fold ensemble OR a single-model breakthrough** — Tucker uses 5-fold; Salman/Boredom/Ali Ozan/Antoine get there single-model.
4. **Either CE or BCE** (Salman: CE+sigmoid-infer; Tucker, Antoine: BCE/ASL).
5. **5s, 20s, or "doesn't matter" window** — Tucker, Boredom, EliKal use 5s; Salman uses 20s. 10s is the "death zone" (XuKong Ji, OpPrime).
6. **NOT**: special head architecture, exotic loss (vanilla ASL without tuning gets −0.03 per To be happy / Dhruv), 3-channel mel, 15s multi-context as the silver bullet.

### Where our 0.906 stalls vs 0.93+

| Lever they have / we don't | Estimated LB lift | Source |
|---|---:|---|
| In-domain pretrain (Sydorskyi backbone yoink OR own XC pretrain) | **+0.030** | Ali Ozan 0.941; Antoine 0.936 (own XC) |
| Own-teacher PL on top of stronger base | **+0.007 to +0.013** | Antoine 0.936→0.943; Boredom 0.938→0.950; chenwenqiang_001 0.886→0.899 |
| Perch knowledge distillation (KLD/MSE on Perch outputs) | **+0.020** | Tucker (with KD ≈0.91, without ≈0.89) |
| Site/hour priors at inference (from labeled SS) | +0.005-0.010 | Tucker, our pp2 |
| 4 or 5-fold ensemble (proper KFold + ensembling) | +0.005-0.010 | v5j → v5j-4fold (caveat: XuKong Ji reports 5-fold WORSE than 1-fold) |
| Attention-quality investigation (Salman) | +0.015 | Salman 0.922→0.937 |

---

## Key takeaways

### Top scores & unlabeled data (2026-05-21)

1. **Arunodhayan: 0.946 single, no Perch** — 7th place, no details disclosed.
2. **Boredom: 0.950+ single model WITH unlabeled data** — non-Perch, timm+SED, ~11min. First iteration on unlabeled. Without unlabeled: 0.938-0.939. Says pipeline ceiling may be 0.953-0.955.
3. **Antoine Masq: 0.943 single, no Perch** — EffNetV2-B0 + XC pretrain + own-teacher PL + custom ASL. 10 min inference. Cross-ref non-Perch limit thread.
4. **Ali Ozan: 0.941 single using Sydorskyi 2025 published backbone** — direct download of `vladimirsydor/bird-clef-2025-all-pretrained-models`, then own head on 2026. Attention dim 1028. ONNX 5-fold ~30 min. Jack confirms parallel attempts; melspec must match exactly (−0.012 LB cost from mismatch).
5. **Salman: 0.937 solo EfficientNetV2-B0** — jump from 0.922 to 0.937 via attention investigation on hardest/easiest loss examples. B0_NS now matching v2B0. Uses PP/TTA. Simple checkpoint (no model soup).
6. **Unlabeled data is THE differentiator for 0.94+** — Boredom: "never been able to break 0.94 with a single model" using labeled data only. 1 iteration of unlabeled → 0.950+.
7. **Tucker quantifies Perch KD at ~+0.02 LB**: with distillation ≈0.91, without ≈0.89. chenwenqiang_001: own-teacher KD (0.886 → 0.899, +0.013).

### Salman's 0.922→0.937 breakthrough

8. **"Modeling" is the key, not just data/augmentation** — Salman: "for me it is one of the most important parts."
9. **Attention investigation on loss extremes** — examined attention head outputs on hard vs easy training examples, debugged from there.
10. **EfficientNetV2-B0 (Fused-MBConv)** at 0.937. B0_NS now also matches with current training strategy.
11. **No model soup, simple checkpoint.**
12. **Post-processing + TTA both required.**
13. **Soundscape training currently hurts Salman** (0.85 with SS) — suspects a bug. Does NOT use soundscapes as background noise ("there are species in those SS recordings").

### Architecture & training

14. **20s full-audio training** (not split+max) was Salman's 0.922 approach. 10s/15s "almost the same / sometimes better."
15. **CE loss + sigmoid inference** remains Salman's loss approach.
16. **Waveform augmentations** (pink noise, time stretch) recommended by yukiZ.
17. **Offline val/auroc is unreliable**: yukiZ's data shows correlation but with significant variance.
18. **Seed sensitivity**: Mahog reports seed changes shift LB by 0.005.

### 5-fold regression

19. **XuKong Ji: 5-fold ensemble (0.889) WORSE than single fold (0.899)** — with EffNet-B0 + LSEHead + BCE. Also confirms 10s input drops to 0.86. Naive ensembling can hurt; 5s is the right window size.

### CV/LB correlation

20. **Gaurav Rawat's CV/LB data (GroupKFold on site)** — non-monotonic (e.g. 0.82 CV → 0.908 LB but 0.87 CV → 0.916 LB).
21. **Anil Ozturk: soundscape-val-only GKF on sites** — OOF 0.608, LB 0.879. Extremely poor predictive value.
22. **Tucker's dual-validation proposal**: train on everything, hold out two disjoint val sets. Val-A holds out specific recordings from seen sites. Val-B holds out an entire site. Combine into a weighted metric for early stopping.

### Multi-context & architecture

23. **hengck23: "context and augmentation is the key"** — 0.896 LB with CNN only (no Perch, no PP) using long input wave.
24. **Tom (14th) confirms similar approach** — "fits on 4x5sec."
25. **Multi-context (hengck23)** = ONE backbone pass on 15s, then SPLIT temporal features into 3 chunks, apply LSEHead per-chunk → 3 predictions. The backbone's convolutional receptive field provides cross-window context, but each prediction is still from a 5s chunk. Different from Tattaka's train_period/infer_period crop. **LB 0.876 → 0.896 (+0.020)**, 4-fold, no PP.
26. **Our v5i (10s single window) got 0.880 LB (worse than v5j's 0.906 at 5s)** — different from multi-context because v5i pools over the full 10s. hengck23's approach preserves 5s pooling resolution per chunk while adding backbone-level context.
27. **Our v7 Phase D (AttHeadMultiContext) failed badly** (−0.054) because we changed the head AND aggregated over all 15s.

### Knowledge distillation & Perch KD

28. **EliKal: KLD loss distillation from Perch** — 10 epochs distillation + 12 epochs finetune achieves 0.909 with EfficientNet-B0. Four backbones (EffNet-B0/B3, DINOv3-small, ConvNeXt-small) all land at 0.906-0.909.
29. **EliKal confirms using Perch embeddings** for all architectures.
30. **Jack: Perch KD causes backbone stability** — less difference between backbones. Without KD, same swings as tennogh.
31. **tennogh: 0.02 diff between model families w/o KD; with KD: cos similarity increases 0.5→0.67.** "Might be worth ensembling KD and non-KD models."

### Perch without deep learning

32. **Tucker Arrants: 0.925 without deep learning** — "simple clustering on the frozen perch embeddings and metadata engineering." MarkDjadchenko (74th) asks for the clustering algorithm — no public reply.

### hengck23's diagnostic framework

33. **Root causes for 0.87+ vs 0.91+**: long tail (some class has few train samples) + domain shift (focal=high SNR single species, soundscape=low SNR multi-species).
34. **Key metric to monitor**: logit std as "proxy to logit distribution" — overfit models show high train logit std growing over training.
35. **Robust model traits**: low logit std, train-val agreement, can train for very long epochs, flat loss valley for low sample classes, low AUC variance across folds.
36. **Overfit model traits**: train-val disagreement, high growing logit std, low train BCE + high train AUC.

### Salman's head preference

37. **Salman says "Attn head's framewise max is better than LSE head"** — we use LSEHead, he uses AttHead.
38. **hengck23 suggests our 0.906 may benefit from "temporal"** — but we tried multi-context and it regressed.

### External data & 2-stage training

39. **Salman uses background noise from prior competitions** — nocall/BG noise from BirdCLEF 2020/2021 + rainforest + environment sound + freefield1010/warblrb/birdvox. Says "same LB without it" but "helpful to generalize better." Does NOT use soundscapes as BG noise.
40. **Ali Ozan: BirdCLEF 2025 pretrained checkpoint → 0.941** — using Sydorskyi's *publicly downloadable* checkpoint with a new head trained on 2026. Jack confirms parallel attempts. Attention dim 1028; melspec must match exactly.
41. **Boredom: 0.938-0.939 w/o unlabeled, 0.950+ with** — first iteration of pseudo-labels on unlabeled data. Uses timm backbone + SED.
42. **Antoine Masq: 0.943 single, XC pretrain + own-teacher PL** — own XC pretrain (not Sydorskyi-yoinked). PL from own teacher, simple (no confidence threshold).

### Sydorskyi backbone reuse

43. **Ali Ozan + Jack are both reusing Sydorskyi's published 2025 pretrained backbone** without redoing the 819K-recording XC pretrain. Public dataset: `vladimirsydor/bird-clef-2025-all-pretrained-models`.
44. **Two architectures in the dataset**: EfficientNetV2-S (`tf_efficientnetv2_s_in21k`) and ECA-NFNet-L0. Both ImageNet-21k → 819K-recording in-domain pretrain.
45. **Attention dim 1028** (Sydorskyi-specific) — must be matched in any new head.
46. **Melspec must match exactly**: 32 kHz / n_mels=128 / n_fft=2048 / hop=512 / f_min=20 / top_db=80 / AmplitudeToDB → standardize → min-max. Jack: −0.012 LB swing on mismatch.
47. **ONNX inference runtime**: Ali Ozan reports ~30 min for 5 folds.

### Own-teacher pseudo-labels (NEW 2026-05-21)

48. **Own-teacher PL is structurally distinct from PL-on-unlabeled with Perch-as-teacher** (which our exp 014/015/025 rejected).
49. **Own-teacher PL lift is roughly +0.007 to +0.013**, mostly independent of teacher strength within the 0.88-0.94 band:
    - Antoine 0.936 → 0.943 (+0.007) from improved teacher
    - Boredom 0.938 → 0.950 (+0.012) from 1-iter
    - chenwenqiang_001 0.886 → 0.899 (+0.013) from own-teacher KD
50. **Antoine's PL protocol is the simplest**: pseudo-label unlabeled SS with current best model, treat as additional labeled rows, retrain. No filtering, no confidence threshold disclosed.

### Pseudo-label iteration findings (NEW 2026-05-27 — Thread 20, Jack's cluster)

This is the most important new content. Multiple top teams independently
report a consistent pattern that supersedes some of the 2026-05-21 takeaways.

51. **PL doesn't lift solo models much for several top teams.** Jack (17th):
    "I haven't seen much gain either from pseudo on solos." Anil Ozturk (6th):
    supervised 0.931 → PL 0.940 (+0.009) and 0.934 → 0.939 (+0.005) — modest,
    not headline. Salman: 0.939 → 0.944 (+0.005) — also modest. **Earlier
    Boredom/Antoine/chenwenqiang headline numbers (+0.012, +0.007, +0.013) are
    upper-band, not typical.** Median real-world solo PL lift is closer to
    +0.005.

52. **"Trim probability" is the critical detail (Ali Ozan, 7th).** Cell-level
    rule: any pseudo-probability below a threshold (e.g., 0.1) gets zeroed
    in the parquet *before training*. "Same as the 2nd place solution"
    (Jiacheng Ma). **Our exp 033 did NOT trim** — we kept soft probabilities
    everywhere ≥ 0.5 but didn't zero the noise below that. This may be one
    cause of our REJECT verdict (the protocol can amplify low-confidence
    noise on a weak base).

53. **PL doesn't iterate past round 1 — multiple independent confirmations.**
    Anil Ozturk: "doesn't work for more than one pass." Boredom: "+1." Zejun_
    (15th): "totally same, only first student is smart." Salman: "Second
    iteration doesn't work for me." **Antoine's 0.936→0.943 may be the
    exception, not the rule.** This changes the strategic value of own-teacher
    PL — it's a one-shot lever, not an iterative ladder.

54. **Where PL DOES pay off: ensembling.** Jack: "[PL] allowed me to ensemble
    those solos and get a much better score in the same amount of time as
    that 5-fold model." Salman: 0.939 solo → 0.944 w/ PL → **0.952 ensemble**
    (+0.008 from PL → +0.008 from ensemble). **PL's primary value is enabling
    diversified ensembles, not lifting solo LB.**

55. **Tucker's contrarian data point**: 6 rounds of PL → best model 0.946
    BUT virtually zero ensemble benefit. Inverts the Jack/Salman pattern.
    MarkDjadchenko's hypothesis: "Maybe you are not training backbone during
    the pseudo-only distillation" — i.e., distillation-only continuation
    may freeze the encoder, losing diversity. Antoine: "Do you ensemble
    models that were trained with the same teachers? If yes, it might be
    because they are too correlated." **Implication**: PL-iterated models
    share the teacher's signal → tight cosine similarity → poor ensemble
    decorrelation.

56. **Rank-ensemble teacher → can't generate soft PL targets.** "To be happy"
    (1149th): rank-ensembling for teacher boosts LB but distorts probabilities
    → can only use hard labels for PL. Open problem on the discussion.

57. **Joseph (317th) reports the inverse pattern**: solo 0.949 > ensemble 0.947.
    Some pipelines saturate solo before ensembling can help. Less common.

### Implications for our exp 033 REJECT verdict (NEW 2026-05-27)

58. **Exp 033 omitted "trim probability"** (Ali Ozan's critical detail). Our
    PL parquet kept all probabilities ≥ 0.5, but did NOT zero cell-level
    probabilities below ~0.1. This is the closest thing to a "missing
    protocol piece" identified in this thread.

59. **Solo PL lift is modest even on stronger bases.** Anil Ozturk 0.931→0.940
    is the strongest disclosed; Salman 0.939→0.944 is more typical. Our base
    (0.906) is lower than either; if the protocol gives +0.005 on a 0.94
    base, it may give less on a 0.906 base — or net negative if the trim
    detail isn't right.

60. **PL → ensemble is the real lever, not PL → solo.** Even if exp 033's
    individual cells stayed at 0.886-0.899, an ensemble of those 10 cells
    might lift if we trust Jack/Salman's pattern. **Untested possibility**:
    rank-ensemble of our 10 PL'd students vs the distilled-sed-kd 5-fold
    ensemble (0.915 LB) as a stretch.

61. **One-shot, not iterative.** Antoine's "0.936 → 0.943 from teacher
    iteration" is now an outlier — 4 other top teams say iteration plateaus
    at round 1. Don't plan Round 2 PL experiments.

### PP-package size + LB ceiling (NEW 2026-05-28 — Thread 21, Nikita/Boredom/Anil)

62. **Full PP package delivers +0.020 LB** (Nikita 1st): partial-PP no-PL = 0.935 → full-PP + PL = **0.955**. This is BOTH a PP upgrade (partial → full) AND a PL add (none → with) — the +0.020 is the combined delta, not pure PP. Boredom (4th) confirms the band: 4 models + PL + full PP = **0.954**. **Implication**: the realistic top-team ceiling is 0.95-0.955, not 0.96+.

63. **First fully-disclosed PL-vs-PP attribution** (Anil 6th): Supervised 0.931 → +Pseudo 0.943 (Δ **+0.012**) → +PP 0.944 (Δ **+0.001** from PP alone). **PL >> PP on Anil's stack — PP added only +0.001 LB.** This contradicts the "PP is the bigger lever" reading from Nikita's combined number; PP attribution is recipe-dependent, not universal. Jack (16th) confirms PP variance: "most things I've tried have only hurt my score."

64. **2-seed ensemble is for stability, NOT score** (Nikita explicit): "I don't think it boosts the score much, but it makes results more stable." Useful for de-noising cross-fold LB but doesn't shift leaderboard. **Implication**: in compute-constrained settings, prefer a single robust submission over 2-seed unless LB variance is causing wrong final-picks.

65. **Open question: soup-of-seeds vs ensemble-of-seeds** (Jack, no answer yet from Nikita). For our exp 034 plan, this is moot until we have multiple seeds — out of scope for the first LB submission.

66. **"Running out of tricks" cluster (Anil 6th, Jack 16th)** — even top-10 teams are saturating around 0.94-0.955. The remaining headroom is small and recipe-dependent. **Reinforces the deep research's mode prior (0.925, not 0.94) for a clean first-attempt Sydorskyi-yoink.**

### Implications for our exp 034 (UPDATED 2026-05-28 with Thread 21 data)

67. **PP layer attribution is highly recipe-dependent.** Anil's PP delta is +0.001; Nikita's combined PL+PP delta is +0.020. We should NOT plan around a specific PP-lift number for exp 034 Phase H; instead, run PP layers stepwise and attribute per layer (TopN → priors → smoothing → TTA). Drop any layer that regresses.

68. **PL gives +0.012 on a 0.931 base (Anil)**; +0.005 on a 0.939 base (Salman); +0.012 on a 0.938 base (Boredom; combined with PP). PL-on-yoinked-base would land roughly +0.005 to +0.012 over the frozen-backbone yoink baseline — small but real. **Defer to exp 034 Phase J / next experiment**; not first-pass scope.

69. **Top teams have converged on the same recipe shape**: in-domain pretrain (XC or Sydorskyi-yoink) + 1-iter PL (own teacher, soft labels with trim < 0.1) + PP package (some combo of TopN / site-hour / smoothing / TTA) + (optional) 2-seed-or-4-model ensemble for stability. **Our exp 034 plan matches this shape**; the asymmetric remaining work is the implementation correctness (Gates 1+2 mandatory).

---

## Occam's-razor priorities for us (updated 2026-05-27, post-exp-033-REJECT)

Three paths to ≥0.93+ for a non-Perch SED, ranked by EV in the remaining ~7 days (deadline 2026-06-03):

**Path A — Sydorskyi backbone yoink (Ali Ozan / Jack path).** Download
`vladimirsydor/bird-clef-2025-all-pretrained-models`. Attach own head matching
attention dim 1028 + melspec exactly (Jack: −0.012 LB on mismatch). Fine-tune
on 2026. Single-model LB target **0.92-0.94**. Cost: ~2-3 days. **Highest EV
remaining lever; Antoine's 0.943 and Ali Ozan's 0.941 both rely on this kind
of XC-pretrained base.**

**Path B — Ensemble lever (cross-decorrelated stacks).** Average exp 023
(Perch probes ensemble, 0.929 LB) with exp 032 distilled-sed-kd-5fold (0.915
LB). Pure submission-time, no retraining. Estimated landing 0.92-0.93.
Cost: ~1 hour code + 1 LB slot. **Best EV/cost ratio of any remaining lever
given exp 023's 0.929 is already validated.**

**Path C — Own-teacher PL with trim probability (re-attempt exp 033 with
the missing detail).** Re-run exp 033 a2-kd canonical cell BUT add the
cell-level trim (zero pseudo-probabilities < 0.1) before training. Ali Ozan
flagged this 2026-05-27. May fix exp 033's REJECT verdict. Cost: ~1 day
retraining + 1 LB submission. **Speculative — Jiacheng Ma already does this
trim and STILL fails, so the effect size is uncertain.**

**Path D (deferred) — Salman's attention investigation.** Compute attention
maps on hard-loss and easy-loss training examples. Look for uniform attention,
frame saturation, attention not localizing on call onsets. Iterate head/loss
to fix observed failure modes. Pairs with any base model. Cost: 1-2 days of
debugging + 1 retrain. **Defer until Path A or B lands.**

**Path E (deferred) — Salman's PP/TTA layer.** Site/hour priors at inference
+ Gaussian smoothing — Tucker quantified ~+0.005-0.010 from this. Pure
inference-time. **Defer; small lever vs. Path A.**

**Combined ceiling estimate**: 0.94-0.95 single-model, 0.95+ ensemble.
Arunodhayan 0.946 and Boredom 0.950+ are the empirical ceilings for non-Perch
single-model right now.

This supersedes the earlier exp 030 mixup-recipe and exp 031 KD-on-HGNet-LSE
work — those levers are smaller than the unexplored levers above.
**Exp 033 own-teacher PL (without XC pretrain base) is now also empirically
refuted on our codebase** — the protocol doesn't transfer at LB without the
stronger base. Path A unlocks Paths B/C/D/E by giving them a 0.93+ starting
point.

### Revised PL strategy (post-Thread 20)

If we revisit own-teacher PL after Path A, do it Salman-style:
- **One pass only** — don't plan Round 2 (4 independent confirmations of
  plateau).
- **Trim probability < 0.1 at cell level** before training (Ali Ozan).
- **Goal = enable ensembling, not lift solo** — train multiple base models
  with diverse seeds/augmentations, PL each with the same teacher, ensemble
  the PL'd students. Jack/Salman both confirm PL's primary value is
  diversified ensembling (Salman: 0.939 → 0.944 → **0.952 ensemble**).
- **Avoid same-teacher correlation trap** (Antoine warning) — ensemble PL'd
  models trained from DIFFERENT teachers if possible.

---

## Cross-references

- `user/murilo/notes/kaggle-discussion-non-perch-limit.md` — companion thread; Arunodhayan 0.946, Antoine 0.936→0.943 full trajectory, Jack's "tried every method" failure mode, MengYe ASL+HGNet recipe.
- `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md` — Tucker's notebook deep-dive.
- `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` — code audit of Tucker's notebook.
- `user/murilo/notes/2026-05-21-nina2025-eos6-protossm-stack-analysis.md` — the Perch-heavy 0.949 ensemble side; what this thread is rebelling against.
- `user/murilo/notes/yuriy-protossm-v5-annotated.md` — annotated ProtoSSM v5 reference (LB 0.925).
- Our exp 016 (ASL rejected on HGNet baseline) — worth re-running with Sydorskyi backbone + PL.
- Our exp 029 (2025-only pretrain rejected at 28K-row scale) — was the right idea, wrong scale; Sydorskyi-yoink supersedes.
- Our exp 014/015/025 (PL rejected with Perch-as-teacher) — was the right concept, wrong teacher; own-teacher PL is what Antoine/Boredom/chenwenqiang_001 use.
