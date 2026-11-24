# Big car or small car? ResNet-18 + CLIP

https://github.com/SDAIAAcademy


Classifies a photo of a car as **big** (SUV, pickup truck, van, minivan) or **small** (sedan, hatchback, coupe, convertible, wagon). It also uses CLIP to answer **"not a car"** for photos that aren't cars. It is tested only on **car models the network never saw during training**.

Course project for *Computer Vision for Developers* (SDAIA Academy, Nov 2026). Team: **Mosaad Alrubaish** and **Majed Almutairi**

---

## 1. Use case card

| | |
|---|---|
| **User** | Operator of a multi-storey car park (mall, university, hospital) that has **compact-only levels** with narrow bays and low ceilings, and **large-bay levels** |
| **Input** | One photo of the arriving car from the entrance camera |
| **Output** | **big** → screen directs the driver to the large-bay levels · **small** → any level · **not a car** → no action |
| **Cost of a false negative** (big car called small) | **High.** An SUV or van sent to a compact level can't park, blocks the ramp, or scrapes the low ceiling and neighbouring cars. |
| **Cost of a false positive** (small car called big) | **Low.** A sedan uses a large bay, which wastes some capacity. |
| **Success metric and target** | On the sealed test set of unseen car models: **recall(big) ≥ 95 %** (at most 1 in 20 big cars missed) with **precision(big) ≥ 80 %**. Also report average precision (PR-AUC). |

Because a missed big car is the costly error, the decision threshold is set low on purpose (see §5).

## 2. Data

- **Source:** [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) (Krause et al., 2013), downloaded as parquet files from the Hugging Face mirror [`tanganke/stanford_cars`](https://huggingface.co/datasets/tanganke/stanford_cars) (~2 GB). **Licence: non-commercial research use**, which fits this course project. Details in [`data/README.md`](data/README.md).
- **Size and classes:** 16,185 web photos of 196 car models (make, model, year). Each model name contains its body type (e.g. *BMW X5 **SUV** 2007*), which we map to **big** (SUV, Cab = pickup, Van, Minivan) or **small** (Sedan, Hatchback, Coupe, Convertible, Wagon). 15 names without a body-type word were labelled by hand (`src/data.py → MANUAL_BODY`; full list in `results/model_labels.csv`). Result: **5,525 big / 10,660 small** (about 1 : 2).
- **Split method:** about 70 / 15 / 15 train / val / test, **grouped by car family** (make + model, e.g. every *Chevrolet Silverado* variant and year). A family is only ever in one split, so **every test car is a model the network never trained on**. The big/small ratio is kept similar in each split. The splits are committed in `splits/`.
- **Cleaning (notebook 01):** dropped images whose shorter side is under 64 px (thumbnails), and photos that are near-duplicates (perceptual hash distance ≤ 4) **across different car families**: the same photo filed under two models, which would leak between splits.
- **Known problems:**
  - The label comes from the **body type in the name, not measured size**. A long luxury sedan counts as *small* and a compact SUV as *big*. Wagons are labelled small.
  - **Imbalance** (1/3 big), so we use a class-weighted loss and report precision/recall for *big*.
  - **Web photos** (ads, reviews, car shows): mostly clean side or front views, not an overhead entrance camera.

## 3. Model

- **Baseline:** ImageNet ResNet-18 with a **frozen backbone**; only a new 2-class head is trained.
- **Trained model:** the same ResNet-18 with **all layers fine-tuned**. This is the *only* difference from the baseline: same data, augmentation (random crop, horizontal flip, colour jitter), class-weighted cross-entropy, AdamW with cosine schedule, 8 epochs, seed 42. We keep the epoch with the best **validation macro-F1**.
- **Key settings:** input 224×224 · batch 64 · lr 1e-3 (frozen) / 2e-4 (fine-tuned) · weight decay 1e-4
- **Curves:** `results/curves.png`

![Training curves](results/curves.png)

- **Result (best validation epoch):** baseline macro-F1 **0.880** (accuracy 0.893) → fine-tuned macro-F1 **0.955** (accuracy 0.961).
- **Reading the curves:** the frozen baseline's train and val loss stay close together, so it is limited by what the frozen ImageNet features can express (underfitting). The fine-tuned model reaches 99.5 % train accuracy while val loss levels off at about 0.14 from epoch 3, a mild sign of overfitting. Keeping the best-validation epoch protects the saved model from it.

## 4. Foundation component: CLIP (ViT-B/16)

A 2-class ResNet **must** answer big or small, even for a person, a bus or an empty bay. CLIP has learned from 400M image–text pairs. It knows the words *SUV, pickup truck, van, sedan* for car models it has seen on the web, including the ones missing from our training set. It is built into the decision rule (`src/system.py`), not run as a separate demo:

1. **Not-a-car gate:** CLIP compares the photo with car prompts and non-car prompts ("a photo of a person", "an empty parking space", "a bus", "a motorcycle", …). If p(not a car) ≥ threshold, the answer is **not a car**. The threshold is the lowest value that rejects ≤ 1 % of validation car photos.
2. **Second opinion:** `p_big = w · ResNet + (1 − w) · CLIP`, with `w` chosen on validation by log-loss.

**Results (validation, notebook 03):**
- CLIP zero-shot, with **no training at all**, reaches macro-F1 **0.941**. The fine-tuned ResNet reaches 0.955.
- Mixing them with **w = 0.8** (80 % ResNet, 20 % CLIP) gives the lowest validation log-loss (0.097, against 0.125 for ResNet alone) and macro-F1 **0.959**. So CLIP adds a small but real improvement and makes the probabilities better calibrated, which matters because we threshold them.
- Not-a-car gate: threshold **0.30** rejects **0.0 %** of validation car photos and **18 / 18** non-car photos (people, animals, text, a rocket, textures…).
- Plots: `results/mixing_weight.png`, `results/not_a_car_gate.png`

## 5. Evaluation (sealed test set: unseen car models)

Test set: **2,189 images (553 big) from 21 car families never seen in training.**

| model | accuracy @0.5 | F1(big) @0.5 | recall(big) @0.5 | avg precision (PR-AUC) |
|:--|--:|--:|--:|--:|
| CLIP zero-shot (no training) | 0.938 | 0.864 | 0.781 | 0.904 |
| ResNet-18 frozen (baseline) | 0.851 | 0.732 | 0.807 | 0.822 |
| ResNet-18 fine-tuned | 0.925 | 0.856 | 0.888 | 0.935 |
| **Fine-tuned + CLIP mix (w = 0.8)** | 0.928 | 0.861 | 0.888 | **0.948** |

Fine-tuning beats the frozen baseline on every metric (PR-AUC 0.822 → 0.935). Adding CLIP raises PR-AUC further to 0.948. CLIP alone has high accuracy but the lowest recall for big cars (0.781): it leans towards calling cars small.

<p><img src="results/pr_curve.png" width="48%"> <img src="results/confusion_matrix.png" width="40%"></p>

- **PR curve:** `results/pr_curve.png` · **Confusion matrix:** `results/confusion_matrix.png` · **Accuracy by body type:** `results/accuracy_by_body_type.csv`
- **Threshold: 0.20** instead of the default 0.5. We chose it on **validation** as the highest p(big) threshold that keeps recall(big) ≥ 95 % (validation: recall 95.2 %, precision 93.6 %). A missed big car is the expensive error, so we accept extra small cars being sent to the large-bay levels.
- **Full system on test (threshold 0.20):** recall(big) **91.9 %** · precision(big) **78.9 %** · F1(big) **0.849** · big cars missed **45** of 553 · small cars sent to big levels **136** of 1,636 · real cars called "not a car" **0.0 %**
- **Against our target** (recall ≥ 95 %, precision ≥ 80 %): we **missed both, narrowly**. The threshold met the target on validation but did not fully transfer to unseen car families. Accuracy by body type (`results/accuracy_by_body_type.csv`) shows where: vans 100 %, pickups 98.8 %, minivans 98.8 %, coupes 99.8 %, convertibles 98.0 %, but **SUVs only 83.5 % and hatchbacks only 76.7 %**. Section 6 explains why.

## 6. Failure analysis

All 12 most-confident test mistakes come from just **three car models**, all on the border between our two classes. Overview: `results/failures_overview.png`. Causes and fixes for all 12 are in `results/failures/failures.csv`.

![Failures](results/failures_overview.png)


| # | Image | Car model | Mistake (p_big) | Likely cause | Fix tried / would try |
|---|---|---|---|---|---|
| 1 | `results/failures/01_missed-big_SUV.jpg` | Hyundai Tucson SUV 2012 | big → small (0.004) | A **compact crossover**: car-height roofline and hatchback-like rear seen from behind. No Tucson was in training (split by family), and the model learned "SUV = tall and boxy". | Label "big" by real dimensions (height/length from a spec table) instead of the body-type word. Add more compact crossovers to training. |
| 2 | `results/failures/02_missed-big_SUV.jpg` | Hyundai Tucson SUV 2012 | big → small (0.008) | Rear three-quarter view in an open landscape: with nothing nearby there is **no sense of scale**, so a small SUV and a large hatchback look alike. | An entrance camera has a fixed position, so car size in pixels becomes a usable cue. Fine-tune on photos from the real camera. |
| 3 | `results/failures/03_missed-big_SUV.jpg` | Hyundai Tucson SUV 2012 | big → small (0.010) | Dealer advert with a large text banner covering part of the car: the network gets less of the body shape. | Tried: the CLIP mix (raised test PR-AUC from 0.935 to 0.948) but did not rescue the Tucson. Would try: crop to the car with a detector first. |
| 4 | `results/failures/07_false-big_Hatchback.jpg` | Nissan Juke Hatchback 2012 | small → big (0.950) | The Juke is a **tall, SUV-styled hatchback** (high bonnet, wheel arches): visually it *is* a small crossover. Arguably the label, not the model, is wrong for parking. | Add a third class ("crossover / check") or decide by measured height. Would try: hard-example mining on tall hatchbacks. |
| 5 | `results/failures/08_false-big_Hatchback.jpg` | Scion xD Hatchback 2012 | small → big (0.947) | Boxy shape and upright tailgate look like a mini SUV, and the low threshold (0.20) pushes borderline cars to "big" on purpose. | A cheap error by our use case (it only wastes a large bay). Could raise the threshold if we accept more missed SUVs. The PR curve shows the trade-off. |
| 6 | `results/failures/09_false-big_Hatchback.jpg` | Nissan Juke Hatchback 2012 | small → big (0.910) | Showroom photo, low camera angle: makes the car look taller and bigger. | Augment with camera-height changes. Evaluate on photos from the real entrance-camera height. |

**Main lesson:** the hard cases are not random. They are **crossovers**, the one car type whose "big vs small" label is genuinely ambiguous. Body-type names from 2012 don't capture it.

## 7. Limitations

- **Test recall 91.9 % is below our 95 % target.** The threshold chosen on validation did not fully transfer to unseen families with ambiguous shapes (Hyundai Tucson). A deployment would need the threshold re-checked on footage from the actual car park.
- "Big" means **body type**, not real dimensions. A real car park would care about height and length. It would need a size measurement (calibrated camera or sensors) or a per-model size lookup.
- Trained on **web photos**: clean, well-lit, mostly side and front views. An entrance camera sees cars from above, at night, in rain and partly hidden by barriers, so accuracy would drop. Production would need footage from the real camera to fine-tune and test.
- Car models from **2012 and earlier** only. Newer designs (crossovers that look between SUV and hatchback) are exactly the hard cases.
- Only **one car per photo** is assumed. Production would add a car detector first.
- The test set has a limited number of car families, so a few hard models can move the numbers noticeably.

## 8. How to run

From a fresh Google Colab (GPU runtime: *Runtime → Change runtime type → T4 GPU*):

1. Open `notebooks/01_data.ipynb` in Colab (*File → Open notebook → GitHub*, paste this repo's URL).
2. Run the first cell. It mounts Google Drive, clones the repo to `MyDrive/car-size` (shared by all four notebooks), installs the one missing library (`imagehash`; everything else comes with Colab, and notebook 02 prints the exact versions), and keeps the images on Colab's fast local disk.
3. Run the notebooks in order:
   - `01_data` downloads ~2 GB from Hugging Face (3–5 min) and saves a resized copy to Drive (`data/cars_resized.zip`), so later notebooks start in seconds.
   - Then `02_train` (~15–25 min on a T4) → `03_foundation` → `04_evaluate`.
4. Results appear in `results/`, and the models are saved in `weights/`.

Run locally or on Hugging Face Spaces: `pip install -r requirements.txt` (pinned versions), then `python app.py`

**Deploy (Hugging Face Spaces):** create a Space with the **Gradio** SDK and upload `app.py`, `requirements.txt`, `src/`, `weights/resnet18_finetuned.pt` and `results/system_config.json`. Live demo: _link_

## Repo layout

```
notebooks/   01_data · 02_train · 03_foundation · 04_evaluate
src/         data.py (download, big/small labels, dedup, split by car family)
             model.py (ResNet-18 + training loop) · clip_gate.py (CLIP zero-shot + not-a-car gate)
             system.py (full decision rule)
splits/      train/val/test CSVs (created by 01)
results/     curves, PR curve, confusion matrix, failures/
weights/     trained models (created by 02; not committed, ~45 MB each; kept in Google Drive)
app.py       Gradio demo
```

Dataset credit: J. Krause, M. Stark, J. Deng, L. Fei-Fei. *3D Object Representations for Fine-Grained Categorization*, 4th IEEE Workshop on 3D Representation and Recognition (3dRR-13), 2013.
