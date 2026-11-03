# Benchmark: our classifiers vs published ResNet-50 results on CUB-200-2011

We compare against **published numbers** (option 2 of section 5.11 of the brief) rather than re-running other people's code.

## Is the comparison fair?

| | Ours | Published papers below |
|---|---|---|
| Test set | Official CUB test split, 5,794 images (`Multimodal-Fatima/CUB_test`) | Same official split |
| Metric | Top-1 accuracy on the test set | Same |
| Labels used | Image + species label only | Same (no boxes or parts) |
| Backbone | ResNet-50 pretrained on ImageNet, fine-tuned | Same |
| Training images | 5,389 (600 kept for val, 5 duplicates removed) | All 5,994 |
| Input size | 224×224 | 224 or 448 (see the table) |

The test split, the metric and the backbone are the same, so the numbers can be compared. The differences are in the training recipe, and they are explained below.

## Results

| Model | Input | Epochs | Test accuracy | Source |
|---|---|---|---|---|
| Scratch CNN (ours) | 224 | 40 | 40.2% | Step 3 |
| ResNet-50 frozen (ours) | 224 | 15 | 64.3% | Step 4 |
| **ResNet-50 fine-tuned (ours, final)** | **224** | **20** | **82.9%** | Step 4 |
| ResNet-50 fine-tuned | 224 | 90 | 81.5% | Li et al., *Dynamic Computational Time for Visual Attention* (DT-RAM), 2017, Table 5 |
| ResNet-50 fine-tuned | 448 | 90 | 84.5% | Same paper, Tables 3 and 5; also used as the baseline in NTS-Net (Yang et al., ECCV 2018) |
| ResNet-50 fine-tuned | 448 | 95 | 85.3% (±0.2) | Xie et al., *Cut-Thumbnail*, ACM MM 2021, Table 6 |
| NTS-Net (ResNet-50 + learned part zooming) | 448 | – | 87.5% | Yang et al., *Learning to Navigate for Fine-grained Classification*, ECCV 2018 |

Links (the full citations are in [references.md](references.md)):
- DT-RAM: https://arxiv.org/abs/1703.10332
- NTS-Net: https://arxiv.org/abs/1809.00287 (ECCV: https://openaccess.thecvf.com/content_ECCV_2018/html/Ze_Yang_Learning_to_Navigate_ECCV_2018_paper.html)
- Cut-Thumbnail: https://arxiv.org/abs/2103.05342

## Where the differences come from

1. **Input size matters most.** At the **same 224 px size**, our 82.9% is slightly above the published 81.5%. The papers that report 84.5–85.3% use 448 px. The DT-RAM paper measured the effect directly: the same ResNet-50 goes from 81.5% at 224 to 84.5% at 448, a gain of 3 points from the image size alone. Species are often separated by small details like wing bars, eye rings and bill shape, which only cover a few pixels at 224.
2. **Better ImageNet weights.** We use torchvision's `IMAGENET1K_V2` weights (80.9% ImageNet accuracy). Papers from 2017–2018 used the original ResNet-50 weights (about 76% on ImageNet). A stronger starting point probably explains why we beat the 224 px number with 20 epochs instead of 90.
3. **Less training data.** We train on 5,389 images, because 600 are kept for validation and 5 duplicates were removed. The papers train on all 5,994 (10% more) and usually don't keep a separate validation set.
4. **Shorter schedule.** We use 20 epochs with AdamW and a cosine schedule. The papers use 90–95 epochs with SGD and step decay. Our val curve had flattened by epoch 16, so more epochs at 224 probably wouldn't add much.
5. **No fine-grained tricks.** NTS-Net adds a module that finds and zooms into informative parts, which gains another 3 points over its own ResNet-50 baseline. We only use the whole image.
6. **Noise.** With 5,794 test images, the 95% confidence interval on our accuracy is about ±1 point. So the 1.4-point gap to the 224 px paper is small and not a big win.

**In short:** at the same input size we match the published ResNet-50 result. The 2–3 point gap to the 448 px results is about what the DT-RAM paper attributes to resolution alone. Training at 448 px is the first thing we would try next, at about 4× the compute per image.
