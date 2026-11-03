# References

Papers, datasets, models and tools used in Milestone 1. This list is section 13 of the report.

## Dataset
- C. Wah, S. Branson, P. Welinder, P. Perona, S. Belongie. *The Caltech-UCSD Birds-200-2011 Dataset.* Caltech Technical Report CNS-TR-2011-001, 2011. https://www.vision.caltech.edu/datasets/cub_200_2011/
- S. Reed, Z. Akata, H. Lee, B. Schiele. *Learning Deep Representations of Fine-Grained Visual Descriptions.* CVPR 2016 (the 10 captions per image). https://arxiv.org/abs/1605.05395
- Hugging Face copies we load: `Multimodal-Fatima/CUB_train` and `Multimodal-Fatima/CUB_test`. https://huggingface.co/datasets/Multimodal-Fatima/CUB_train

## Models and methods
- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016 (ResNet-50). https://arxiv.org/abs/1512.03385
- torchvision ResNet-50 weights `IMAGENET1K_V2` (80.9% ImageNet top-1). https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
- R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017. https://arxiv.org/abs/1610.02391
- I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization* (AdamW). ICLR 2019. https://arxiv.org/abs/1711.05101
- I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine learning-rate schedule). ICLR 2017. https://arxiv.org/abs/1608.03983
- J. Bergstra, Y. Bengio. *Random Search for Hyper-Parameter Optimization.* JMLR 2012. https://www.jmlr.org/papers/v13/bergstra12a.html

## Benchmark papers (see `docs/benchmark.md`)
- Z. Li, Y. Yang, X. Liu, F. Zhou, S. Wen, W. Xu. *Dynamic Computational Time for Visual Attention* (DT-RAM). arXiv 2017. Gives ResNet-50 at 81.5% (224 px) and 84.5% (448 px), Tables 3 and 5. https://arxiv.org/abs/1703.10332 (tables readable at https://ar5iv.labs.arxiv.org/html/1703.10332)
- Z. Yang, T. Luo, D. Wang, Z. Hu, J. Gao, L. Wang. *Learning to Navigate for Fine-grained Classification* (NTS-Net). ECCV 2018. 87.5%, with a ResNet-50 baseline at 84.5%. https://arxiv.org/abs/1809.00287, https://openaccess.thecvf.com/content_ECCV_2018/html/Ze_Yang_Learning_to_Navigate_ECCV_2018_paper.html
- T. Xie, X. Cheng, X. Wang, M. Liu, J. Deng, T. Zhou, M. Liu. *Cut-Thumbnail: A Novel Data Augmentation for Convolutional Neural Network.* ACM MM 2021. ResNet-50 at 85.31 ± 0.21% (448 px), Table 6. https://arxiv.org/abs/2103.05342

## Tools
- PyTorch and torchvision: https://pytorch.org
- Hugging Face `datasets`: https://huggingface.co/docs/datasets
- MLflow (experiment tracking): https://mlflow.org
- scikit-learn (precision, recall, F1, confusion matrix): https://scikit-learn.org
- ImageHash (perceptual hashes for the duplicate check): https://github.com/JohannesBuchner/imagehash
- Matplotlib: https://matplotlib.org
