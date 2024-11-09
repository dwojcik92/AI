# AI: Implementing Models from Research Papers

Welcome to the **AI** repository! This repository is dedicated to implementing machine learning models from cutting-edge research papers across various domains of Artificial Intelligence. Whether you're looking to replicate the latest breakthroughs or understand how to bring theory into practice, this repo will provide clean, well-documented implementations to help you get started.

---

## Table of Contents

1. [About](#about)
2. [Implemented Papers and Models](#implemented-papers-and-models)
3. [Installation](#installation)
4. [Usage](#usage)
6. [License](#license)

---

## About

This repository features implementations of important machine learning models, accompanied by their original research papers. Each project contains:
- Model implementations with clean, well-documented code
- Links to research papers for reference
- Instructions for running, training, and testing models
- Key insights and discussions about the model architectures

The goal is to provide a comprehensive learning resource for those who want to explore the underlying techniques of modern AI research.


## Implemented Papers and Models

Here’s the ordered list of models that will be implemented, each based on influential papers. This list spans from foundational models to cutting-edge architectures.

### 1. **Perceptron & Multilayer Perceptron (MLP)**
   - **Paper**: *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain* (1958) by Frank Rosenblatt
   - **Paper**: *Learning representations by back-propagating errors* (1986) by David E. Rumelhart et al.
   - **Link**: [Paper1](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf), [Paper2](https://www.nature.com/articles/323533a0#preview)

### 2. **Recurrent Neural Network (RNN)**
   - **Paper**: Finding Structure in Time by Jeffrey L. Elman
   - **Link**: [Paper1](https://doi.org/10.1207/s15516709cog1402_1)

### 3. **LeNet (Convolutional Neural Networks)**
   - **Paper**: *Backpropagation Applied to Handwritten Zip Code Recognition* (1989) by Y. LeCun et al.
   - **Paper**: *Gradient-Based Learning Applied to Document Recognition* (1998) by Yann LeCun et al.
   - **Link**: [Paper](http://yann.lecun.com/exdb/lenet/)

### 4. **LSTM (Long Short-Term Memory)**
   - **Paper**: *Long Short-Term Memory* (1997) by Sepp Hochreiter and Jürgen Schmidhuber
   - **Link**: [Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)

### 5. **Autoencoders (AE)**
   - **Paper**: *Reducing the Dimensionality of Data with Neural Networks* (2006) by Geoffrey Hinton and Ruslan Salakhutdinov
   - **Link**: [Paper](https://www.science.org/doi/10.1126/science.1127647)

### 6. **AlexNet (Deep CNNs)**
   - **Paper**: *ImageNet Classification with Deep Convolutional Neural Networks* (2012) by Alex Krizhevsky et al.
   - **Link**: [Paper](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

### 7. **Deep Q-Network (DQN)**
   - **Paper**: *Playing Atari with Deep Reinforcement Learning* (2013) by Mnih et al.
   - **Link**: [Paper](https://arxiv.org/abs/1312.5602)

### 8. **Variational Autoencoders (VAE)**
   - **Paper**: *Auto-Encoding Variational Bayes* (2013) by Kingma and Welling
   - **Link**: [Paper](https://arxiv.org/abs/1312.6114)

### 9. **Attention Mechanism & Seq2Seq**
   - **Paper**: *Neural Machine Translation by Jointly Learning to Align and Translate* (2014) by Dzmitry Bahdanau et al.
   - **Link**: [Paper](https://arxiv.org/abs/1409.0473)

### 10. **GAN (Generative Adversarial Networks)**
   - **Paper**: *Generative Adversarial Nets* (2014) by Ian Goodfellow et al.
   - **Link**: [Paper](https://arxiv.org/abs/1406.2661)

### 11. **ResNet**
   - **Paper**: *Deep Residual Learning for Image Recognition* (2105) by Kaiming He et al.
   - **Link**: [Paper](https://arxiv.org/abs/1512.03385)

### 12. **Gated Recurrent Unit (GRU)**
   - **Paper**: *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation* (2014) by Cho et al.
   - **Link**: [Paper](https://arxiv.org/abs/1406.1078)
   - **Significance**: GRUs are a simpler and more efficient variant of LSTMs, with fewer gates. They retain the ability to model long-term dependencies but are computationally less expensive.

### 13. **YOLO (You Only Look Once)**
   - **Paper**: *You Only Look Once: Unified, Real-Time Object Detection* (2016) by Joseph Redmon et al.
   - **Link**: [Paper](https://arxiv.org/abs/1506.02640)

### 14. **Sub-Pixel Convolutional Neural Network**
   - **Paper**: *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network* (2016) by Shi et al.
   - **Link**: [Paper](https://arxiv.org/abs/1609.05158)

### 15. **Transformer (Attention Is All You Need)**
   - **Paper**: *Attention Is All You Need* (2017) by Vaswani et al.
   - **Link**: [Paper](https://arxiv.org/abs/1706.03762)

### 16. **Proximal Policy Optimization (PPO)**
   - **Paper**: *Proximal Policy Optimization Algorithms* (2017) by Schulman et al.
   - **Link**: [Paper](https://arxiv.org/abs/1707.06347)

### 17. **WideResNet**
   - **Paper**: *Wide Residual Networks* (2017) by Zagoruyko and Komodakis
   - **Link**: [Paper](https://arxiv.org/abs/1605.07146)

### 18. **MobileNet v1**
   - **Paper**: *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications* (2017) by Howard et al.
   - **Link**: [Paper](https://arxiv.org/abs/1704.04861)

### 19. **DenseNet**
   - **Paper**: *Densely Connected Convolutional Networks* (2017) by Huang et al.
   - **Link**: [Paper](https://arxiv.org/abs/1608.06993)

### 20. **BERT (Bidirectional Encoder Representations from Transformers)**
   - **Paper**: *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2019) by Devlin et al.
   - **Link**: [Paper](https://arxiv.org/abs/1810.04805)

### 21. **MobileNet v2**
   - **Paper**: *Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection, and Segmentation* (2018) by Sandler et al.
   - **Link**: [Paper](https://arxiv.org/abs/1801.04381)

### 22. **EfficientNet**
   - **Paper**: *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks* (2019) by Tan and Le
   - **Link**: [Paper](https://arxiv.org/abs/1905.11946)

### 23. **Contrastive Learning (SimCLR, MoCo)**
   - **Paper**: *A Simple Framework for Contrastive Learning of Visual Representations* (2020) by Chen et al.
   - **Link**: [Paper](https://arxiv.org/abs/2002.05709)
   - **Significance**: Key development in self-supervised learning, influencing many computer vision applications.

### 24. **MobileNeXt**
   - **Paper**: *MobileNeXt: Enhanced Inverted Residuals for Efficient Mobile Vision Applications* (2020) by Zhou et al.
   - **Link**: [Paper](https://arxiv.org/abs/2007.02269)

### 25. **Vision Transformers (ViT)**
   - **Paper**: *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2020) by Dosovitskiy et al.
   - **Link**: [Paper](https://arxiv.org/abs/2010.11929)

### 26. **Diffusion Models**
   - **Paper**: *Denoising Diffusion Probabilistic Models* (2020) by Ho et al.
   - **Link**: [Paper](https://arxiv.org/abs/2006.11239)

### 27. **CLIP (Contrastive Language–Image Pretraining)**
   - **Paper**: *Learning Transferable Visual Models From Natural Language Supervision* (2021) by Radford et al.
   - **Link**: [Paper](https://arxiv.org/abs/2103.00020)
   - **Significance**: Integrated image and text representations, showing significant progress in multimodal learning.

### 28. **Stable Diffusion and Modern Generative Models**
   - **Paper**: *High-Resolution Image Synthesis with Latent Diffusion Models* (2022) by Rombach et al.
   - **Link**: [Paper](https://arxiv.org/abs/2112.10752)
   - **Significance**: An evolution of generative models, making high-quality image generation practical.



## License

This repository is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0) - see the [LICENSE](LICENSE) file for details.