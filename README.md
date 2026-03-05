# New Sneaker Designs Generation from Existing Design Data
Implementing VAE to learn from existing popular sneaker designs for the fashion patterns, generating new novel shoe design candidates.

## Collaborators
- Ethan Wang
- Steven Si
- Runyu Yang
- Danqing Chen
- Wenbo Zhao

**Web App**: [http://dfuaix9nq1.com/](http://dfuaix9nq1.com/)

This repository presents a full notebook-driven workflow for sneaker image generation using a Beta-VAE model, from preprocessing to interactive custom design.

## Notebook Navigation

1. [01 - Data Preprocessing](#01---data-preprocessing)
2. [02 - Baseline Training (MSE + Fixed Beta)](#02---baseline-training-mse--fixed-beta)
3. [03 - Improved Training (BCE + KL Annealing)](#03---improved-training-bce--kl-annealing)
4. [04 - Latent Analysis with Rich Visualizations](#04---latent-analysis-with-rich-visualizations)
5. [05 - Interactive Custom Sneaker Design](#05---interactive-custom-sneaker-design)

---

## 01 - Data Preprocessing

**Notebook**: `notebooks/01_data_preprocessing.ipynb`

### Goal
Convert raw sneaker images into a clean model-ready dataset (`64 x 64`, RGB, white background, square format), while preserving brand/class folder structure.

### Workflow Summary
- Read original raw images recursively.
- Convert each image to RGB.
- Apply center padding to square white canvas.
- Resize to `64 x 64`.
- Save processed output under the unified data directory.

### Key Result
**[Content to be added here.]**

---

## 02 - Baseline Training (MSE + Fixed Beta)

**Notebook**: `notebooks/02_baseline_training.ipynb`

### Goal
Train the first Beta-VAE baseline using:
- MSE reconstruction loss
- Fixed beta regularization

### Training Output
![Baseline Training Curve](docs/images/02_baseline_training_output_01.png)

### Observations
**[Content to be added here.]**

---

## 03 - Improved Training (BCE + KL Annealing)

**Notebook**: `notebooks/03_improved_training_with_annealing.ipynb`

### Goal
Improve generation quality and latent behavior using:
- BCE reconstruction loss
- KL annealing schedule (dynamic beta)

### What Changed vs Baseline (Three Core Differences)

1. **Initialization from the baseline model**
- The improved model is trained by loading the baseline checkpoint first, instead of starting from random weights.
- This means Stage 2 training starts from a model that already reconstructs sneaker structure reasonably well.
- Practical effect: faster convergence and more stable improvement.

2. **KL annealing (dynamic beta in ELBO)**
- **a) ELBO and what beta means**
  - VAE objective (minimized as loss):
  - $\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot D_{KL}(q_\phi(z|x)\,\|\,p(z))$
  - Here, $\beta$ controls the strength of latent regularization:
  - Larger $\beta$ enforces a more organized latent space, but can hurt reconstruction quality.
- **b) How we implemented it**
  - Instead of using a fixed $\beta$ from epoch 1, we linearly increase $\beta$ from `0` to `2.0` during the first `10` epochs, then keep it at `2.0`.
  - This is the `KL annealing` schedule used in Notebook 03.
- **c) Why this helps**
  - Early epochs focus more on reconstruction (easier optimization).
  - Later epochs gradually enforce latent regularization.
  - Net result: better balance between visual quality and latent controllability.

3. **BCE reconstruction loss instead of MSE**
- **a) BCE vs MSE**
  - **MSE** penalizes squared pixel distance and often favors smoother outputs.
  - **BCE** treats normalized pixels as probabilities and penalizes mismatch with cross-entropy.
- **b) Why BCE can be better here**
  - For normalized shoe images in `[0, 1]` with sigmoid output, BCE typically preserves local contrast and edge sharpness better.
  - Practical effect in this project: cleaner contours and visually sharper sneaker details.

### Training Output
![Improved Training Curve](docs/images/03_01.png)

![Improved Training Curve](docs/images/03_02.png)

### Observations
- The first figure isolates **KL Loss + Beta**, making annealing behavior easy to explain.
- The second figure focuses on **Total/Reconstruction Loss + Beta**, showing optimization trend without KL scale compression.
- Together, these plots show both optimization stability and regularization progression.

---

## 04 - Latent Analysis with Rich Visualizations

**Notebook**: `notebooks/04_latent_analysis_rich_visualization.ipynb`

### Goal
Analyze what each latent dimension learns and how controllable the generator is.

### Visual Results

#### Reconstruction Quality (Input vs Reconstruction)
![Latent Analysis Output 1](docs/images/04_latent_analysis_rich_visualization_output_01.png)

#### Latent Traversal Grid
![Latent Analysis Output 2](docs/images/04_latent_analysis_rich_visualization_output_02.png)

#### Latent Mean Distribution
![Latent Analysis Output 3](docs/images/04_latent_analysis_rich_visualization_output_03.png)

#### Latent Correlation Heatmap
![Latent Analysis Output 4](docs/images/04_latent_analysis_rich_visualization_output_04.png)

#### KL Contribution per Dimension
![Latent Analysis Output 5](docs/images/04_latent_analysis_rich_visualization_output_05.png)

#### Dimension Impact Bar
![Latent Analysis Output 6](docs/images/04_latent_analysis_rich_visualization_output_06.png)

#### Random Prior Samples
![Latent Analysis Output 7](docs/images/04_latent_analysis_rich_visualization_output_07.png)

#### Latent Interpolation
![Latent Analysis Output 8](docs/images/04_latent_analysis_rich_visualization_output_08.png)

### Insights
**[Content to be added here.]**

---

## 05 - Interactive Custom Sneaker Design

**Notebook**: `notebooks/05_custom_design_interactive.ipynb`  
**Web App**: [http://dfuaix9nq1.com/](http://dfuaix9nq1.com/)

### Goal
Provide direct user control over latent dimensions (`Dim0` to `Dim15`) to generate custom sneaker designs interactively.

### Features
- Slider + numeric input for each latent dimension.
- One-click generation.
- Reset / randomize options.
- HTML deployment with backend model inference.

### Example Generated Design
![Custom Sneaker Example](artifacts/generated/custom_sneaker.png)

### Demo Notes
**[Content to be added here.]**

---

## Project Conclusion

- End-to-end pipeline is fully notebook-based.
- Model is trained, analyzed, and deployed to a live interactive website.
- The system supports both research-style latent inspection and practical controllable design generation.

### Final Reflection
**[Content to be added here.]**
