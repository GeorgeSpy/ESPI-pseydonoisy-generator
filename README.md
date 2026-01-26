# ESPI-PseudoNoisy: Physics-Aware Synthetic Noise Generation

A comprehensive toolkit for generating realistic synthetic noise in ESPI (Electronic Speckle Pattern Interferometry) images for data augmentation and training purposes.

## 🎯 Overview

This repository contains physics-aware synthetic noise generators that create realistic noisy ESPI images from clean reference images. The generated pseudo-noisy data is essential for:

- **Data Augmentation:** Creating balanced training datasets (80/20 real:pseudo pairs)
- **Fine-tuning:** Training denoising models with synthetic data
- **Performance Enhancement:** Critical for achieving 96.4% accuracy in hybrid classification

## 🔬 Physics-Aware Noise Models

### Noise Cascade
1. **Multiplicative Speckle (Gamma):** Primary ESPI speckle noise
2. **Poisson Shot Noise:** Photon counting statistics
3. **Gaussian Floor:** Electronic noise and quantization

### Advanced Features
- **Frequency-aware motion blur:** Frequency and amplitude → blur length
- **Material-aware profiles:** Wood and carbon fiber specific parameters
- **Spatially-varying speckle:** ROI and distance-based variations
- **Temporal correlation:** AR(1) smoothing for sequence generation
- **Adaptive calibration:** Self-calibration from real single-shot vs averaged pairs

## 🚀 Quick Start

### Installation
```bash
pip install numpy pillow
# Optional for enhanced features:
pip install scipy scikit-image opencv-python pandas
```

### Basic Usage
```bash
# Generate pseudo-noisy images
python make_pseudo_noisy_plus.py \
    --clean-dir /path/to/clean/images \
    --out-dir /path/to/output \
    --material wood \
    --frequency 180 \
    --amplitude 0.5
```

### Advanced Usage
```bash
# Generate temporal sequences with correlation
python make_pseudo_noisy_plus.py \
    --clean-dir /path/to/clean \
    --out-dir /path/to/output \
    --material carbon_fiber \
    --frequency 335 \
    --amplitude 0.3 \
    --n-frames 10 \
    --temporal-correlation 0.85
```

## 📊 Key Scripts

### `make_pseudo_noisy_plus.py` - Main Generator
- **Features:** Full physics-aware noise cascade
- **Calibration:** Adaptive parameter estimation
- **Materials:** Wood and carbon fiber profiles
- **Output:** 8-bit PNG or 16-bit TIFF

### `make_pseudo_noisy_v3.py` - Alternative Version
- **Features:** Simplified noise model
- **Use case:** Quick generation for testing
- **Performance:** Faster processing

### `generate_pseudo_noisy.py` - Batch Generator
- **Features:** Batch processing capabilities
- **Integration:** Works with pipeline automation
- **Scaling:** Handles large datasets

### `make_pseudo_noisy_matched.py` - Matched Pairs
- **Features:** Creates matched clean/noisy pairs
- **Training:** Optimized for fine-tuning workflows
- **Quality:** Ensures consistent pairing

## 🔧 Parameters

### Material Profiles
- **Wood:** Lower frequency, higher amplitude variations
- **Carbon Fiber:** Higher frequency, lower amplitude variations

### Noise Parameters
- **Speckle (k, peak, sigma):** Gamma distribution parameters
- **Poisson (lambda):** Shot noise intensity
- **Gaussian (mu, sigma):** Electronic noise floor

### Motion Blur
- **Frequency-dependent:** Higher frequencies → more blur
- **Amplitude-dependent:** Larger amplitudes → more blur
- **Material-specific:** Different blur characteristics per material

## 📈 Performance Metrics

The generator includes comprehensive metrics:
- **PSNR:** Peak Signal-to-Noise Ratio
- **SSIM:** Structural Similarity Index
- **MPI:** Modal Preservation Index (ESPI-specific)
- **Phase Coherence:** Phase relationship preservation

## 🎓 Research Applications

### Data Augmentation
- **Balanced Training:** 80/20 real:pseudo pairs
- **Class Imbalance:** Generate samples for minority classes
- **Domain Adaptation:** Adapt to different experimental conditions

### Model Training
- **Fine-tuning:** Pre-train on synthetic data
- **Transfer Learning:** Cross-domain adaptation
- **Robustness:** Train on diverse noise conditions

### Validation
- **Ablation Studies:** Test different noise components
- **Parameter Sensitivity:** Analyze noise parameter effects
- **Generalization:** Cross-material validation

## 📁 Repository Structure

```
ESPI-PseudoNoisy/
├── README.md                          # This file
├── make_pseudo_noisy_plus.py          # Main generator
├── make_pseudo_noisy_v3.py            # Alternative version
├── generate_pseudo_noisy.py           # Batch generator
├── make_pseudo_noisy_matched.py        # Matched pairs
├── requirements.txt                    # Dependencies
├── examples/                          # Example configurations
│   ├── wood_profile.json              # Wood material profile
│   ├── carbon_profile.json            # Carbon fiber profile
│   └── sample_outputs/                # Example outputs
└── docs/                              # Documentation
    ├── noise_models.md                # Detailed noise models
    ├── calibration.md                 # Calibration procedures
    └── examples.md                    # Usage examples
```

## 🔬 Scientific Background

### ESPI Noise Characteristics
- **Speckle:** Interference pattern from coherent illumination
- **Shot Noise:** Quantum nature of light detection
- **Electronic Noise:** Amplifier and ADC noise
- **Motion Blur:** Vibration-induced temporal averaging

### Calibration Methodology
1. **Real Data Analysis:** Single-shot vs averaged pairs
2. **Parameter Estimation:** Maximum likelihood fitting
3. **Validation:** Cross-correlation with real noise
4. **Adaptation:** Material and frequency-specific tuning

## 📊 Results Integration

This toolkit is essential for the ESPI-DnCNN pipeline:
- **Training Data:** Generates synthetic pairs for fine-tuning
- **Performance:** Critical for 96.4% hybrid classification accuracy
- **Validation:** Enables robust cross-validation strategies
- **Reproducibility:** Ensures consistent synthetic data generation

## 🤝 Citation

If you use this code in your research, please cite:

```bibtex
@software{espi_pseudonoisy_2025,
  title={ESPI-PseudoNoisy: Physics-Aware Synthetic Noise Generation for ESPI Data Augmentation},
  author={Spyridakis Georgios},
  year={2025},
  url={https://github.com/[your-username]/ESPI-PseudoNoisy}
}
```

## 📄 License

This project is part of academic research. Please cite appropriately if used in your work.

---

*For detailed usage and examples, see the `docs/` directory*
