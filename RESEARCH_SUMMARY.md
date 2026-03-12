# Research Summary and Thesis Notes

This document summarizes the internal development path of the ESPI pseudo-noisy generation pipeline. It is kept as a **development-history / thesis-notes** document, not as a stand-alone statement of the final thesis conclusion.

---

## 1. Starting point

The initial objective was to create pseudo-noisy ESPI images that could support DnCNN denoising experiments when large sets of matched real noisy/clean pairs were not yet available.

The early working scripts included:

- `make_pseudo_noisy_plus.py`
- related experimental generator variants used during calibration and ablation phases

The broader research goal was to generate synthetic supervision realistic enough to be useful for denoising development while documenting the limitations of purely synthetic training.

---

## 2. Early pseudo-noisy generation phase

An initial generation phase was performed on averaged ESPI inputs, producing pseudo-noisy samples for board-specific splits and ROI-constrained experiments.

At this stage, the work focused on:

- building usable pseudo-noisy datasets,
- handling file-format and filename mismatches,
- setting up consistent board-based splits,
- testing whether synthetic supervision could support the denoising pipeline.

---

## 3. Manifest building and LOBO-style organization

Training and validation manifests were organized with leave-one-board-out logic so that generator outputs could be integrated into denoising experiments under controlled splits.

This phase was mainly operational: matching naming conventions, handling suffix differences, and ensuring pseudo-noisy outputs could be aligned with the denoising workflow.

---

## 4. First denoising training attempts

Early denoising runs using pseudo-noisy supervision produced limited gains on real data. The main observation was not a decisive success, but rather the appearance of a **domain gap** between synthetic supervision and real ESPI measurements.

This was an important thesis result: pseudo-noisy generation was useful as an experimental tool, but realism and alignment mattered more than synthetic quantity alone.

---

## 5. Blur and hybrid-data ablations

Several ablation steps were explored, including:

- removing motion blur,
- mixing full pseudo-noisy and reduced-blur samples,
- comparing different synthetic regimes under the same validation logic.

These experiments helped clarify that the problem was not a single augmentation component in isolation. Instead, the key issue was whether the synthetic regime matched the statistics of the real acquisition process closely enough.

---

## 6. Log-domain direction

A log-domain denoising direction was also investigated because it is more natural for multiplicative speckle-like corruption. This improved the conceptual correctness of the denoising setup, but it did not remove the main bottleneck by itself: the quality of the supervision regime still dominated the outcome.

---

## 7. Domain-gap confirmation

Evaluation on real single-shot vs real averaged pairs confirmed that synthetic training could fail to transfer cleanly to the real regime. Fine-tuning on imperfect synthetic approximations did not automatically solve this problem and could even make results worse.

This stage is important in the thesis narrative because it shows that pseudo-noisy generation is helpful only when used with careful calibration and with clear awareness of its limitations.

---

## 8. Pair matching and alignment milestone

The development phase later introduced more careful matching and alignment between real single-shot inputs and their corresponding averaged references. Internal steps such as exact key matching, alignment correction, and suffix normalization were important milestones in improving evaluation reliability.

These steps should be understood as **internal development milestones**, not as the sole final thesis conclusion by themselves. Their value was in making the denoising evaluation more trustworthy and in clarifying how much of the earlier instability came from pairing and alignment issues.

---

## 9. Calibration and board-specific behavior

Subsequent calibration passes showed that the pseudo-noisy model behaved differently across boards. Some settings produced useful, stable gains on specific boards, while others exposed limitations and specimen-specific mismatch.

This board-dependent behavior is scientifically important because it shows that calibration quality and specimen realism matter directly. The repository therefore supports the thesis as a calibrated pseudo-noisy generation component, not as a universally transferable synthetic-noise engine.

---

## 10. Thesis interpretation

The final thesis interpretation is broader than any single internal milestone reported here:

- pseudo-noisy generation is valuable when real denoising supervision is scarce,
- calibration from real single-shot vs averaged references is necessary to reduce the synthetic-real gap,
- internal steps such as pair matching, alignment correction, and board-specific calibration improved the development workflow,
- but the strongest final denoising conclusions in the thesis depend on the later denoising repository and the curated downstream evaluation package, not on this repository alone.

---

## 11. Final note

This repository should therefore be read as the **public pseudo-noisy generation component** of the thesis. It documents how physically calibrated synthetic supervision was constructed and refined, while preserving the development history that motivated the later denoising conclusions.