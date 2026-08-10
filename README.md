# Prosodic and Phoneme-level Data Augmentation for Dysarthric Speech
 
This repository contains my individual contribution to a group research project on improving Automatic Speech Recognition (ASR) robustness for dysarthric speech. This repository isolates and preserves the parts of the work that I personally developed.
 
## Project Overview
 
The broader project investigated whether data augmentation can improve the robustness of HuBERT-based ASR systems for dysarthric speech recognition. Prosodic, spectral, and phoneme-level augmentation strategies were evaluated on the **UASpeech** and **TORGO** datasets, using Word Error Rate (WER) and Character Error Rate (CER) as evaluation metrics.
 
Dysarthric speech — caused by neurologically induced motor speech disorders — presents large acoustic and articulatory variability, and is typically underrepresented in standard ASR training data. Data augmentation was explored as a way to expose models to a broader range of realistic acoustic and articulatory conditions without collecting more real-world recordings.
 
## My Contributions
 
**Code:**
- **Prosodic augmentation pipeline** — waveform-level transformations simulating variation in speech rate and pitch (time stretching and pitch shifting, applied via `librosa`), designed to reflect the temporal and pitch inconsistencies observed in dysarthric speech.
- **Phoneme-level augmentation pipeline** — transcript-to-IPA conversion (via `eng_to_ipa`) followed by rule-based phoneme perturbations (substitution, deletion, prolongation) mapped back onto the waveform to simulate dysarthria-specific articulation errors such as fricative stopping, liquid simplification, and velar fronting.
**Project coordination:**
- Organized and hosted group meetings (via my personal Zoom room).
- Maintained meeting schedules and shared notes/updates for the team via WhatsApp.
## Method Summary
 
- **Prosodic augmentation**: Time stretching (p = 0.5, range [0.85, 1.15]) and pitch shifting (p = 0.5, range ±3 semitones), applied independently so each utterance could receive either, both, or neither transformation.
- **Phoneme-level augmentation**: Transcripts converted to IPA, then perturbed with phoneme substitution (dysarthria-informed mappings), phoneme deletion (p = 0.10), and phoneme prolongation (p = 0.15). These phoneme-level edits were used to drive corresponding waveform-level edits (zeroing segments for deletions, repeating segments for prolongations, adding spectral noise for substitutions).
Models were fine-tuned from HuBERT checkpoints (`facebook/hubert-large-ls960-ft` for UASpeech, `facebook/hubert-base-ls960` for TORGO) using CTC loss, and evaluated with WER and CER on held-out test sets.
 
## Results (Prosodic & Phoneme-level Augmentation)
 
| Dataset   | Method         | Test WER | Test CER |
|-----------|----------------|----------|----------|
| UASpeech  | No augmentation| 26.33%   | 6.65%    |
| UASpeech  | Prosodic       | 14.59%   | 4.14%    |
| UASpeech  | Phoneme-level  | 13.43%   | 4.17%    |
| TORGO     | No augmentation| 46.08%   | 17.22%   |
| TORGO     | Prosodic       | 28.53%   | 10.79%   |
| TORGO     | Phoneme-level  | 23.86%   | 8.88%    |
 
Both techniques produced substantial and consistent improvements over the no-augmentation baseline on both datasets. Phoneme-level augmentation achieved the best overall result on TORGO, directly targeting the articulation-related errors (substitutions, deletions, prolongations) characteristic of dysarthric speech.
 
## Datasets
 
- **UASpeech** — recordings from individuals with dysarthria [Kim et al., 2023]
- **TORGO** — recordings from individuals with neurologically induced speech disorders [Christensen et al., 2012]
These datasets are not included in this repository and must be obtained separately under their respective usage terms.
 
## Model
 
- **HuBERT** [Hsu et al., 2021] — a self-supervised speech representation model, fine-tuned with CTC loss for ASR.
## Notes
 
This repository reflects only the portions of the project that I authored. It is not the full group project and does not represent a complete, standalone pipeline (e.g., spectral augmentation and full experimental orchestration were contributed by other group members and are not included here).
 
## References
 
- Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units," IEEE/ACM TASLP, 2021.
- Kim et al., "UASpeech," 2023.
- Christensen et al., "A comparative study of adaptive, automatic recognition of disordered speech," Language Resources and Evaluation, 2012.
- Ko et al., "Audio augmentation for speech recognition," Interspeech, 2015.
- McLeod and Soderholm, "eng-to-ipa: A Python Library for Phonetic Transcription," 2020.
