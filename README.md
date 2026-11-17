# Speaker Change Under Variation

Detecting speaker changes across languages and emotions

When a bilingual speaker switches from English to Mandarin, can a speech model tell that the same person is still talking? This project studies two related tasks: **speaker verification** asks whether two clips contain the same person; **speaker-change detection** asks *when* a new person starts speaking. Four notebooks compare layers of WavLM Large with one another and, in Experiments 2–4, with ECAPA-TDNN, a model trained to distinguish speakers.

Intermediate WavLM layers detected conversational speaker changes better than the final layer. ECAPA made fewer verification errors across both languages and emotions. Locating the moment of a change was harder: using the same half-second tolerance, ECAPA found 128 of 159 changes in joined bilingual clips but only 20 of 158 in joined clips of acted speech.

[Read the illustrated research report (Word)](Speaker_Change_Research_Report.docx) · Adyant Srinivasan

## Experiments and held-out results

| Notebook | Test | What the experiment found |
| --- | --- | --- |
| [1 · Conversational speaker changes](notebooks/Experiment_1_wavlm_layerwise_scd.ipynb) | Compare a cosine-distance detector and a trained logistic probe across 24 WavLM layers on VoxConverse. | The validation-selected layer 8 probe found 97 of 137 changes within ±0.25 s (F1 **0.342**, versus **0.083** at layer 24). It also raised 333 false alarms. |
| [2 · Same speaker, different emotion](notebooks/Experiment_2_emotion_invariant_speaker_embeddings.ipynb) | Compare different emotional readings of the same sentence at normal intensity on RAVDESS. Eight actors were held out. | Across emotions, ECAPA had **3.89% EER**, versus **18.30%** for the development-selected WavLM layer 3. Layer 3 gave similar scores to the same actor across emotions, but also scored different actors highly. |
| [3 · Same speaker, different language](notebooks/Experiment_3_Language_Invariance.ipynb) | Compare English and Mandarin utterances from five held-out ASCEND speakers. | On 150 same-speaker and 150 different-speaker cross-language pairs, EER was **12.67%** for ECAPA, **29.33%** for development-selected WavLM layer 1, and **41.33%** for layer 24. |
| [4 · Locating changes in joined clips](notebooks/Experiment_4_Robust_Speaker_Change_Detection.ipynb) | Join clips at known times. Some joins change speaker; others change only language or emotion, or keep both the speaker and condition the same. | At ±0.5 s, ECAPA reached boundary F1 of **0.725** on ASCEND and **0.204** on RAVDESS, on the same 255 joined sequences per corpus for all detectors. Tightening the tolerance to ±0.25 s reduced F1 to **0.606** and **0.143**. |

**How to read the numbers:** Boundary F1 balances the share of alarms that were real changes (precision) with the share of real changes found (recall); higher is better. Equal error rate (EER) measures verification errors at the threshold where false accepts and false rejects are equal; lower is better. The two metrics answer different questions.

## Run a notebook

The notebooks run independently in Google Colab. Each starts with its own configuration and dependency setup; use a fresh runtime for each because their package versions differ. Choose a GPU runtime for the full runs. Experiment 3 explicitly requires CUDA. The notebooks need Google Drive access for data and outputs, and internet access for packages and model weights.

1. Open the notebook for the question you want to test. Before executing its data-loading cells, prepare the relevant input below. Paths are relative to `/content/drive/MyDrive/`.
2. Run the cells in order. If a setup cell requests a runtime restart, follow its instructions before loading the models. For a quick setup check, select `MODE = "smoke"` in Experiments 1 or 2, set `CFG['smoke_test']` to `True` in Experiment 3, or set `SMOKE_TEST = True` in Experiment 4.
3. To use the reported configurations, select `pilot` in Experiments 1 and 2 and disable smoke testing in Experiments 3 and 4. Smoke runs use smaller samples and are for checking the pipeline.

| Input | Used by | Expected layout with default settings |
| --- | --- | --- |
| [VoxConverse](https://github.com/joonson/voxconverse) | 1 | Place the development recordings as WAV files under `voxdata/`, or change `CFG["audio_root"]`. Audio download is off by default; the notebook handles the RTTM speaker annotations separately. |
| [RAVDESS](https://zenodo.org/records/1188976) | 2, 4 | Extract the audio-only **speech** WAVs under `RAVDESS/`. Keep their original filenames: they encode actor, emotion, sentence and repetition. For another location, use `DATASET_DIR_OVERRIDE` in notebook 2 or `CFG['ravdess']` in notebook 4. |
| [ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND) | 3, 4 | Put the dataset card at `ASCEND/README.md` and the five Parquet shards (three train, one validation, one test) under `ASCEND/main/`. To change the root, use `CFG['root']` in notebook 3 or `CFG['ascend']` in notebook 4. |

The notebooks load [WavLM Large](https://huggingface.co/microsoft/wavlm-large) and, where applicable, [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb). Their encoder weights stay fixed. Experiment 1 trains a separate logistic classifier on features from each WavLM layer.

## Find the outputs

| Experiment | Default location in Google Drive | What to look for |
| --- | --- | --- |
| 1 | `wavlm_layerwise_scd/results/<mode>/` | Recording split manifest, layer-wise detector results and figures; the results ZIP sits one directory above. |
| 2 | `RAVDESS/experiment_results/runs/<run-id>/` | Verification results, actor comparisons and figures. |
| 3 | `ASCEND/experiment3_results/` | `verification_pairs.csv`, `layer_metrics.csv` and `experiment3_report.md`. Smoke runs use a `smoke/` subdirectory. |
| 4 | `Experiment_4_Results/<full-or-smoke>/<config-hash>/` | `ground_truth_boundaries.csv`, `predictions.csv`, figures and `experiment4_report.md`. The comparison above uses joined sequences only; summaries that include uninterrupted controls use a different evaluation set. |

## Scope of the findings

VoxConverse was split by recording; speakers may recur across splits. Its boundary scores cover regions selected using the reference annotations, excluding overlap and long silences. The verification studies used small held-out speaker groups, and multiple trials reused actors or recordings. In Experiment 4, a boundary was the point where two files were joined; silence can put that timestamp before the audible change. The results do not establish performance on unconstrained conversations or on changes of accent.

The layer comparison was motivated by [Mitra et al. (2025), on representation selection for speech emotion recognition](https://arxiv.org/abs/2503.22711).
