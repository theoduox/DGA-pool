# DGA-Pool usage

[Home](../README.md) | [中文](使用说明.md)

## Where to start

To process images or video, use the enhancement entry points in scripts/ with the supplied checkpoint. For retraining, see the training entry points and configs/. Read src/ for the algorithm implementation, and run tests/ when you want to check installation and entry-point behavior.

| Directory | Contents | How to use it |
|---|---|---|
| scripts/ | Runnable command-line entry points | Select a task and run python -m scripts.module_name |
| src/ | Networks, losses, Gamma search, ROI and data-loading implementations | Imported by scripts/; usually not run directly |
| tests/ | Focused functional checks | Run when needed after installation; not required before ordinary image processing |
| configs/ | JSON parameter settings | Select with --config; no training data are stored here |
| data/ | Examples and dataset extraction locations | Place full data in dataset-images/ and dataset-videos/ separately |
| weights/ | Supplied enhancement checkpoint enhancement_realtime.pth | Select with --weights; no classifier checkpoints are included |

All paths below are relative to the project root, where README.md, scripts/ and src/ are located. Run commands from that directory rather than from inside scripts/ or src/.

## scripts: choose an entry point by task

| File | Purpose | Inputs and outputs |
|---|---|---|
| [scripts/enhance_images.py](../scripts/enhance_images.py) | Process an image or directory. By default, generate references, search for Gamma and enhance at the original resolution; reference-only output is also available. | Images and enhancement weights → PNGs and enhancement.json |
| [scripts/enhance_video.py](../scripts/enhance_video.py) | Process video or camera input with background Gamma estimation and frame-by-frame LUT enhancement; includes playback pacing, comparison previews and exit handling. | Video/camera and enhancement weights → MP4 and a JSON record |
| [scripts/extract_roi.py](../scripts/extract_roi.py) | Locate and crop melt-pool ROIs using contour and HSV-based scoring. | Images → ROI PNGs and roi.json |
| [scripts/train_enhancement.py](../scripts/train_enhancement.py) | Read configuration and training images, then invoke enhancement training, optionally loading initial weights as configured. | Unlabeled image directory → per-epoch weights and config.json |
| [scripts/train_classifier.py](../scripts/train_classifier.py) | Train ConvNeXt-Tiny independently using labeled folders and optional CSV subsets; does not create splits or enhance images. | Labeled images, split and configuration → best.pth, last.pth, training.csv and config.json |
| [scripts/evaluate_classifier.py](../scripts/evaluate_classifier.py) | Load a classifier checkpoint and report per-image predictions, probabilities and metrics; optional CSV subset selection is supported. | Labeled images and classification weights → predictions.csv and metrics.json |
| [scripts/reproduce_classification.py](../scripts/reproduce_classification.py) | Run configured historical evaluation cases and compare against recorded metrics. Requires additional historical images and classifier checkpoints; the public dataset alone is insufficient for all cases. | Historical material root and case → evaluation outputs and reference_comparison.json |

For example, list all image-enhancement options:

```powershell
.\.venv\Scripts\python.exe -m scripts.enhance_images --help
```

## src: implementations called by the entry points

| File | Responsibility |
|---|---|
| [src/common.py](../src/common.py) | Shared configuration, device selection, image discovery, strict checkpoint loading and JSON output. |
| [src/enhancement/model.py](../src/enhancement/model.py) | EnhanceNetLight architecture: depthwise-separable convolutions, SE blocks and iterative curves. Returns an intermediate image, final reference and curve parameters. |
| [src/enhancement/losses.py](../src/enhancement/losses.py) | The six enhancement-training losses: color, spatial consistency, exposure, total variation, chroma and illumination. |
| [src/enhancement/training.py](../src/enhancement/training.py) | Enhancement image loading, model initialization, loss combination, optimizer, learning-rate schedule and checkpoint saving. |
| [src/enhancement/gamma.py](../src/enhancement/gamma.py) | Network reference generation, Gamma grids, grayscale MSE comparison, window sampling and LUT correction. |
| [src/enhancement/stream.py](../src/enhancement/stream.py) | Asynchronous video scheduling: collect windows, submit estimation, apply the latest completed Gamma and skip new full windows while the worker is busy. |
| [src/classification/core.py](../src/classification/core.py) | Construct ConvNeXt-Tiny, define train/evaluation transforms, read labeled folders and CSV subsets, and compute three-class metrics. |
| [src/preprocessing/roi.py](../src/preprocessing/roi.py) | ROI configuration, contour/HSV scoring, crop coordinates and handling of images without a valid region. |

To locate code, follow the calls: image enhancement uses model.py and gamma.py; video enhancement adds scheduling through stream.py. The two training entry points use separate enhancement-training and classification implementations, rather than joint training.

The following `__init__.py` files mark Python packages. They are currently empty and support module imports; they are not additional algorithms or runnable entry points:

[scripts/__init__.py](../scripts/__init__.py), [src/__init__.py](../src/__init__.py), [src/enhancement/__init__.py](../src/enhancement/__init__.py), [src/classification/__init__.py](../src/classification/__init__.py), [src/preprocessing/__init__.py](../src/preprocessing/__init__.py).

## tests: functional checks

| File | What it checks |
|---|---|
| [tests/test_dataset_splits.py](../tests/test_dataset_splits.py) | Check class order, case-insensitive folder names, CSV subsets, duplicate/missing IDs and invalid options; exercise training/evaluation entry points using temporary images and a tiny model. |
| [tests/test_video_demo.py](../tests/test_video_demo.py) | Check video FPS, overrides, output size, waiting/active states and Q/Esc/window-close handling. Uses a short synthetic video and simulated GUI events; no camera is required. |

Run all 13 existing checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These checks validate program behavior; they do not retrain the paper models or verify the reported classification accuracy. Video checks create small files under outputs/video_demo_tests/, and some GUI behavior is tested with simulated events.

## configs: configuration files

| File | Purpose |
|---|---|
| [configs/enhancement_train.json](../configs/enhancement_train.json) | Standard enhancement-network training settings. |
| [configs/gamma_realtime.json](../configs/gamma_realtime.json) | Default image/video settings: initial Gamma range 1.1–3.0, four-second windows and 15 samples. |
| [configs/gamma_paper.json](../configs/gamma_paper.json) | Initial Gamma search range of 0.1–5.1 described in the paper. |
| [configs/roi_historical.json](../configs/roi_historical.json) | Contour/HSV ROI extraction parameters. |
| [configs/classification_historical.json](../configs/classification_historical.json) | Recovered historical branch: 30 epochs, momentum 0.9 and a learning-rate step every seven epochs. |
| [configs/classification_paper.json](../configs/classification_paper.json) | Paper-settings example: 50 epochs, momentum 0.5 and a learning-rate step every ten epochs; separate from the historical configuration. |
| [configs/enhancement_install_check.json](../configs/enhancement_install_check.json) | One-epoch CPU enhancement training for installation checks. |
| [configs/classification_install_check.json](../configs/classification_install_check.json) | One-epoch classification training for installation checks. |
| [configs/classification_reproductions.json](../configs/classification_reproductions.json) | Relative paths, preprocessing and expected metrics for historical classification cases. |

Subsequent local Gamma searches use the previous Gamma ±0.5, with a lower bound of 0.1. The two classification configurations describe different training settings; choose the one appropriate to your experiment.

## Installation and data locations

Python 3.12 is recommended. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then install CPU dependencies:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --torch-backend cpu
```

Bundled examples are in data/examples/. The image ZIP contains dataset-images/ at its top level; the video ZIP contains dataset-videos/. Extract both into data/ without merging or overwriting files. The code package's two empty dataset directories contain only .gitkeep placeholders. Videos are optional for image experiments.

Image metadata and four historical split lists are in data/dataset-images/. No default split is selected. See README.md and the CSV files in splits/ inside the downloaded package for contents, classes and citation information, or start with the [code-side data guide](../data/README.md).

## Common commands

### Image enhancement

Run the three examples without retraining:

```powershell
.\.venv\Scripts\python.exe -m scripts.enhance_images --input data/examples/images --output outputs/examples/enhanced --weights weights/enhancement_realtime.pth --device cpu
```

Replace the input with your own image or directory, keeping the output outside the input directory. Gamma is estimated independently for each image; a directory is not treated as a temporal window. Outputs preserve relative directories, append .png to filenames and include enhancement.json.

Add `--mode reference` to save only 224×224 network references. The default Gamma mode saves final enhancement at the original resolution. See the [example directory](../data/examples/) for original images, ROIs and expected outputs.

### Video enhancement

```powershell
.\.venv\Scripts\python.exe -m scripts.enhance_video --source data/examples/videos/normal_video.mp4 --output outputs/examples/video_comparison.mp4 --weights weights/enhancement_realtime.pth --realtime --side-by-side --show --device cpu
```

--realtime paces file playback at the source FPS; --side-by-side saves the comparison; --show opens a preview. Press Q or Esc, or close the window to stop. Omit --show for headless processing. Without --side-by-side, output is a clean enhanced video. --preview-scale only affects preview/comparison size, while --fps overrides the effective frame rate.

Defaults are four-second windows and 15 samples, starting with Gamma 1. Updates are applied when background computation finishes; the latest available value remains active in the meantime. New full windows are skipped while the worker is busy, partial final windows do not trigger estimation, and written frames are not revised. The entry point neither sets exposure nor crops ROIs. Use --source 0 for the default camera without --realtime; physical-camera operation was not tested during release preparation.

### ROI extraction

```powershell
.\.venv\Scripts\python.exe -m scripts.extract_roi --input data/examples/originals --output outputs/roi --config configs/roi_historical.json
```

The default crop is 448×448, clipped at image borders without padding. Images without a contour are logged as no_roi and skipped; roi.json records crop coordinates.

### Enhancement-network training

Prepare an RGB image directory, for example [SICE](https://github.com/csjcai/SICE), obtained under its applicable terms. Nested directories are supported and state labels are unnecessary. SICE training images are not bundled. Replace the placeholder path below:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_enhancement --images "path/to/training/images" --output outputs/enhancement --config configs/enhancement_train.json
```

Defaults are 224×224 inputs, batch size 8 and 200 epochs, saving a checkpoint each epoch. --cpu forces CPU training; classification losses are not used. For a one-epoch installation check, select configs/enhancement_install_check.json instead.

### Classification training and evaluation

The classifier is trained independently of enhancement. Each image root contains excessive, insufficient and normal folders (initial capitals are also accepted), indexed 0, 1 and 2. Choose original, ROI or pre-enhanced images and train each scheme separately. The program does not enhance images or generate splits automatically.

After downloading the image package, select a historical CSV split explicitly. The following performs real training, rather than a read-only check. --init imagenet downloads pretrained weights if uncached; --init random initializes randomly:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_classifier --train data/dataset-images/images/original --val data/dataset-images/images/original --split-file data/dataset-images/splits/historical_original_80_10_10.csv --train-subset train --val-subset val --output outputs/classifier --config configs/classification_historical.json --init imagenet
.\.venv\Scripts\python.exe -m scripts.evaluate_classifier --images data/dataset-images/images/original --split-file data/dataset-images/splits/historical_original_80_10_10.csv --subset test --weights outputs/classifier/best.pth --output outputs/evaluation --preprocessing center_crop
```

Training and validation may share an image root, but their selected CSV members must not overlap. Alternatively, pass separate existing folders without a CSV. This split is a usage example, not the project's default. Evaluation requires a classifier checkpoint for the relevant image scheme; the bundled enhancement checkpoint cannot substitute for it.

center_crop means Resize(256)+CenterCrop(224); resize directly scales to 224×224. Both use ImageNet normalization. Confusion-matrix rows are true classes and columns are predictions; metrics also include accuracy, per-class precision/recall/F1 and macro F1.

### Historical classification checks

--list only displays cases and does not load weights or images:

```powershell
.\.venv\Scripts\python.exe -m scripts.reproduce_classification --list
.\.venv\Scripts\python.exe -m scripts.reproduce_classification --source-root "path/to/historical/materials" --case divided_val --device cpu
```

The second command requires additional historical material whose relative paths are defined in configs/classification_reproductions.json. --case supports scheme1, scheme2, scheme3, scheme4, divided_val and all. Without those materials, skip this entry point; image/video enhancement, retraining and ordinary classifier evaluation remain usable.
