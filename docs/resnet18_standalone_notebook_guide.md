# ResNet-18 Standalone Notebook Guide (Kaggle)

## 1) Add dataset input
Attach the CIFAKE dataset to your Kaggle notebook.

Expected folder structure:
- `.../train/REAL`
- `.../train/FAKE`
- `.../test/REAL`
- `.../test/FAKE`

## 2) Open notebook
Open:
- `notebooks/ResNet18_Full_Training_Evaluation_Standalone.ipynb`

## 3) Fast smoke test
In the configuration cell:
- `FAST_DEV_RUN = True`
- `RUN_TRAINING = True`
- `RUN_CLEAN_EVAL = True`
- `RUN_ROBUSTNESS_EVAL = True`
- `RUN_VISUALIZATION = True`
- `RUN_INFERENCE_DEMO = True`

Run top-to-bottom.

## 4) Full training
For full experiment:
- `FAST_DEV_RUN = False`
- set desired `EPOCHS`, `BATCH_SIZE`, and sample caps (`MAX_TRAIN_EACH`, `MAX_TEST_EACH`)
- keep `RUN_TRAINING = True`

## 5) Download outputs
The final cell creates:
- `resnet18_standalone_outputs.zip`

Download it from Kaggle’s Output panel.
