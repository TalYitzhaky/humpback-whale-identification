# Humpback Whale Identification

Image classification experiments for identifying individual humpback whales from photographs of their tails. The project uses the Kaggle `humpback-whale-identification` competition dataset and compares a custom PyTorch CNN with transfer learning and a classical machine learning baseline.

## Project files

| File | Description |
| --- | --- |
| [humpback-whale-identification.ipynb](humpback-whale-identification.ipynb) | Main notebook containing data exploration, training, evaluation, and discussion. |
| [humpback-whale-identification-outputs.html](humpback-whale-identification-outputs.html) | Exported notebook outputs; download and open in a browser to view. |
| [report.pdf](report.pdf) | Project report. |

The dataset and trained model checkpoints are not included in this repository.

## Experiments

- **Data exploration:** class imbalance, image dimensions, and sample images.
- **Custom CNN:** three convolutional blocks, 128 × 128 input images, and five-fold stratified cross-validation.
- **Augmentation:** random rotation, color jitter, and test-time prediction averaging, with examples of correct and incorrect predictions.
- **Transfer learning:** ResNet18, DenseNet121, MobileNetV2, and VGG16 with 224 × 224 inputs. Each model trains its classification head before fine-tuning all layers. This section reserves 10% of the selected data for testing and uses five-fold cross-validation on the remainder.
- **Feature extraction:** pretrained DenseNet121 features reduced to 50 dimensions with PCA, followed by a 100-tree random forest.

The notebook selects 15 frequent whale IDs by skipping the most frequent dataset label, which it assumes is `new_whale`. Initial CNN experiments withhold `w_23a388d`, leaving 14 represented identities; a later experiment adds it back. These experiments evaluate a selected subset of known whales, rather than the full competition task.

## Running in Google Colab

The notebook is written for Google Colab and uses `/content` paths and Google Drive integration. A GPU runtime is recommended for training; the model code falls back to CPU when CUDA is unavailable.

1. Open `humpback-whale-identification.ipynb` in Colab and select a GPU runtime.
2. Obtain the Kaggle competition data using an account with access to it.
3. Choose **one** of the data-loading approaches in the opening cells:
   - Mount Google Drive and copy an extracted `humpback-whale-identification` folder from `MyDrive`.
   - Use the Kaggle download cell, supplying your own `kaggle.json` credentials when prompted.
   - Mount Google Drive and extract `MyDrive/humpback-whale-identification.zip` using the ZIP setup cell.

   Skip the other setup approaches. The Drive ZIP cell deletes and recreates `/content/humpback`, so running every setup cell in sequence can remove previously prepared data. Keep Kaggle credentials out of version control.

4. Confirm that the extracted data has this layout:

   ```text
   /content/humpback/
   ├── train.csv
   └── train/
       ├── <image filename>.jpg
       └── ...
   ```

   `train.csv` must contain the `Image` and `Id` columns. Images are resolved relative to `TRAIN_DIR`.

5. Ensure the notebook dependencies are available. If needed, run this in a notebook cell:

   ```python
   %pip install torch torchvision numpy pandas Pillow matplotlib seaborn scikit-learn tqdm kaggle
   ```

6. Run the imports and configuration cell, then the remaining analysis and experiment cells in order. Later cells reuse datasets, models, and validation splits created earlier. Pretrained models download their weights on first use, requiring internet access.

## Configuration

The main settings are defined near the beginning of the notebook:

| Setting | Default |
| --- | --- |
| `TRAIN_DIR` | `/content/humpback/train` |
| `CSV_PATH` | `/content/humpback/train.csv` |
| `NUM_CLASSES_TO_USE` | `15` |
| `BATCH_SIZE` | `32` |
| `NUM_EPOCHS` | `20` per CNN fold |
| `K_FOLDS` | `5` |

Transfer learning separately uses five epochs for head training and five for fine-tuning, with learning rates of `0.001` and `0.0001`, respectively.

To run locally, use a Python 3 Jupyter environment with the dependencies above, skip the Colab-specific Drive and upload cells, extract the data yourself, and update `TRAIN_DIR` and `CSV_PATH`. Package versions are not pinned in this repository.

## Results and interpretation

The notebook reports roughly 60% accuracy for the initial CNN and substantially higher validation accuracy with transfer learning. Its final comparison table lists DenseNet121 at approximately 97.18% validation accuracy. See the exported HTML and report for the recorded plots and comparisons.

Transfer-learning validation results reflect the best recorded validation accuracy across folds and fine-tuning epochs, rather than a cross-validation mean. The augmented CNN shares random transforms between training and validation, and the random forest section fits PCA before cross-validation. Keep these evaluation choices in mind when interpreting the reported scores. Training and augmentation are not fully seeded, so reruns can differ.
