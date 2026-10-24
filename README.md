# UK Crop Yield Prediction

A Streamlit interface for the saved Random Forest, Linear Regression, and Neural Network crop-yield models.

## Run locally

Use Python 3.11 in a virtual environment. The scikit-learn artefacts were saved with scikit-learn 1.6.1, so the pinned version in `requirements.txt` is intentional.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. Select Python 3.11 in the deployment settings if available.
5. Deploy and test all three model choices.

## Model files and data

The saved model artefacts are included in `models/`. The training notebook is in `training/`. The original training dataset is not required to run predictions from user-entered inputs.

## Important validation note

The region and crop dropdowns were carried over from the notebook's interface. Before treating predictions as reliable, verify that the labels exactly match those used in the training data and test predictions against the original notebook. The app should be considered a demo until this validation is complete.
