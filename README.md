# Leveraging Artificial Intelligence for Project Risk Prediction and Mitigation

Code for my **M.Sc. Computer Science thesis** (IU International University of Applied Sciences, 2024). The project predicts the stability category of a supply chain from its risk indicators, so that project and supply chain managers can see risk early and act before it becomes a problem.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-model-189FDD)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?logo=flask&logoColor=white)

## What it does

1. **Preprocessing** (`data_preprocessing.py`): fills missing values, builds degree-2 polynomial features from the risk indicators and cost, scales them, and maps the stability categories to class labels.
2. **Training** (`model_training.py`): trains logistic regression, a decision tree, random forest, gradient boosting and XGBoost, then combines them in a stacking classifier.
3. **Evaluation** (`model_evaluation.py`): reports accuracy and macro precision, recall and F1 for every model.
4. **Prediction API** (`flask_app.py`): serves the stacking model through a `POST /predict` endpoint.

`main.py` runs steps 1 to 3 in order.

## Data

The [SCRM time-series dataset (2018)](dataset/license.txt) by H. Banerjee et al., SRM Institute of Science & Technology, with risk indices for supplier, distributor, manufacturer and retailer, total cost, and a supply chain stability category. It is redistributed here under its license, which is included in `dataset/license.txt`.

| Feature | Meaning |
|---|---|
| `RI_Supplier1`, `RI_Distributor1`, `RI_Manufacturer1`, `RI_Retailer1` | Risk index at each stage of the supply chain |
| `Total_Cost` | Total cost at that point in time |
| `SCMstability_category` | Target: supply chain stability class |

## Results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Logistic regression | 0.987 | 0.500 | 0.493 | 0.497 |
| Decision tree | 0.987 | 0.500 | 0.493 | 0.497 |
| Random forest | 0.987 | 0.500 | 0.493 | 0.497 |
| Gradient boosting | 0.987 | 0.500 | 0.493 | 0.497 |
| XGBoost | 0.987 | 0.500 | 0.493 | 0.497 |
| Stacking classifier | 1.000 | 1.000 | 1.000 | 1.000 |

Screenshots of every step are in [`docs/results`](docs/results).

### How to read these numbers

- The base models reach high accuracy but a macro F1 of about 0.50. The classes are strongly imbalanced, so a model can score well on accuracy while missing the rarer, riskier classes.
- The stacking model scored perfectly on this test split. A perfect score on real-world data usually means the split is too easy or information leaks between training and test data, so this result needs further validation before it can be trusted.
- Next steps I would take: stratified and time-based validation, class weighting or resampling for the minority classes, and confusion matrices per class.

## Run it

```bash
pip install -r requirements.txt
python main.py          # preprocess, train and evaluate (the training set is large, so this takes a while)
python flask_app.py     # start the prediction API on http://127.0.0.1:5000
```

Example request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"RI_Supplier1": 1.6396, "RI_Distributor1": 1.0422, "RI_Manufacturer1": 3.1884, "RI_Retailer1": 3.3609, "Total_Cost": 133.06}]}'
```

## Author

**Manicharan Ravi** · IT Specialist & Cloud Engineer · M.Sc. Computer Science · Mainz, Germany
[LinkedIn](https://linkedin.com/in/manicharan-ravi) · [GitHub](https://github.com/manicharan0485)
