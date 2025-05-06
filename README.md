#  Uber Eats Restaurant Rating Prediction

This project implements a self-training approach using pseudo-labeling to improve restaurant score predictions based on menu and metadata. The dataset is sourced from [Kaggle - Uber Eats USA Restaurants Menus](https://www.kaggle.com/datasets/ahmedshahriarsakib/uber-eats-usa-restaurants-menus).

### 🧠 Project Structure

* **`pre-processing.ipynb`**: Loads and cleans the raw dataset, extracting relevant features such as `price_range`, `ratings`, and `category`. Missing value indicators are added.
* **`comparison.py`**: Sweeps across different pseudo-label weights (0.0 to 1.0), evaluating RMSE on the validation set and visualizing the effect of weight tuning.
* **`training.ipynb`**: Trains a baseline CatBoost regression model on labeled data, applies pseudo-labeling to unlabeled rows, and retrains using various pseudo-label weights.

### 📁 Dataset

**Source**: [Kaggle - Uber Eats USA Restaurants Menus](https://www.kaggle.com/datasets/ahmedshahriarsakib/uber-eats-usa-restaurants-menus)

After preprocessing, each restaurant entry includes:

* `score`: Numerical rating (target variable, partially missing).
* `price_range`: Numerical encoding of price tier.
* `ratings`: Aggregated user ratings.
* `category`: Type of cuisine.
* `*_missing`: Binary flags for missingness of target or features.

### ⚙️ Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

### 🚀 Usage

**1. Preprocessing**

Run the preprocessing notebook:

```bash
jupyter notebook pre-processing.ipynb
```

**2. Pseudo-label Weight Sweep**

Run the grid search over pseudo-label weights:

```bash
python comparison.py --data restaurants_cleaned.csv
```

**3. Training**

Run baseline and self-training models:

```bash
jupyter notebook training.ipynb
```

This will generate:

* `rmse_vs_weight.png`: RMSE vs. pseudo-label weight plot.
* `rmse_vs_weight.csv`: Numerical results.
* `rmse_vs_weight.pkl`: Serialized results for further use.

### 📈 Goal

Evaluate the effect of incorporating pseudo-labeled data on model performance and determine the optimal balance between real and synthetic labels.