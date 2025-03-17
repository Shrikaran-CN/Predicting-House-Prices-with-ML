# Predicting-House-Prices-with-ML

## Overview

This project aims to build a machine learning model to predict house prices based on various features such as location, size, number of bathrooms, and total square feet area. The project walks through data cleaning, feature engineering, outlier removal, and model building using multiple algorithms to select the best-performing model.


## Dataset

The dataset contains 13,320 entries with the following columns:

* `area_type` : Type of area (e.g., Built-up Area, Carpet Area)
* `availability` : Availability status (e.g., Ready to Move)
* `location` : Location of the property
* `size` : Number of bedrooms (e.g., 2 BHK)
* `society` : Name of the society
* `total_sqft` : Total area in square feet
* `bath` : Number of bathrooms
* `balcony` : Number of balconies
* `price` : Price in lakhs (Indian Rupees)


## Data Cleaning & Feature Engineering

1. **Dropped unnecessary columns:** `area_type`, `society`, `balcony`, and `availability`.
2. **Handled missing values:** Dropped rows with missing data.
3. **Created 'BHK' column:** Extracted number of bedrooms from the `size` column.
4. **Converted 'total\_sqft' to numerical:** Handled ranges (e.g., '1133-1384') by taking the average.
5. **Calculated 'price\_per\_sqft'**: Derived a new feature for price per square foot.
6. **Location simplification:** Grouped locations with less than 10 entries into 'Other'.


## Outlier Removal

1. **Minimum square feet per BHK:** Removed properties with less than 300 sq. ft per bedroom.
2. **Price per square foot outliers:** Removed extreme outliers based on mean and standard deviation per location.
3. **BHK price anomaly removal:** Removed cases where a larger BHK has a lower price per square foot than a smaller BHK within the same location.
4. **Bathroom count filtering:** Removed properties with bathrooms exceeding BHK count by more than 1.


## Model Building

### Data Preparation

* **One-hot encoding:** Applied to `location`.
* **Train-test split:** 80% training, 20% testing.

### Model Training

* **Linear Regression:** Base model achieved 86.9% accuracy.
* **K-Fold Cross Validation:** Validated performance with 5 splits.
* **GridSearchCV:** Tested `Linear Regression`, `Lasso`, and `Decision Tree` models to find the best model and parameters.

| Model             | Best Score | Best Parameters                                      |
| ----------------- | ---------- | ---------------------------------------------------- |
| Linear Regression | 0.8537     | {'fit\_intercept': False}                            |
| Lasso             | 0.7275     | {'alpha': 1, 'selection': 'random'}                  |
| Decision Tree     | 0.7263     | {'criterion': 'friedman\_mse', 'splitter': 'random'} |

Linear Regression performed best.


## Price Prediction Function

A custom function `predict_price()` was created to take inputs (`location`, `sqft`, `bath`, `bhk`) and return the predicted price.

```python
predict_price('1st Phase JP Nagar', 1000, 2, 2)
```


## Exporting the Model

* **Pickle file:** Saved the trained model (`house_prices_model.pickle`).
* **JSON file:** Saved column information (`columns.json`) for later use in web applications.


## Technologies Used

* **Python:** Pandas, NumPy, Matplotlib
* **Machine Learning:** Scikit-learn
* **Jupyter Notebook**


## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
2. Run the Jupyter Notebook.
3. Use the `predict_price()` function to test predictions.
4. Deploy the saved model (`house_prices_model.pickle`) in your web app or API.


## Future Improvements

* **More features:** Incorporate amenities, age of property, etc.
* **Advanced models:** Try Gradient Boosting, XGBoost, or Neural Networks.
* **Web integration:** Build a web interface to make predictions user-friendly.


## Contributors

* **Shrikaran C N**
* **Dhriti Wahi**

Feel free to contribute or report issues!



