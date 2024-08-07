# Gold Price Prediction 🧈🤑

## Project Overview
This project aims to develop and evaluate predictive models to forecast the next day's gold price. By comparing the performance of different models, we identify the most effective approach for predicting gold prices. The insights gained can help understand the strengths and limitations of various modeling techniques in the context of financial time-series data.

## Data Collection
- **Source**: The dataset is sourced from Sahil Wagh on Kaggle - [Gold Stock Prices](https://www.kaggle.com/datasets/sahilwagh/gold-stock-prices).
- **Composition**: The dataset includes daily records of gold prices with features such as Date, Open, High, Low, Close/Last, and Volume.

## Models Used
1. **Prophet**
2. **Linear Regression**
3. **Random Forest**
4. **XGBoost**
5. **LSTM**

## Key Findings
- **Linear Regression**: Performed the best with the lowest MSE and MAE, and an R² value of 0.99.
- **LSTM**: Also performed well, capturing temporal dependencies with a high R² value of 0.97.
- **Prophet**: Showed poor performance, indicating it may not be suitable for this type of data.
- **Random Forest and XGBoost**: Showed moderate performance, limited by the linear relationships in the data.

## Next Day Prediction
- **Linear Regression**: Predicted price with an absolute error of 57.73 USD.
- **LSTM**: Predicted price with an absolute error of 71.37 USD.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gold-price-prediction.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook or Python scripts to preprocess data, train models, and make predictions.

## License
This project is licensed under the Apache v2.0 License.

---

Feel free to reach out if you have any questions or suggestions!
