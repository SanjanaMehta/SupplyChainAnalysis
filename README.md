
# Supply Chain Delivery Delay Prediction

## 🔍 Overview

This project analyzes supply chain data to predict delivery delays using machine learning techniques. The analysis includes data preprocessing, exploratory data analysis, correlation analysis, and building a Random Forest classifier to predict delivery delays.

## ✨ Features

- **Data Preprocessing**: Handles missing values and outlier removal
- **Exploratory Data Analysis**: Comprehensive analysis of supply chain metrics
- **Correlation Analysis**: Identifies relationships between different variables
- **Delay Prediction**: Machine learning model to predict delivery delays
- **Performance Metrics**: Detailed evaluation of model accuracy, precision, and recall

## 📊 Dataset

The project uses `SupplyChainDataset.csv` which contains the following key information:

### Customer Data
- Customer demographics (City, Country, State, Email, etc.)
- Customer segmentation
- Customer identification details

### Order Data
- Order details (ID, Date, Region, Status)
- Product information (Price, Discount, Quantity)
- Financial metrics (Sales, Profit)
- Shipping information (Scheduled vs Real shipping days)

### Key Target Variables
- `Delay`: Binary indicator of delivery delay
- `Late_delivery_risk`: Risk factor for late delivery
- `Days for shipping (real)`: Actual shipping duration
- `Days for shipment (scheduled)`: Planned shipping duration


### Key Steps in the Analysis

1. **Data Loading and Exploration**
   ```python
   df = pd.read_csv("SupplyChainDataset.csv")
   df.info()
   ```

2. **Data Cleaning**
   - Removes rows with missing zipcodes
   - Drops unnecessary columns
   - Filters outliers using 5th and 95th percentiles

3. **Feature Engineering**
   - Separates customer and order data
   - Creates correlation matrix for numerical features

4. **Model Training**
   - Uses Random Forest Classifier with 1000 estimators
   - 75/25 train-test split
   - Predicts delivery delays

## 📈 Model Performance

The Random Forest model achieves:
- **High Accuracy**: Detailed in confusion matrix
- **Weighted Precision**: Accounts for class imbalance
- **Weighted Recall**: Comprehensive evaluation metric

### Confusion Matrix
The model generates a detailed confusion matrix showing prediction accuracy across different delay categories.

## 🔍 Key Findings

Based on the correlation analysis:

1. **Delivery Delays** are most dependent on:
   - Late delivery risk factors
   - Real shipping days

2. **Sales Performance** shows:
   - Weak correlation with benefit per order
   - Moderate correlation with product price and discounts

3. **Data Quality**:
   - 3 rows removed due to missing zipcode data
   - Outlier removal improves model stability



## 🔧 Configuration

### Model Parameters
- **Random Forest Estimators**: 1000 trees
- **Test Size**: 25% of the dataset
- **Outlier Threshold**: 5th and 95th percentiles

### Data Processing
- **Missing Data**: Rows with null zipcodes are removed
- **Feature Selection**: Excludes object-type columns for modeling
- **Target Variable**: `Delay` column for binary classification

## 📊 Visualizations

The project generates several visualizations:
- Correlation heatmap showing relationships between numerical variables
- Distribution plots for key metrics
- Confusion matrix for model evaluation


## 📝 Future Improvements

- [ ] Add cross-validation for more robust model evaluation
- [ ] Implement feature importance analysis
- [ ] Add time-series analysis for seasonal patterns
- [ ] Include additional algorithms (XGBoost, SVM)
- [ ] Create interactive dashboard for results visualization

---

*This project demonstrates the application of machine learning in supply chain optimization and delivery prediction.*
