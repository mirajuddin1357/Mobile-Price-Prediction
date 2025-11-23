<center><h2><span style="font-weight:bolder; color: red; font-size:120%",>📱Mobile Price Prediction Using Machine Learning Algorithms (Regression)</span></h2></center>

---

<div style="
  text-align:center;
  background: linear-gradient(135deg, #000000, #1a1a1a);
  border:2px solid #ff0800ff;
  border-radius:18px;
  padding:25px;
  box-shadow:0 0 25px rgba(255,215,0,0.3);
  transition:all 0.4s ease;
">
  <img src="image.png"
       width="680"
       style="
         border-radius:15px;
         box-shadow:0 0 35px rgba(255,215,0,0.4);
         transition: transform 0.4s ease, box-shadow 0.4s ease;
       "
       onmouseover="this.style.transform='scale(1.0)'; this.style.boxShadow='0 0 50px rgba(255,215,0,0.9)';"
       onmouseout="this.style.transform='scale(1.0)'; this.style.boxShadow='0 0 30px rgba(255,215,0,0.4)';"
  >
  <p style="
     color:#FFD700;
     font-size:20px;
     font-family:'Poppins', sans-serif;
     font-weight:600;
     margin-top:12px;
     letter-spacing:1px;
     text-shadow:0 0 10px rgba(255,215,0,0.8);
  ">
  </p>
</div>

---

<div style="
    text-align:center;
    background: linear-gradient(135deg, #000000, #1a1a1a);
    border:2px solid #ff0800;
    border-radius:18px;
    padding:25px;
    box-shadow:0 0 25px rgba(255,215,0,0.3);
    transition:all 0.4s ease;
">
<h1 style='text-align:center; color:white; background:#111; padding:18px; border-radius:8px;'>
        📱 Mobile Price Insights
</h1>
<img src="data:image/png;base64,{encoded_string}" width="680" 
    style="
        border-radius:15px;
         box-shadow:0 0 35px rgba(255,215,0,0.4);
         transition: transform 0.4s ease, box-shadow 0.4s ease;
     " 
     onmouseover="this.style.transform='scale(1.02)'; this.style.boxShadow='0 0 50px rgba(255, 215, 0, 1)';" 
     onmouseout="this.style.transform='scale(1.0)'; this.style.boxShadow='0 0 30px rgba(255,215,0,0.4)';">
</div>

---

<div
  style="border-radius:10px; border:#8B0000 solid; padding:15px; background-color:#FAF3F3; font-size:100%; text-align:left; color:#222;">
  <h3 align="left" style="color:#8B0000;">Introduction</h3>
  <p style="margin:6px 0 0; line-height:1.6;"> This project focuses on
    <strong>predicting the price of mobile phones</strong> using various <strong>Machine Learning algorithms</strong>.
    The dataset used in this project contains multiple features of different smartphones such as <em>RAM</em>,
    <em>Battery Capacity</em>, <em>Processor Type</em>, <em>Screen Size</em>, and other hardware specifications, along
    with their respective <strong>target prices</strong>. The primary goal of this project is to build a strong
    <strong>regression model</strong> that can accurately predict the <strong>mobile price</strong> based on the given
    features. By analyzing the relationship between these technical attributes and the price, this project aims to help
    users and manufacturers understand how different specifications affect the overall market price of a smartphone.
  </p>
</div>

---

### Project structure
```text

📁 Mobile-Price-Prediction-Using-ML-Algorithms/
│
├── smartphone.csv                                    # Individual smartphone dataset
├── smartphone_cleaned_v1.csv                         # Cleaned smartphone dataset
├── image.png                                         # Project Image
├── mobile-price-prediction-using-ml-algorithms.ipynb # Main Jupyter Notebook
├── app.py                                            # Streamlit Web App for deployment
├── smartphone_price_model.pkl                        # Trained ML model file
└── README.md                                         # Full project documentation
```

---

<div
  style="border-radius:10px; border:#8B0000 solid; padding:15px; background-color:#FAF3F3; font-size:100%; text-align:left; color:#222;">
  <h3 align="left" style="color:#8B0000;">Used dataset: <a href="https://www.kaggle.com/datasets/abdurrahman22224/smartphone-new-data" target="_blank">Click here for the Mobile Price Prediction dataset (Kaggle)</a>
  </h3>
  <ul style="list-style-type:none; padding-left:0; line-height:1.6;">
  </ul>
</div>

---

<div
  style="border-radius:10px; border:#8B0000 solid; padding:15px; background-color:#FAF3F3; font-size:100%; text-align:left; color:#222;">
  <h3 align="left" style="color:#8B0000;">Dataset Information</h3>
  <ul style="list-style-type:none; padding-left:0; line-height:1.6;">
    <li>
      <strong>This dataset contains 987 rows, with details about 25 features.</strong>
    </li>
    <li>
      <strong>There are some empty or NaN and duplicates values in this dataset.</strong>
    </li>
  </ul>
</div>

---

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
  <div class="container">
    <h3>Smartphone Dataset Attributes</h3>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>No.</th>
            <th>Attribute</th>
            <th>Feature Description</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td>
            <td>brand_name</td>
            <td>Brand name of the smartphone</td>
            <td>Categorical</td>
          </tr>
          <tr>
            <td>2</td>
            <td>model</td>
            <td>Model name of the smartphone</td>
            <td>Categorical</td>
          </tr>
          <tr>
            <td>3</td>
            <td>rating</td>
            <td>Average user rating (out of 5)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>4</td>
            <td>has_5g</td>
            <td>Indicates if 5G connectivity is supported (1=True, 0=False)</td>
            <td>Boolean</td>
          </tr>
          <tr>
            <td>5</td>
            <td>has_nfc</td>
            <td>Indicates if NFC is supported (1=True, 0=False)</td>
            <td>Boolean</td>
          </tr>
          <tr>
            <td>6</td>
            <td>has_ir_blaster</td>
            <td>Indicates if IR Blaster is available</td>
            <td>Boolean</td>
          </tr>
          <tr>
            <td>7</td>
            <td>processor_brand</td>
            <td>Brand of processor (e.g., Qualcomm, MediaTek, Exynos)</td>
            <td>Categorical</td>
          </tr>
          <tr>
            <td>8</td>
            <td>num_cores</td>
            <td>Number of processor cores</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>9</td>
            <td>processor_speed</td>
            <td>Processor clock speed (in GHz)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>10</td>
            <td>battery_capacity</td>
            <td>Total battery capacity (in mAh)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>11</td>
            <td>fast_charging_available</td>
            <td>Fast charging power (in Watts)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>12</td>
            <td>fast_charging</td>
            <td>Fast charging power (in Watts)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>13</td>
            <td>ram_capacity</td>
            <td>RAM size (in GB)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>14</td>
            <td>internal_memory</td>
            <td>Internal storage capacity (in GB)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>15</td>
            <td>screen_size</td>
            <td>Screen size (in inches)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>16</td>
            <td>refresh_rate</td>
            <td>Display refresh rate (in Hz)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>17</td>
            <td>resolution</td>
            <td>Display resolution (e.g., 2400×1080)</td>
            <td>Categorical</td>
          </tr>
          <tr>
            <td>18</td>
            <td>num_rear_cameras</td>
            <td>Number of rear cameras</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>19</td>
            <td>num_front_cameras</td>
            <td>Number of front cameras</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>20</td>
            <td>os</td>
            <td>Operating system (Android / iOS)</td>
            <td>Categorical</td>
          </tr>
          <tr>
            <td>21</td>
            <td>primary_camera_rear</td>
            <td>Main rear camera resolution (in Megapixels)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>22</td>
            <td>primary_camera_front</td>
            <td>Main front camera resolution (in Megapixels)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>23</td>
            <td>extended_memory_available</td>
            <td>Indicates if expandable memory slot is available</td>
            <td>Boolean</td>
          </tr>
          <tr>
            <td>24</td>
            <td>extended_upto</td>
            <td>Maximum expandable storage supported (in GB/TB)</td>
            <td>Numeric</td>
          </tr>
          <tr>
            <td>25</td>
            <td>price</td>
            <td>Price of the smartphone (Target Variable)</td>
            <td>Numeric</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>

---

 ### Project Tools

* **Programming:** Python
* **Data Handling:** NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn, Plotly
* **Machine Learning Tools:** Scikit-Learn, XGBoost, LightGBM
* **Coding Platforms:** VS Code, Jupyter Notebook
* **Version Control:** Git, GitHub
* **Web & Markdowns:** Streamlit, HTML, CSS
---

### Project Workflow (Steps)

1. **Import Libraries**

- Import Libraries `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly`, `sklearn`, and all the necessary ML algorithms.

2. **Introduction**:

- Defined the project objectives and scope.
- About the dataset and its features.

---

3. **Data Collection**:

- Load the dataset.
- Loaded the dataset into a Pandas DataFrame.
- Displayed the first few rows of the dataset to understand its structure.

---

4. **Data Preprocessing**:

- Handled missing values, duplicates values, outliers, fix datatypes and performed feature engineering.
- Encoded categorical variables and scaled numerical features.
- Split the dataset into training and testing sets.
- Data cleaning

---

5. **Feature Engineering**:
- Extract only required Information from some column's
- Created new features based on existing ones.
- Remove some features

---

6. **Exploratory Data Analysis (EDA)**:

- Analyze dataset shape, data types, and distributions and info.
- Univariate Analysis
  - Categorical
  - Numerical
- Bivariate Analysis
  - Categorical
  - Numerical
- Multivariate Analysis
- Visualize key patterns using histograms, boxplots, pie, and pairplots.
- Identify correlations between features and target variable using a heatmap

---

7. **Applied Machine Learning Model (Regression's)**:

- Trained multiple regression models including 
1. Linear Regression
2. Support Vector Regressor
3. Decision Tree Regressor
4. K-Nearest Neighbors Regressor
5. Bagging Regressor
6. Voting Regressor
7. Random Forest Regressor
8. LightGBM Regressor
9. Stacking Regressor
10. Adaboost Regressor
11. Gradient Boosting Regressor
12. Extra Tree Regressor
13. XGBoost Regressor
14. CatBoost Regressor
15. ElasticNet Regressor
16. Lasso Regression
17. Ridge Regression
18. Huber Regressor
19. Kernal Ridge Regressor
20. Multi-Layer Perceptron Regressor

---

1. **Model Selection and Comparison**:
- Use **optuna** for hyperparameter tuning to optimize model performance, find best model and best parameters.
- Compared the performance of different models to select the best one.

---

<div style="border-radius:10px; border:#8B0000 solid 2px; padding:15px; background-color:#FAF3F3; color:#222; overflow-x:auto;">
  <h3 align="center" style="color:#8B0000;">Model's Comparison</h3>
  <table style="border-collapse:collapse; width:100%; min-width:700px; font-size:15px; text-align:left;">
    <thead style="background-color:#F5D7D7;">
      <tr>
        <th style="padding:10px; border-bottom:2px solid #8B0000;">#</th>
        <th style="padding:10px; border-bottom:2px solid #8B0000;">Model Name</th>
        <th style="padding:10px; border-bottom:2px solid #8B0000;">R² Score</th>
        <th style="padding:10px; border-bottom:2px solid #8B0000;">Mean Absolute Error (MAE)</th>
        <th style="padding:10px; border-bottom:2px solid #8B0000;">Mean Squared Error (MSE)</th>
      </tr>
    </thead>
    <tbody>
      <tr style="font-weight:bold; color:#8B0000;">
        <td style="padding:10px;">1</td>
        <td style="padding:10px;">Extra Trees Regressor</td>
        <td style="padding:10px;">0.8851</td>
        <td style="padding:10px;">2907</td>
        <td style="padding:10px;">18612758</td>
      </tr>
      <tr>
        <td style="padding:10px;">2</td>
        <td style="padding:10px;">Bagging Regressor</td>
        <td style="padding:10px;">0.8816</td>
        <td style="padding:10px;">3033</td>
        <td style="padding:10px;">19176969</td>
      </tr>
      <tr>
        <td style="padding:10px;">3</td>
        <td style="padding:10px;">Random Forest Regressor</td>
        <td style="padding:10px;">0.8808</td>
        <td style="padding:10px;">3040</td>
        <td style="padding:10px;">19307148</td>
      </tr>
      <tr>
        <td style="padding:10px;">4</td>
        <td style="padding:10px;">Gradient Boosting Regressor</td>
        <td style="padding:10px;">0.8756</td>
        <td style="padding:10px;">3120</td>
        <td style="padding:10px;">20151321</td>
      </tr>
      <tr>
        <td style="padding:10px;">5</td>
        <td style="padding:10px;">XGBoost Regressor</td>
        <td style="padding:10px;">0.8890</td>
        <td style="padding:10px;">2969</td>
        <td style="padding:10px;">17979785</td>
      </tr>
      <tr>
        <td style="padding:10px;">6</td>
        <td style="padding:10px;">CatBoost Regressor</td>
        <td style="padding:10px;">0.8708</td>
        <td style="padding:10px;">2920</td>
        <td style="padding:10px;">20931812</td>
      </tr>
    </tbody>
  </table>
</div>

---

9. **Conclusion and Future Work**:

- Summarized the findings and performance of the best model.
- Discussed potential improvements and future work, such as hyperparameter tuning and using more advanced models.

---

10. **Model Deployment**:

- Deployed the best-performing model using Streamlit, HTML and CSS to create a web application.
- Users can input smartphone features and get price predictions in real-time.





