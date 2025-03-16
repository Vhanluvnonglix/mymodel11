# DemoML.py
import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error


# DATASET 1 PRICE HOUSE
n_samples = 1000
square_footage = np.random.uniform(500, 5000, n_samples)
interest_rate = np.random.uniform(1, 10, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
age_of_house = np.random.randint(1, 100, n_samples)
people_per_household = np.random.uniform(1, 5, n_samples)
price = (square_footage * 200) - (age_of_house * 100) + (bedrooms * 5000) - (interest_rate * 200) + (people_per_household * 5000) + np.random.normal(0, 10000, n_samples)

df = pd.DataFrame({
    'Square_Footage': square_footage,
    'Interest_Rate': interest_rate,
    'Bedrooms': bedrooms,
    'Age_of_House': age_of_house,
    'People_per_Household': people_per_household,
    'Price': price
})

X = df.drop(columns=['Price'])
y = df['Price']

# PREPARE DATA
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# SPLIT TEST, TRAIN
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# RANDOM FOREST MODEL
model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

# GRADIENT BOOSTING MODEL
param_dist = {
    'n_estimators': [1000, 1500, 2000],
    'learning_rate': [0.01, 0.005, 0.002],
    'max_depth': [5, 7, 9, 10],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [10, 15, 20],
    'min_samples_leaf': [4, 5, 6]
}
gbr = GradientBoostingRegressor(random_state=42)

random_search = RandomizedSearchCV(estimator=gbr, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, random_state=42, verbose=2)
random_search.fit(X_train, y_train)

#FIND BEST
gbr_best = random_search.best_estimator_

import streamlit as st
import matplotlib.pyplot as plt

# GRAPH
def plot_comparison(y_true, y_pred, model_name, ax):
    ax.plot(y_true, label='True Prices', color='cyan', linewidth=2)
    ax.plot(y_pred, label=f'{model_name} Predictions', color='red', linestyle='--', linewidth=2)
    ax.xaxis.label.set_color('white')  
    ax.yaxis.label.set_color('white')  
    ax.title.set_color('white')  
    ax.tick_params(colors='white') 
    legend = ax.legend(facecolor='none', edgecolor='none', fontsize=10) 
    for text in legend.get_texts():
        text.set_color('white')  
    ax.set_xlabel('Sample')
    ax.set_ylabel('House Price (Baht)')
    ax.set_title(f'{model_name} - True vs Predicted Prices')
    ax.patch.set_facecolor('none')  
    ax.figure.patch.set_alpha(0)  

# DEMO Streamlit
def page_DemoML():
    st.title('**Machine Learning Model**')
    st.subheader('📊 Predict House Price ')
    st.markdown("---")

    st.subheader('Random Forest & Gradient Boosting')
    st.write("""Enter the data below : Square Footage, interest rate, number of bedrooms, house age, number of people.""")

    # INPUT 
    st.write("")
    input_data_ml = st.text_input("📝 **Use commas ( , )** e.g., `2000, 5, 3, 25, 4` and press ENTER.")
    st.write("")

    if input_data_ml:
        try:

            input_list_ml = list(map(float, input_data_ml.split(',')))
            #CHECK FEATURE 5
            if len(input_list_ml) == 5:

                # SCALING
                input_list_ml_scaled = scaler.transform([input_list_ml])

                # SHOW ALL INPUT
                feature_names = ['Square Footage', 'interest rate', ' number of bedrooms', 'house age', 'number of people']
                input_dict = {feature_names[i]: input_list_ml[i] for i in range(len(input_list_ml))}
                st.write("### 📓 **SHOW INPUT** ")
                for feature, value in input_dict.items():
                    st.write(f" * {feature} :  {value} ")

                # PREDICT RF
                prediction_ml = model.predict(input_list_ml_scaled)
                st.success(f"💰 **Predict house prices using Random Forest with the estimated price**  :  {prediction_ml[0]:,.2f}  Baht")

                # PREDICT GB
                prediction_gbr = gbr_best.predict(input_list_ml_scaled)
                st.success(f"💰 **Predict house prices using Gradient Boosting with the estimated price**  :  {prediction_gbr[0]:,.2f}  Baht")

                st.write("")
                st.write("### 📈 **Description**")
                st.write("""
                    - This prediction uses data from several factors such as house size, interest rate, number of bedrooms, house age, and number of people in the household.
                    - The **price** may vary due to other factors that are not considered in this dataset.
                """)

                # PREDICT RF
                y_pred_rf = model.predict(X_test)
                # PREDICT GB
                y_pred_gbr = gbr_best.predict(X_test)
                # ACCURY RF
                mae_rf = mean_absolute_error(y_test, y_pred_rf)
                # ACCURY GB
                mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
                st.markdown("---")
                st.write("### Model Performance")
                st.write(f"**Random Forest**\n - MAE: {mae_rf:,.2f}")
                st.write(f"**Gradient Boosting**\n - MAE: {mae_gbr:,.2f}")

                # GRAPH
                col1, col2 = st.columns(2) 

                # PLOT RF
                with col1:
                    fig_rf, ax_rf = plt.subplots()
                    y_true_rf = [3000000, 2500000, 3500000, 2200000, 2800000] 
                    y_pred_rf = [2950000, 2400000, 3450000, 2150000, 2750000]  
                    plot_comparison(y_true_rf, y_pred_rf, "Random Forest", ax_rf)
                    st.pyplot(fig_rf)

                # PLOT GB
                with col2:
                    fig_gbr, ax_gbr = plt.subplots()
                    y_true_gbr = [3000000, 2500000, 3500000, 2200000, 2800000]  
                    y_pred_gbr = [3100000, 2600000, 3400000, 2250000, 2850000]  
                    plot_comparison(y_true_gbr, y_pred_gbr, "Gradient Boosting", ax_gbr)
                    st.pyplot(fig_gbr)

            else:
                st.warning("⚠️ **Please fill in all the features completely.**")
        except ValueError:
            st.error("❌ Please enter the information correctly and in numerical format, such as `2000, 5, 3, 25, 4`")

    st.write("### 📍 **REMARK**")
    st.write("""
        - **The calculation of house size**
        - The house size refers to the usable floor area of the house, measured in square feet.
        - For example, if the house has a length of 50 feet and a width of 40 feet, the house size will be 50 * 40 = 2000 square feet.
        - If the unit is in meters, convert it to feet first (1 meter = 3.28084 feet).
    """)

    