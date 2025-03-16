import streamlit as st
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential


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

np.random.seed(42)
n_samples = 1000
interest_rate_change = np.random.uniform(-1, 2, n_samples)
gdp_growth = np.random.uniform(-5, 5, n_samples)
inflation_rate = np.random.uniform(0, 10, n_samples)
unemployment_rate = np.random.uniform(0, 15, n_samples)
policy_announcement = np.random.choice([0, 1], size=n_samples)
impact = 0.5 * interest_rate_change + 0.3 * gdp_growth - 0.2 * inflation_rate - 0.1 * unemployment_rate + np.random.normal(0, 0.5, n_samples)

df = pd.DataFrame({
    'Interest_Rate_Change': interest_rate_change,
    'GDP_Growth': gdp_growth,
    'Inflation_Rate': inflation_rate,
    'Unemployment_Rate': unemployment_rate,
    'Policy_Announcement': policy_announcement,
    'Impact': impact
})

#PREPARE DATA
missing_percentage = 0.1
n_missing = int(missing_percentage * df.size)
missing_indices = np.random.choice(df.index, size=n_missing, replace=True)

df.iloc[missing_indices, 0] = np.nan
df.iloc[missing_indices, 1] = np.nan
df.iloc[missing_indices, 2] = np.nan

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df.drop(columns=['Impact'])), columns=df.columns[:-1])
df_imputed['Impact'] = df['Impact']

#SPLIT DATA
X = df_imputed.drop(columns=['Impact'])
y = df_imputed['Impact']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1))

#COMPLIE
optimizer = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])

# EARY STOPPING
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)

model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
loss, mae = model.evaluate(X_test, y_test)

def plot_comparison(y_true, y_pred, model_name, ax):
    scaler = MinMaxScaler()
    y_true_scaled = scaler.fit_transform(y_true.to_numpy().reshape(-1, 1))
    y_pred_scaled = scaler.transform(y_pred.reshape(-1, 1))

    ax.plot(y_true_scaled, label='True Prices', color='cyan', linewidth=2)
    ax.plot(y_pred_scaled, label=f'{model_name} Predictions', color='red', linestyle='--', linewidth=2)
    
    ax.xaxis.label.set_color('white')  
    ax.yaxis.label.set_color('white')  
    ax.title.set_color('white')  
    ax.tick_params(colors='white') 
    legend = ax.legend(facecolor='none', edgecolor='none', fontsize=10) 
    for text in legend.get_texts():
        text.set_color('white')  
    ax.set_xlabel('Sample')
    ax.set_ylabel('Price (Scaled)')
    ax.set_title(f'{model_name} - True vs Predicted Prices')
    ax.patch.set_facecolor('none')  
    ax.figure.patch.set_alpha(0) 

# DEMO Streamlit NN
def page_DemoNN():
    st.title('Neural Network Model Demo')
    st.subheader('📊 Monetary Policy Impact ')
    st.markdown("---")

    st.subheader('Neural Network')
    st.write(""" 
        Enter the data below : interest rate change, GDP growth, inflation rate, unemployment rate, monetary policy announcement.
    """)

    st.write("")
    input_data_nn = st.text_input("📝 **Use commas ( , )** e.g., `1.5, 2, 3.5, 4, 1` and press ENTER.")
    st.write("")
    
    if input_data_nn:
        try:
            # FLOAT
            input_list_nn = list(map(float, input_data_nn.split(',')))
            feature_names_nn = ['Interest rate change', 'GDP growth', 'inflation rate', 'unemployment rate', 'monetary policy announcement']

            # CHECK FEATURE
            if len(input_list_nn) == len(feature_names_nn):
                # CHECK 0 1
                last_value = int(input_list_nn[-1])  
                if last_value not in [0, 1]:
                    st.error("The final value must be either 0 or 1 only !")
                else:
                    input_dict_nn = {feature_names_nn[i]: input_list_nn[i] for i in range(len(input_list_nn))}

                    st.write("### 📓 **SHOW INPUT** ")
                    for feature, value in input_dict_nn.items():
                        st.write(f" * {feature} :  {value} ")

                    prediction_nn = model.predict(np.array([input_list_nn]))  
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    st.markdown("---")

                    st.success(f"🔍 **Prediction of Impact :** {prediction_nn[0][0]:.2f}")
                    st.success(f"📉 **MAE :** {mae:.2f}")
                    st.write("")
                    st.write("### 📈 **Description Predicted**")
                    st.write("""
                        - **The prediction results have two types: positive impact and negative impact.** 
                        - Positive impact: The economy is growing, businesses are expanding, leading to more investments and job creation. This results in higher income and increased spending among the public.
                        - Negative impact: The economy may slow down, with high interest rates and inflation. This leads to higher prices of goods, causing people to spend less. 
                    """)

                    st.write("")
                    st.write("")

                    #GRAPH
                    with st.container():
                        fig_nn, ax_nn = plt.subplots(figsize=(8, 6))  
                        y_true_nn = y_test  
                        y_pred_nn = model.predict(X_test)  
                        plot_comparison(y_true_nn, y_pred_nn, "Neural Network", ax_nn)  
                        st.pyplot(fig_nn)  

            else:
                st.warning("⚠️ **Please fill in all the features completely.**")

        except ValueError:
            st.error("❌  Please enter the information correctly and in numerical format, such as `1.5, 2, 3.5, 4, 1`")

    st.write("### 📍 **REMARK**")
    st.write("""
        - **Substituting the monetary policy announcement**
        - 0 : No monetary policy announcement (No changes or new policies have been introduced)
        - 1 : Monetary policy announcement (Changes or new policies have been introduced)
    """)
    



