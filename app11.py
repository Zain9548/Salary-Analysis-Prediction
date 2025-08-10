import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.stats import chi2_contingency
import math

# Page configuration
st.set_page_config(page_title="Employee Salary Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; padding: 10px 24px; }
    .stSelectbox, .stSlider, .stTextInput { margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# Global variables
cat_cols = ['jobType', 'degree', 'major', 'industry']
num_cols = ['yearsExperience', 'milesFromMetropolis']

# Load data function with caching
@st.cache_data
def load_data():
    try:
        train_data = pd.read_csv("train_dataset.csv")
        train_salaries = pd.read_csv("train_salaries.csv")
        test_data = pd.read_csv("test_dataset.csv")
        
        # Validate merge keys
        if not all(train_data['jobId'].isin(train_salaries['jobId'])):
            st.warning("Some jobIds in train_data don't have salary information")
        
        # Merge datasets
        merged = pd.merge(train_data, train_salaries, on='jobId', how='inner')
        
        # Filter salaries >= 30
        filtered = merged[merged["salary"] >= 30].reset_index(drop=True)
        
        st.success(f"Data loaded successfully. Train samples: {len(filtered)}")
        return filtered, test_data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Load datasets
train_data, test_data = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                           ["Data Overview", "Exploratory Analysis", 
                            "Statistical Tests", "Model Training", 
                            "Salary Prediction"])

# Page logic
if options == "Data Overview":
    st.header("Dataset Overview")
    
    st.subheader("Train Data Preview")
    st.write(train_data.head())
    
    st.subheader("Test Data Preview")
    st.write(test_data.head())
    
    st.subheader("Data Description")
    st.write("""
    - **jobId**: Unique ID for employee
    - **companyId**: Unique ID for company
    - **jobType**: Employee's job position
    - **degree**: Education level completed
    - **major**: Field of specialization
    - **industry**: Industry of employment
    - **yearsExperience**: Years of work experience
    - **milesFromMetropolis**: Distance from company to home in miles
    - **salary**: Salary in thousands of dollars (e.g., 250 = $250,000)
    """)
    
    st.subheader("Data Statistics")
    st.write(train_data.describe())
    
    st.subheader("Missing Values Check")
    st.write("Train Data Missing Values:")
    st.write(train_data.isna().sum())
    st.write("Test Data Missing Values:")
    st.write(test_data.isna().sum())

elif options == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")

    st.subheader("Numerical Features Distribution")
    num_cols_with_salary = ['yearsExperience', 'milesFromMetropolis', 'salary']
    
    for col in num_cols_with_salary:
        fig, ax = plt.subplots()
        sns.histplot(train_data[col], kde=True, ax=ax)
        st.pyplot(fig)
    
    st.subheader("Categorical Features Distribution")
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data=train_data, x=col, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.subheader("Salary by Job Type")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=train_data, x='jobType', y='salary')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Correlation Matrix")
    corr = train_data[['yearsExperience', 'milesFromMetropolis', 'salary']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif options == "Statistical Tests":
    st.header("Statistical Tests")
    
    st.subheader("Chi-Square Test for Categorical Variables")
    
    def perform_chi_square_test(var1, var2):
        contingency_table = pd.crosstab(train_data[var1], train_data[var2])
        chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
        
        st.write(f"**{var1} vs {var2}**")
        st.write(f"Chi2 Statistic: {chi2_stat:.2f}")
        st.write(f"P-value: {p_val:.4f}")
        st.write(f"Degrees of Freedom: {dof}")
        
        if p_val < 0.05:
            st.write("Result: Significant association (reject null hypothesis)")
        else:
            st.write("Result: No significant association (fail to reject null hypothesis)")
        st.write("---")
    
    for i in range(len(cat_cols)):
        for j in range(i+1, len(cat_cols)):
            perform_chi_square_test(cat_cols[i], cat_cols[j])
    
    st.subheader("ANOVA Test for Numerical vs Categorical")
    
    def perform_anova_test(cat_var, num_var):
        groups = train_data.groupby(cat_var)[num_var].apply(list)
        f_stat, p_val = stats.f_oneway(*groups)
        
        st.write(f"**{cat_var} vs {num_var}**")
        st.write(f"F-statistic: {f_stat:.2f}")
        st.write(f"P-value: {p_val:.4f}")
        
        if p_val < 0.05:
            st.write("Result: Significant difference in means (reject null hypothesis)")
        else:
            st.write("Result: No significant difference in means (fail to reject null hypothesis)")
        st.write("---")
    
    for cat_var in cat_cols:
        perform_anova_test(cat_var, 'salary')

elif options == "Model Training":
    st.header("Model Training")
    try:
        # Prepare data
        train_processed = train_data.drop(['jobId', 'companyId'], axis=1)
        X = train_processed.drop('salary', axis=1)
        y = train_processed['salary']
        
        if len(X) != len(y):
            raise ValueError(f"Feature/target size mismatch. X: {len(X)}, y: {len(y)}")
        
        # Encoding categorical variables
        encoder = OneHotEncoder(drop='first', sparse=False)
        encoded = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        
        # Combine encoded and numerical features
        X_processed = pd.concat([encoded_df, X[num_cols]], axis=1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        st.success(f"Data split successful. Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Scaling numerical features
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        
        # Model selection
        model_type = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting"])
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                if model_type == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                
                model.fit(X_train, y_train)
                
                # Save model and preprocessors in session
                st.session_state['model'] = model
                st.session_state['encoder'] = encoder
                st.session_state['scaler'] = scaler
                
                # Predict and evaluate
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train MAE", f"{mean_absolute_error(y_train, train_pred):.2f}")
                    st.metric("Val MAE", f"{mean_absolute_error(y_val, val_pred):.2f}")
                with col2:
                    st.metric("Train RMSE", f"{np.sqrt(mean_squared_error(y_train, train_pred)):.2f}")
                    st.metric("Val RMSE", f"{np.sqrt(mean_squared_error(y_val, val_pred)):.2f}")
                with col3:
                    st.metric("Train R²", f"{r2_score(y_train, train_pred):.4f}")
                    st.metric("Val R²", f"{r2_score(y_val, val_pred):.4f}")
                
                # Feature importance plot
                st.subheader("Feature Importance")
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.barplot(data=importance.head(15), x='Importance', y='Feature', ax=ax)
                    plt.title('Top 15 Important Features')
                    st.pyplot(fig)
                
                st.success("Model trained successfully!")
    
    except Exception as e:
        st.error(f"Error during training: {str(e)}")

elif options == "Salary Prediction":
    st.header("Salary Prediction")
    if 'model' not in st.session_state:
        st.warning("Please train a model first on the Model Training page.")
    else:
        model = st.session_state['model']
        encoder = st.session_state['encoder']
        scaler = st.session_state['scaler']

        st.subheader("Predict for Test Data")
        if st.button("Predict Salaries for Test Dataset"):
            with st.spinner("Making predictions..."):
                test_processed = test_data.drop(['jobId', 'companyId'], axis=1, errors='ignore')
                
                # Encode categorical features
                encoded_test = encoder.transform(test_processed[cat_cols])
                encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cat_cols), index=test_processed.index)
                
                # Combine and scale numerical features
                X_test = pd.concat([encoded_test_df, test_processed[num_cols]], axis=1)
                X_test[num_cols] = scaler.transform(X_test[num_cols])
                
                # Predict salaries
                predictions = model.predict(X_test)
                
                results = test_data[['jobId']].copy()
                results['predicted_salary'] = predictions
                
                st.subheader("Sample Predictions")
                st.write(results.head())
                
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name='salary_predictions.csv',
                    mime='text/csv'
                )
        
        st.subheader("Predict for Custom Input")
        with st.form("custom_prediction"):
            col1, col2 = st.columns(2)
            
            with col1:
                job_type = st.selectbox("Job Type", train_data['jobType'].unique())
                degree = st.selectbox("Degree", train_data['degree'].unique())
                major = st.selectbox("Major", train_data['major'].unique())
            
            with col2:
                industry = st.selectbox("Industry", train_data['industry'].unique())
                years_exp = st.slider("Years of Experience", 0, 30, 5)
                miles_from_metro = st.slider("Miles from Metropolis", 0, 100, 20)
            
            if st.form_submit_button("Predict Salary"):
                input_data = pd.DataFrame({
                    'jobType': [job_type],
                    'degree': [degree],
                    'major': [major],
                    'industry': [industry],
                    'yearsExperience': [years_exp],
                    'milesFromMetropolis': [miles_from_metro]
                })
                
                encoded_input = encoder.transform(input_data[cat_cols])
                encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(cat_cols))
                
                X_input = pd.concat([encoded_input_df, input_data[num_cols]], axis=1)
                X_input[num_cols] = scaler.transform(X_input[num_cols])
                
                predicted_salary = model.predict(X_input)[0]
                
                st.success(f"Predicted Salary: ${predicted_salary * 1000:,.2f} per year")

else:
    st.warning("Please select a valid option from the sidebar")
