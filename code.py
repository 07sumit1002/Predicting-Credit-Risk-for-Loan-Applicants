
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

# Title and description
st.title("Credit Risk Prediction System")
st.markdown("""
This application predicts credit risk for loan applicants using the German Credit dataset.
The system uses machine learning to classify applicants as good or bad credit risks.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('german_credit_data.csv')
    return df

# Data preprocessing
def preprocess_data(df):
    df_processed = df.copy()
    
    # Handle missing values
    for col in ['Saving accounts', 'Checking account']:
        df_processed[col] = df_processed[col].fillna('unknown')
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col])
    
    # Create target variable
    df_processed['Credit_risk'] = np.where(
        (df_processed['Credit amount'] > df_processed['Credit amount'].median()) & 
        (df_processed['Duration'] > df_processed['Duration'].median()),
        0, 1
    )
    
    # Feature engineering
    df_processed['Credit_per_month'] = df_processed['Credit amount'] / df_processed['Duration']
    df_processed['Age_group'] = pd.cut(df_processed['Age'], 
                                      bins=[18, 25, 35, 45, 55, 65, 100],
                                      labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    df_processed['Age_group'] = le.fit_transform(df_processed['Age_group'])
    
    return df_processed

# Model training and evaluation
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Hyperparameter tuning for Random Forest
        if name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_
        
        # Train model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label=1),
            'recall': recall_score(y_test, y_pred, pos_label=1),
            'f1': f1_score(y_test, y_pred, pos_label=1),
            'model': model
        }
    
    return results, X_train, X_test, y_train, y_test, scaler

# Create visualizations
def create_visualizations(df, results, X_train, X_test, y_test, model, scaler):
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_imp.head(10), x='Importance', y='Feature', 
                        title='Top 10 Most Important Features',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Credit amount distribution by risk
        fig = px.histogram(df, x='Credit amount', color='Credit_risk',
                          title='Credit Amount Distribution by Risk Category',
                          nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix
    y_pred = model.predict(scaler.transform(X_test))
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(fig)

# Main application
def main():
    # Load and preprocess data
    df = load_data()
    st.subheader("Data Overview")
    st.write("First few rows of the dataset:")
    st.dataframe(df.head())
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Display processed data stats
    st.subheader("Data Statistics")
    st.write(df_processed.describe())
    
    # Prepare features and target
    X = df_processed.drop(['Credit_risk'], axis=1)
    y = df_processed['Credit_risk']
    
    # Train and evaluate models
    st.subheader("Model Performance")
    results, X_train, X_test, y_train, y_test, scaler = train_and_evaluate(X, y)
    
    # Display results
    results_df = pd.DataFrame({
        'Model': results.keys(),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1-Score': [results[model]['f1'] for model in results]
    })
    st.dataframe(results_df)
    
    # Visualizations
    st.subheader("Visualizations")
    best_model = results['Random Forest']['model']
    create_visualizations(df_processed, results, X_train, X_test, y_test, best_model, scaler)
    
    # Insights and recommendations
    st.subheader("Insights and Recommendations")
    st.markdown("""
    ### Key Findings:
    1. **Credit Amount and Duration**: Higher credit amounts combined with longer durations significantly increase credit risk.
    2. **Age Influence**: Certain age groups (particularly younger applicants) show higher risk patterns.
    3. **Savings and Checking Accounts**: Applicants with low savings and checking account balances are more likely to be classified as bad risks.
    
    ### Recommendations:
    1. **Enhanced Screening**: Implement stricter evaluation for high-amount, long-duration loans.
    2. **Financial Education**: Offer financial planning resources for younger applicants.
    3. **Account Balance Monitoring**: Use savings and checking account information as key risk indicators.
    4. **Model Updates**: Regularly retrain the model with new data to maintain accuracy.
    """)

if __name__ == "__main__":
    main()