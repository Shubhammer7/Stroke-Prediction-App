import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stroke Risk Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }

    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2E86AB;
        padding-bottom: 0.5rem;
    }

    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
        border-radius: 5px;
        color: #333333;
    }
    
    .insight-box h4 {
        color: #2E86AB;
        margin-bottom: 0.5rem;
    }
    
    .insight-box ul {
        color: #333333;
    }
    
    .insight-box li {
        color: #555555;
        margin-bottom: 0.3rem;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    return df

def create_age_distribution_plot(df):
    fig = px.histogram(df, x='age', nbins=40,
                       title='Age Distribution in Dataset',
                       labels={'age': 'Age (years)', 'count': 'Count'},
                       color_discrete_sequence=['#FF6B6B'])

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_x=0.5
    )
    return fig

def create_stroke_analysis_plots(df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Stroke Distribution', 'Age vs Stroke',
                        'Gender vs Stroke', 'Hypertension vs Stroke'),
        specs=[[{"type": "pie"}, {"type": "box"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    stroke_counts = df['stroke'].value_counts()
    fig.add_trace(
        go.Pie(labels=['No Stroke', 'Stroke'],
               values=[stroke_counts[0], stroke_counts[1]],
               marker_colors=['#4ECDC4', '#FF6B6B']),
        row=1, col=1
    )

    stroke_0 = df[df['stroke'] == 0]['age']
    stroke_1 = df[df['stroke'] == 1]['age']

    fig.add_trace(go.Box(y=stroke_0, name='No Stroke', marker_color='#4ECDC4'), row=1, col=2)
    fig.add_trace(go.Box(y=stroke_1, name='Stroke', marker_color='#FF6B6B'), row=1, col=2)

    gender_stroke = df.groupby(['gender', 'stroke']).size().unstack(fill_value=0)
    gender_stroke_pct = gender_stroke.div(gender_stroke.sum(axis=1), axis=0) * 100

    fig.add_trace(
        go.Bar(x=['Female', 'Male'], y=gender_stroke_pct[1],
               name='Stroke Rate (%)', marker_color='#FF6B6B'),
        row=2, col=1
    )

    hyper_stroke = df.groupby(['hypertension', 'stroke']).size().unstack(fill_value=0)
    hyper_stroke_pct = hyper_stroke.div(hyper_stroke.sum(axis=1), axis=0) * 100

    fig.add_trace(
        go.Bar(x=['No Hypertension', 'Hypertension'], y=hyper_stroke_pct[1],
               name='Stroke Rate (%)', marker_color='#FF6B6B'),
        row=2, col=2
    )

    fig.update_layout(height=800, showlegend=False, title_text="Stroke Risk Analysis Dashboard")

    return fig


def create_correlation_heatmap(df):
    numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'stroke', 'hypertension', 'heart_disease']
    corr_matrix = df[numerical_cols].corr()

    fig = px.imshow(corr_matrix,
                    title='Correlation Matrix - Risk Factors',
                    color_continuous_scale='RdBu_r',
                    aspect='auto')

    fig.update_layout(
        title_x=0.5,
        height=500
    )

    return fig


def prepare_data_for_modeling(df):
    # Code from jupyter notebook
    df_model = df.copy()

    X = df_model.drop('stroke', axis=1)
    y = df_model['stroke']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    num_cols = ['age', 'avg_glucose_level', 'bmi']
    cat_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                'smoking_status']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return X_train, X_test, y_train, y_test, preprocessor

def train_models(X_train, y_train, preprocessor):
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(scale_pos_weight=10, random_state=42)
    }

    trained_models = {}
    model_scores = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')

        trained_models[name] = pipeline
        model_scores[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

    return trained_models, model_scores


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_pred_proba)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    return {
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'auc_score': auc_score,
        'fpr': fpr,
        'tpr': tpr,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def create_roc_curve(models_results):
    fig = go.Figure()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, (name, results) in enumerate(models_results.items()):
        fig.add_trace(go.Scatter(
            x=results['fpr'],
            y=results['tpr'],
            mode='lines',
            name=f'{name} (AUC: {results["auc_score"]:.3f})',
            line=dict(color=colors[i], width=3)
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title_x=0.5,
        height=500
    )

    return fig


def main():
    st.markdown('<h1 class="main-header">Stroke Risk Prediction Dashboard</h1>', unsafe_allow_html=True)

    st.sidebar.markdown("## Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Overview", "Exploratory Analysis", "Model Training", "Predictions"]
    )

    df = load_data()

    if page == "Data Overview":
        st.markdown('<div class="sub-header">Dataset Overview</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{len(df):,}</h3>
                <p>Total Patients</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            stroke_rate = (df['stroke'].sum() / len(df)) * 100
            st.markdown(f"""
            <div class="metric-container">
                <h3>{stroke_rate:.1f}%</h3>
                <p>Stroke Rate</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_age = df['age'].mean()
            st.markdown(f"""
            <div class="metric-container">
                <h3>{avg_age:.1f}</h3>
                <p>Average Age</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            missing_bmi = df['bmi'].isnull().sum()
            st.markdown(f"""
            <div class="metric-container">
                <h3>{missing_bmi}</h3>
                <p>Missing BMI</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sub-header">Data Sample</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Dataset Information")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write("**Target Variable:** Stroke (Binary: 0=No, 1=Yes)")
            st.write("**Missing Values:**")
            st.write(df.isnull().sum()[df.isnull().sum() > 0])

        with col2:
            st.markdown("### Statistical Summary")
            st.write(df.describe())

        st.markdown("""
        <div class="insight-box">
            <h4>🔍 Key Insights</h4>
            <ul>
                <li>Dataset contains 5,110 patient records with 11 features</li>
                <li>Stroke is rare (~4.9% of cases) - class imbalanced problem</li>
                <li>BMI has 201 missing values that need imputation</li>
                <li>Age ranges from 0.08 to 82 years with mean of 43.2 years</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Exploratory Analysis":
        st.markdown('<div class="sub-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

        st.markdown("### Age Distribution")
        fig_age = create_age_distribution_plot(df)
        st.plotly_chart(fig_age, use_container_width=True)

        st.markdown("### Stroke Risk Factors Analysis")
        fig_stroke = create_stroke_analysis_plots(df)
        st.plotly_chart(fig_stroke, use_container_width=True)

        st.markdown("### Feature Correlations")
        fig_corr = create_correlation_heatmap(df)
        st.plotly_chart(fig_corr, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Glucose Level Analysis")
            q1 = df['avg_glucose_level'].quantile(0.25)
            q3 = df['avg_glucose_level'].quantile(0.75)
            iqr = q3 - q1
            outliers_pct = (df['avg_glucose_level'] > (q3 + 1.5 * iqr)).sum() / len(df) * 100
            st.write(f"**Outliers (>170 mg/dL):** {outliers_pct:.1f}% of patients")

            fig_glucose = px.box(df, x='stroke', y='avg_glucose_level',
                                 title='Glucose Levels by Stroke Status',
                                 labels={'stroke': 'Stroke Status', 'avg_glucose_level': 'Average Glucose Level'})
            st.plotly_chart(fig_glucose, use_container_width=True)

        with col2:
            st.markdown("### 📊 BMI Analysis")
            bmi_outliers_pct = (df['bmi'] > 47.5).sum() / len(df) * 100
            st.write(f"**BMI Outliers (>47.5):** {bmi_outliers_pct:.1f}% of patients")

            fig_bmi = px.box(df, x='stroke', y='bmi',
                             title='BMI Distribution by Stroke Status',
                             labels={'stroke': 'Stroke Status', 'bmi': 'Body Mass Index'})
            st.plotly_chart(fig_bmi, use_container_width=True)

    elif page == "Model Training":
        st.markdown('<div class="sub-header">Machine Learning Models</div>', unsafe_allow_html=True)

        with st.spinner('Training models... This may take a moment.'):
            X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_modeling(df)

            trained_models, model_scores = train_models(X_train, y_train, preprocessor)

            models_results = {}
            for name, model in trained_models.items():
                models_results[name] = evaluate_model(model, X_test, y_test)

        st.markdown("### 🎯 Model Performance Comparison")

        performance_data = []
        for name in trained_models.keys():
            performance_data.append({
                'Model': name,
                'CV AUC (Mean)': f"{model_scores[name]['cv_mean']:.3f}",
                'CV AUC (Std)': f"{model_scores[name]['cv_std']:.3f}",
                'Test AUC': f"{models_results[name]['auc_score']:.3f}",
                'Precision': f"{models_results[name]['classification_report']['1']['precision']:.3f}",
                'Recall': f"{models_results[name]['classification_report']['1']['recall']:.3f}",
                'F1-Score': f"{models_results[name]['classification_report']['1']['f1-score']:.3f}"
            })

        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)

        st.markdown("### 📈 ROC Curve Comparison")
        fig_roc = create_roc_curve(models_results)
        st.plotly_chart(fig_roc, use_container_width=True)

        best_model_name = max(models_results.keys(), key=lambda k: models_results[k]['auc_score'])
        best_auc = models_results[best_model_name]['auc_score']

        st.markdown(f"""
        <div class="insight-box">
            <h4>🏆 Best Performing Model</h4>
            <p><strong>{best_model_name}</strong> achieved the highest AUC score of <strong>{best_auc:.3f}</strong></p>
            <p>This model shows good ability to distinguish between stroke and non-stroke cases.</p>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.trained_models = trained_models
        st.session_state.preprocessor = preprocessor

    elif page == "Predictions":
        st.markdown('<div class="sub-header">Make Stroke Risk Predictions</div>', unsafe_allow_html=True)

        if 'trained_models' not in st.session_state:
            st.warning("⚠️ Please train the models first by visiting the 'Model Training' section.")
            return

        st.markdown("### 🔮 Individual Risk Assessment")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 0, 100, 45)
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
            heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
            ever_married = st.selectbox("Ever Married", ["Yes", "No"])

        with col2:
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
            avg_glucose_level = st.slider("Average Glucose Level", 50.0, 300.0, 100.0)
            bmi = st.slider("BMI", 10.0, 50.0, 25.0)
            smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

        if st.button("🔍 Predict Stroke Risk", type="primary"):
            input_data = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'Residence_type': [residence_type],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [smoking_status]
            })

            col1, col2, col3 = st.columns(3)

            for i, (name, model) in enumerate(st.session_state.trained_models.items()):
                prob = model.predict_proba(input_data)[0, 1]
                risk_level = "High" if prob > 0.5 else "Medium" if prob > 0.2 else "Low"
                color = "#FF6B6B" if prob > 0.5 else "#FFD93D" if prob > 0.2 else "#4ECDC4"

                with [col1, col2, col3][i]:
                    st.markdown(f"""
                    <div style="background: {color}; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                        <h4>{name}</h4>
                        <h2>{prob:.1%}</h2>
                        <p>Risk Level: {risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("### 🎯 Risk Factor Analysis")

            risk_factors = []
            if age > 65:
                risk_factors.append("Age > 65 years")
            if hypertension:
                risk_factors.append("Hypertension present")
            if heart_disease:
                risk_factors.append("Heart disease present")
            if avg_glucose_level > 150:
                risk_factors.append("High glucose level (>150 mg/dL)")
            if bmi > 35:
                risk_factors.append("Obesity (BMI > 35)")
            if smoking_status == "smokes":
                risk_factors.append("Current smoker")

            if risk_factors:
                st.markdown("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
            else:
                st.markdown("✅ **Low risk profile** - No major risk factors identified")

            st.markdown("""
            <div class="insight-box">
                <h4>Important Disclaimer</h4>
                <p>This prediction tool is for educational purposes only.</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()