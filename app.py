import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Lab 2 - Regularized Regression", layout="wide")
st.title("Regularized Regression: Lasso vs Ridge")
st.caption("MAI511-2 Advanced Machine Learning | Pavnee Modi | 2648537")

@st.cache_data
def get_data():
    X, y = make_regression(n_samples=1000, n_features=12, n_informative=8, noise=20, random_state=42)
    names = [f"Feature_{i}" for i in range(1,13)]
    data = pd.DataFrame(X, columns=names)
    data["Target"] = y
    return data

df = get_data()
X = df.drop("Target", axis=1)
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

alpha_values = np.logspace(-3, 2, 12)

def evaluate(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return {
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R²": r2_score(y_test, pred)
    }

linear = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]).fit(X_train, y_train)
lasso = GridSearchCV(
    Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=20000, random_state=42))]),
    {"model__alpha": alpha_values}, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
).fit(X_train, y_train)
ridge = GridSearchCV(
    Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
    {"model__alpha": alpha_values}, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
).fit(X_train, y_train)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Results", "Coefficients", "Visualizations"])

with tab1:
    st.subheader("Dataset")
    c1,c2,c3 = st.columns(3)
    c1.metric("Instances", len(df))
    c2.metric("Features", X.shape[1])
    c3.metric("Target", "Continuous")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Missing values:", int(df.isnull().sum().sum()))
    st.write("Duplicate records:", int(df.duplicated().sum()))

with tab2:
    rows=[]
    for name, model in [("Linear Regression",linear),("Lasso Regression",lasso.best_estimator_),("Ridge Regression",ridge.best_estimator_)]:
        m=evaluate(model)
        rows.append([name,m["MAE"],m["MSE"],m["RMSE"],m["R²"]])
    out=pd.DataFrame(rows,columns=["Model","MAE","MSE","RMSE","R²"])
    st.dataframe(out.round(4), use_container_width=True)
    c1,c2=st.columns(2)
    c1.metric("Best Lasso α", f"{lasso.best_params_['model__alpha']:.6g}")
    c2.metric("Best Ridge α", f"{ridge.best_params_['model__alpha']:.6g}")

with tab3:
    names=list(X.columns)
    coef=pd.DataFrame({
        "Feature": names,
        "Linear": linear.named_steps["model"].coef_,
        "Lasso": lasso.best_estimator_.named_steps["model"].coef_,
        "Ridge": ridge.best_estimator_.named_steps["model"].coef_
    }).set_index("Feature")
    st.dataframe(coef.round(4), use_container_width=True)
    st.write("Lasso zero-coefficient features:", list(coef.index[np.isclose(coef["Lasso"],0)]))
    st.bar_chart(coef)

with tab4:
    choice=st.selectbox("Actual vs Predicted model",["Lasso Regression","Ridge Regression"])
    model=lasso.best_estimator_ if choice.startswith("Lasso") else ridge.best_estimator_
    pred=model.predict(X_test)
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(y_test,pred,alpha=0.7)
    mn,mx=min(y_test.min(),pred.min()),max(y_test.max(),pred.max())
    ax.plot([mn,mx],[mn,mx],"--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted - {choice}")
    st.pyplot(fig)
    fig2,ax2=plt.subplots(figsize=(8,5))
    lrm=np.sqrt(-lasso.cv_results_["mean_test_score"])
    rrm=np.sqrt(-ridge.cv_results_["mean_test_score"])
    ax2.semilogx(alpha_values,lrm,"o-",label="Lasso")
    ax2.semilogx(alpha_values,rrm,"o-",label="Ridge")
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("5-Fold CV RMSE")
    ax2.set_title("Cross-Validation Performance vs Alpha")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

st.sidebar.header("Lab Requirements")
st.sidebar.write("✓ Linear Regression baseline")
st.sidebar.write("✓ Lasso + GridSearchCV")
st.sidebar.write("✓ Ridge + GridSearchCV")
st.sidebar.write("✓ 5-fold cross-validation")
st.sidebar.write("✓ MAE, MSE, RMSE, R²")
st.sidebar.write("✓ Coefficient comparison")
st.sidebar.write("✓ Visualizations")
st.sidebar.write("✓ Self-learning: expanded alpha search")
