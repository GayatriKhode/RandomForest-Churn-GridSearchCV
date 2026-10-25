import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/telecom_churn.csv')
args = parser.parse_args()
path = Path(args.data)
if not path.exists():
    raise SystemExit(f'Dataset not found: {path}. Put your churn CSV there or use --data PATH')
df = pd.read_csv(path)
target = next((c for c in df.columns if c.lower() in ('churn','exited')), None)
if target is None:
    raise SystemExit('Expected a target column named Churn or Exited.')
y = df[target]
if y.dtype == 'object' or str(y.dtype) == 'category':
    values = y.astype(str).str.strip().str.lower()
    if set(values.unique()) <= {'yes','no'}:
        y = values.map({'yes':1,'no':0})
    elif set(values.unique()) <= {'true','false'}:
        y = values.map({'true':1,'false':0})
    else:
        raise SystemExit('Churn target must be Yes/No, True/False, or 0/1.')
if y.isna().any() or y.nunique() != 2:
    raise SystemExit('Churn target must contain two classes and no missing values.')
X = df.drop(columns=[target])
X = X.drop(columns=[c for c in X.columns if c.lower() in ('customerid','customer_id','rownumber','surname')], errors='ignore')
if 'TotalCharges' in X.columns:
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
numeric = X.select_dtypes(include='number').columns.tolist()
categorical = X.select_dtypes(exclude='number').columns.tolist()
preprocess = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), numeric),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical)
])
model = Pipeline([('preprocess', preprocess), ('forest', RandomForestClassifier(random_state=42, n_jobs=-1))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
param_grid = {'forest__n_estimators':[50,100], 'forest__max_depth':[5,10,None], 'forest__max_features':['sqrt','log2']}
search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_
best.fit(X_train, y_train)
pred = best.predict(X_test)
out = Path('outputs'); out.mkdir(exist_ok=True)
report = f'Best parameters: {search.best_params_}\nBest 3-fold CV accuracy: {search.best_score_:.4f}\nTest accuracy: {accuracy_score(y_test,pred):.4f}\n\n{classification_report(y_test,pred,zero_division=0)}'
(out/'results.txt').write_text(report, encoding='utf-8')
pd.DataFrame(search.cv_results_).to_csv(out/'grid_search_results.csv', index=False)
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title('Random Forest Churn Confusion Matrix')
plt.tight_layout(); plt.savefig(out/'confusion_matrix.png', dpi=150)
print(report)
print('Saved results in outputs/')
