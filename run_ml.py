import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                              classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)
import joblib

os.makedirs('data/models', exist_ok=True)
os.makedirs('data/charts', exist_ok=True)

print('Loading processed data...')
df = pd.read_csv('data/flights_processed.csv', low_memory=False)
print(f'Loaded {len(df):,} rows')

FEATURES = ['DEPARTURE_DELAY','DEPARTURE_HOUR','DISTANCE',
            'DAY_OF_WEEK','MONTH','AIRLINE','ORIGIN_AIRPORT',
            'TIME_OF_DAY','SEASON','IS_FOG_RISK','IS_WEEKEND']
TARGET = 'IS_DELAYED'

df_ml = df[FEATURES + [TARGET]].dropna().copy()
le = {}
for col in ['AIRLINE','ORIGIN_AIRPORT','TIME_OF_DAY','SEASON']:
    le[col] = LabelEncoder()
    df_ml[col] = le[col].fit_transform(df_ml[col].astype(str))

X = df_ml[FEATURES]; y = df_ml[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {len(X_train):,}   Test: {len(X_test):,}')

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print('\nTraining Logistic Regression...')
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s); lr_prob = lr.predict_proba(X_test_s)[:,1]
lr_acc = accuracy_score(y_test, lr_pred); lr_auc = roc_auc_score(y_test, lr_prob)
print(f'Logistic Regression  ->  Accuracy: {lr_acc*100:.2f}%   AUC: {lr_auc:.4f}')

print('\nTraining Random Forest (takes 1-2 mins)...')
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                             min_samples_leaf=50, class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test); rf_prob = rf.predict_proba(X_test)[:,1]
rf_acc = accuracy_score(y_test, rf_pred); rf_auc = roc_auc_score(y_test, rf_prob)
print(f'Random Forest        ->  Accuracy: {rf_acc*100:.2f}%   AUC: {rf_auc:.4f}')

joblib.dump(rf,     'data/models/random_forest.joblib')
joblib.dump(lr,     'data/models/logistic_regression.joblib')
joblib.dump(scaler, 'data/models/scaler.joblib')
joblib.dump(le,     'data/models/label_encoders.joblib')
print('\nModels saved to data/models/')

BG = '#FFF8F0'; ACCENT = '#FF6B35'; BLUE = '#1B4F72'
fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.patch.set_facecolor(BG)

for (name, prob, auc), color in zip([
    ('Logistic Regression', lr_prob, lr_auc),
    ('Random Forest', rf_prob, rf_auc)
], [BLUE, ACCENT]):
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[0].plot(fpr, tpr, linewidth=2.5, color=color, label=f'{name} (AUC={auc:.3f})')
axes[0].plot([0,1],[0,1],'k--', linewidth=1, label='Random Baseline')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves — Indian Flight Delay Prediction')
axes[0].legend(loc='lower right'); axes[0].set_facecolor(BG)

fi = pd.DataFrame({'Feature': FEATURES, 'Importance': rf.feature_importances_})
fi = fi.sort_values('Importance', ascending=False)
clrs = [ACCENT if i < 3 else BLUE for i in range(len(fi))]
axes[1].barh(fi['Feature'][::-1], fi['Importance'][::-1], color=clrs[::-1], edgecolor='white')
axes[1].set_xlabel('Feature Importance'); axes[1].set_facecolor(BG)
axes[1].set_title('Feature Importances — Random Forest')

plt.tight_layout()
plt.savefig('data/charts/ml_evaluation.png', dpi=150, bbox_inches='tight'); plt.close()
print('Charts saved to data/charts/ml_evaluation.png')

print('\n========== FINAL RESULTS ==========')
print(f'Logistic Regression  ->  Accuracy: {lr_acc*100:.2f}%   AUC: {lr_auc:.4f}')
print(f'Random Forest        ->  Accuracy: {rf_acc*100:.2f}%   AUC: {rf_auc:.4f}')
print('====================================')
print('\nClassification Report — Random Forest:')
print(classification_report(y_test, rf_pred, target_names=['On Time','Delayed'], digits=3))
print('Done!')
