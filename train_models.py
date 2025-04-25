
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from joblib import dump


os.makedirs('models', exist_ok=True)
# Paths
data_path = 'data/data_clean.csv'
model_dir = 'models'

# Encoder and model paths
enc_gender_path = os.path.join(model_dir, 'gender_encoder.joblib')
enc_category_path = os.path.join(model_dir, 'category_encoder.joblib')
scaler_path = os.path.join(model_dir, 'scaler.joblib')
kmeans_path = os.path.join(model_dir, 'kmeans.joblib')
rf_path = os.path.join(model_dir, 'rf_classifier.joblib')

# 1. Load and encode data
def load_and_prepare_data(path=data_path):
    df = pd.read_csv(path)
    # Label encoding
    gender_enc = LabelEncoder()
    category_enc = LabelEncoder()

    df['gender_encoded'] = gender_enc.fit_transform(df['gender'])
    df['category_encoded'] = category_enc.fit_transform(df['preferred_category'])
    
    # Save encoders
    dump(gender_enc, enc_gender_path)
    dump(category_enc, enc_category_path)
    
    return df

# 2. Train and save models
def train_and_save_models(df, n_clusters=4, rf_trees=100):  # Changed default n_clusters to 4
    os.makedirs(model_dir, exist_ok=True)
    features = ['age','spending_score','membership_years','purchase_frequency','last_purchase_amount','gender_encoded','category_encoded']
    X = df[features]
    
    # Scale
    dscaler = StandardScaler().fit(X)
    dump(dscaler, scaler_path)
    X_scaled = dscaler.transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    dump(kmeans, kmeans_path)
    df['Cluster'] = kmeans.predict(X_scaled)

    # RandomForest
    rf = RandomForestClassifier(n_estimators=rf_trees, random_state=42).fit(X, df['Cluster'])
    dump(rf, rf_path)

    print(f"Models saved under '{model_dir}/'")

if __name__ == '__main__':
    data = load_and_prepare_data()
    train_and_save_models(data)  # Will now use 4 clusters by default


