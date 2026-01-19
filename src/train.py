import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import yaml
import pickle

# Charger les paramètres
with open('params.yaml') as f:
	params = yaml.safe_load(f)
	
# Charger les données
data = pd.read_csv('data/raw/iris.csv')
X = data.drop('variety', axis=1)
y = data['variety']

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# Démarre une run MlFlow
with mlflow.start_run():
	# Logger les paramètres
	mlflow.log_params(params["model"])

	# Entrainer le modele
	model = RandomForestClassifier(**params['model'])
	model.fit(X_train, y_train)

	# Evaluation
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred, average="weighted")

	# Logger les métriques
	mlflow.log_metric("accuracy", accuracy)
	mlflow.log_metric("f1_score", f1)

	# Logger le modèle
	mlflow.sklearn.log_model(model, "model")

	# sauvegarde localement
	with open("models/model.pkl", "wb") as f:
		pickle.dump(model, f)

	print(f"modele entrainé - Accuracy: {accuracy:.3f}")
	
