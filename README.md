# Architecture du projet


```
- data/
  - raw/
    - iris.csv
- models/
  - model.pkl       # modele entrainé 
- src/
  - tain.py         # Script d'entrainement
- api/              
  - app.py          # API FastAPI
- tests/
  - test_model.py   # tests unitaires
params.yaml
requirements.txt    # Dependances Python
dvc.yaml            # Pipeline DVC (orchestration)
dlc.lock            # Etat de la pipeline (auto-généré)
.dvcignore
.env                # Si secrets ?
.gitignore
- .github/
  - workflows/
    - train.yml     # CI/CD Automatique
README.md
```
