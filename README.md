# Cas Pratiques de Machine Learning en Finance

Ce projet propose deux cas d'usage de machine learning appliqués au domaine financier:
1. **Prédiction de défaut de paiement** (apprentissage supervisé) avec le dataset `credit-g`
2. **Distinction des espèces florales avec le cas d'usage Iris** (apprentissage non supervisé) avec le dataset `Iris`

## Prérequis

```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Cas 1: Prédiction de défaut de paiement (Supervised Learning)

### Objectif
Développer un modèle qui prédit la probabilité qu'un client ne rembourse pas son prêt en fonction de ses caractéristiques financières et personnelles.

### Données
Nous utiliserons le dataset `credit-g` (également connu sous le nom "German Credit Data"), qui contient des informations sur des demandeurs de prêt:

- Attributs personnels (âge, statut matrimonial, etc.)
- Historique de crédit
- Détails du prêt demandé
- Variable cible: risque de crédit (good = bon payeur, bad = mauvais payeur)

### Approche étape par étape

#### 1. Chargement et exploration des données

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Charger les données
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True)
X = credit_data.data
y = credit_data.target

# Convertir la cible en valeurs numériques
y = y.map({'good': 0, 'bad': 1})

# Afficher les premières lignes
print(X.head())

# Statistiques descriptives
print(X.describe())

# Vérifier les valeurs manquantes
print(X.isnull().sum())

# Information sur les colonnes
print(X.info())

# Distribution de la variable cible
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title('Distribution des classes (0=bon payeur, 1=mauvais payeur)')
plt.savefig('target_distribution.png')
plt.show()
```

#### 2. Prétraitement des données

```python
# Identifier les colonnes numériques et catégorielles
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Définir les préprocesseurs pour chaque type de colonne
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

# Créer un préprocesseur pour toutes les colonnes
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

#### 3. Entraînement du modèle

```python
# Créer le pipeline complet avec préprocessement et modèle
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Entraîner le modèle
pipeline.fit(X_train, y_train)

# Prédictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
```

#### 4. Évaluation du modèle

```python
# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bon payeur', 'Mauvais payeur'], 
            yticklabels=['Bon payeur', 'Mauvais payeur'])
plt.xlabel('Prédiction')
plt.ylabel('Réel')
plt.title('Matrice de Confusion')
plt.savefig('confusion_matrix.png')
plt.show()

# Rapport de classification
print(classification_report(y_test, y_pred))

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC - Prédiction de défaut de paiement')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()
```

#### 5. Analyse des features importantes

```python
# Extraire le modèle du pipeline
model = pipeline.named_steps['classifier']

# Récupérer les noms des features après préprocessing
preprocessor = pipeline.named_steps['preprocessor']
cat_features = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)
feature_names = list(numeric_features) + list(cat_features)

# Récupération et visualisation de l'importance des features
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]  # Prendre les 15 plus importantes

plt.figure(figsize=(10, 8))
plt.title('15 caractéristiques les plus importantes')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importance relative')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
```

#### 6. Optimisation du modèle (Bonus)

```python
from sklearn.model_selection import GridSearchCV

# Définir un pipeline plus court pour la recherche sur grille
pipeline_opt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Définir les hyperparamètres à tester
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Recherche par grille
grid_search = GridSearchCV(pipeline_opt, 
                          param_grid, 
                          cv=5, 
                          scoring='roc_auc', 
                          n_jobs=-1)
grid_search.fit(X_train, y_train)

# Meilleurs paramètres
print("Meilleurs paramètres:", grid_search.best_params_)
print("Meilleur score:", grid_search.best_score_)

# Modèle optimisé
best_model = grid_search.best_estimator_
```

## Cas 2: Segmentation des espèces florales avec le cas d'usage Iris. (Unsupervised Learning)

### Objectif  
Identifier automatiquement les différentes espèces de fleurs **Iris** à partir de leurs caractéristiques morphologiques, en utilisant l'algorithme de clustering K-Means.

### Données  
Nous utiliserons le dataset `iris`, un jeu de données classique en apprentissage machine, contenant des mesures de fleurs appartenant à trois espèces différentes :  
- Caractéristiques mesurées :
  - Longueur et largeur des sépales (`sepal length`, `sepal width`)
  - Longueur et largeur des pétales (`petal length`, `petal width`)
- Les étiquettes des espèces sont disponibles (Setosa, Versicolor, Virginica), mais elles seront ignorées pour l'apprentissage non supervisé.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Charger le dataset iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Afficher les premières lignes
print("Premières lignes du dataset Iris:")
print(iris_df.head())

# Statistiques descriptives
print("\nStatistiques descriptives:")
print(iris_df.describe())

# Vérifier les valeurs manquantes
print("\nVérification des valeurs manquantes:")
print(iris_df.isnull().sum())

# Information sur les colonnes
print("\nInformations sur les colonnes:")
print(iris_df.info())

# Standardiser les données
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_df)

# Méthode du coude pour déterminer le nombre optimal de clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(iris_scaled)
    inertia.append(kmeans.inertia_)

# Visualisation de la méthode du coude
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.grid(True)
plt.savefig('elbow_method_iris.png')
plt.show()

# Nombre optimal de clusters (pour Iris, nous savons que c'est 3)
n_clusters = 3

# Création du modèle K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(iris_scaled)

# Ajouter les labels des clusters au dataframe original
iris_df['Cluster'] = cluster_labels

# Réduction de dimension pour visualisation (PCA)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(iris_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels

# Visualisation des clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100)
plt.title('Visualisation des clusters de fleurs Iris')
plt.savefig('iris_clusters.png')
plt.show()

# Analyse des caractéristiques par cluster
cluster_analysis = iris_df.groupby('Cluster').mean()
print("\nCaractéristiques moyennes par cluster:")
print(cluster_analysis)

# Visualisation des caractéristiques moyennes par cluster
plt.figure(figsize=(14, 10))
cluster_analysis.T.plot(kind='bar', figsize=(14, 8))
plt.title('Caractéristiques moyennes par cluster de fleurs Iris')
plt.ylabel('Valeur moyenne')
plt.xlabel('Caractéristiques')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('iris_cluster_characteristics.png')
plt.show()

# Interprétation des clusters (nous connaissons les vraies classes)
true_labels = iris.target
cluster_mapping = {}
for i in range(n_clusters):
    cluster_indices = np.where(cluster_labels == i)[0]
    most_common_label = np.bincount(true_labels[cluster_indices]).argmax()
    cluster_mapping[i] = iris.target_names[most_common_label]

print("\nInterprétation des clusters (correspondance avec les vraies espèces):")
for cluster, species in cluster_mapping.items():
    print(f"Cluster {cluster}: Principalement {species}")

# Visualisation comparant les clusters au vrai étiquetage
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], 
                hue=cluster_labels, palette='viridis', s=100)
plt.title('Clusters KMeans')

plt.subplot(1, 2, 2)
sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], 
                hue=iris.target, palette='viridis', s=100)
plt.title('Classification réelle')
plt.tight_layout()
plt.savefig('iris_cluster_vs_real.png')
plt.show()

# Recommandations par cluster (exemple)
recommendations = {
    0: ["Entretien minimal", "Environnement sec", "Peu d'arrosage"],
    1: ["Entretien modéré", "Environnement mi-ombragé", "Arrosage régulier"],
    2: ["Entretien soigneux", "Environnement humide", "Arrosage fréquent"]
}

print("\nRecommandations d'entretien par cluster:")
for i in range(n_clusters):
    print(f"Cluster {i} ({cluster_mapping[i]}):")
    for rec in recommendations[i]:
        print(f"- {rec}")
    print()
```

## Conclusion

Ces deux cas pratiques démontrent l'application du machine learning à des problématiques financières:

1. **Apprentissage supervisé**: Nous avons développé un modèle de classification pour prédire les défauts de paiement, permettant à l'institution financière d'évaluer le risque associé à chaque client.

2. **Apprentissage non supervisé**: Nous avons segmenté les espèces de fleurs en clusters distincts pour faciliter la distinction et la compréhension des caractéristiques de celles-ci. 

Les étudiants peuvent adapter ces exemples en modifiant les paramètres des modèles ou en explorant d'autres algorithmes (comme les SVM, les réseaux de neurones pour la classification, ou DBSCAN pour le clustering).

## Pour aller plus loin

- Implémenter d'autres algorithmes de classification (logistic regression, SVM, etc.)
- Explorer des techniques de feature engineering spécifiques aux données financières
- Mettre en place un système de recommandation basé sur les clusters identifiés
- Combiner les deux approches pour une stratégie marketing ciblée (par ex: identifier les produits à faible risque pour les clients à haut risque de défaut)
- Utiliser des méthodes d'évaluation plus avancées comme la validation croisée stratifiée
- Explorer d'autres méthodes de clustering comme DBSCAN ou l'analyse de clustering hiérarchique
