# Cas Pratiques de Machine Learning en Finance

Ce projet propose deux cas d'usage de machine learning appliqués au domaine financier:
1. **Prédiction de défaut de paiement** (apprentissage supervisé) avec le dataset `credit-g`
2. **Segmentation de clients pour recommandation de produits** (apprentissage non supervisé) avec le dataset `bank-marketing`

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

## Cas 2: Segmentation de clients pour recommandation de produits (Unsupervised Learning)

### Objectif
Segmenter les clients bancaires en groupes distincts pour faciliter la recommandation de produits financiers adaptés à chaque segment.

### Données
Nous utiliserons le dataset `bank-marketing` qui contient des informations démographiques et comportementales sur des clients bancaires:

- Information démographiques (âge, emploi, statut matrimonial, etc.)
- Informations de contact et campagnes marketing
- Données économiques du client

### Approche étape par étape

#### 1. Chargement et exploration des données

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Charger les données
bank_data = fetch_openml(name='bank-marketing', version=1, as_frame=True)
bank_df = bank_data.data

# Supprimer la colonne cible si présente (nous faisons du non supervisé)
if 'y' in bank_df.columns:
    bank_df = bank_df.drop('y', axis=1)

# Afficher les premières lignes
print(bank_df.head())

# Statistiques descriptives
print(bank_df.describe())

# Vérifier les valeurs manquantes
print(bank_df.isnull().sum())

# Information sur les colonnes
print(bank_df.info())
```

#### 2. Prétraitement des données

```python
# Identifier les colonnes numériques et catégorielles
numeric_features = bank_df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = bank_df.select_dtypes(include=['object', 'category']).columns

# Nous utilisons uniquement certaines colonnes d'intérêt pour le clustering
selected_numeric = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
selected_categorical = ['job', 'marital', 'education', 'housing', 'loan']

# Définir les préprocesseurs
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

# Créer un préprocesseur pour toutes les colonnes
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, selected_numeric),
        ('cat', categorical_transformer, selected_categorical)
    ])

# Appliquer la transformation
bank_df_processed = preprocessor.fit_transform(bank_df)
```

#### 3. Détermination du nombre optimal de clusters (méthode du coude)

```python
# Méthode du coude pour déterminer le nombre optimal de clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(bank_df_processed)
    inertia.append(kmeans.inertia_)

# Visualisation de la méthode du coude
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.grid(True)
plt.savefig('elbow_method.png')
plt.show()
```

#### 4. Création des clusters

```python
# Nombre optimal de clusters (à déterminer après visualisation de la méthode du coude)
n_clusters = 4  # À ajuster selon la méthode du coude

# Création du modèle K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(bank_df_processed)

# Ajouter les labels des clusters au dataframe original
bank_df['Cluster'] = cluster_labels
```

#### 5. Visualisation et analyse des clusters

```python
# Réduction de dimension pour visualisation (PCA)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(bank_df_processed)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels

# Visualisation des clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100)
plt.title('Visualisation des clusters de clients')
plt.savefig('customer_clusters.png')
plt.show()

# Analyse des caractéristiques numériques par cluster
cluster_analysis_num = bank_df.groupby('Cluster')[selected_numeric].mean()
print(cluster_analysis_num)

# Visualisation des caractéristiques numériques moyennes par cluster
plt.figure(figsize=(14, 10))
cluster_analysis_num.T.plot(kind='bar', figsize=(14, 8))
plt.title('Caractéristiques numériques moyennes par cluster')
plt.ylabel('Valeur moyenne')
plt.xlabel('Caractéristiques')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('cluster_characteristics.png')
plt.show()

# Analyse des caractéristiques catégorielles par cluster
for cat_feature in selected_categorical:
    plt.figure(figsize=(12, 6))
    for i in range(n_clusters):
        plt.subplot(1, n_clusters, i+1)
        cluster_data = bank_df[bank_df['Cluster'] == i]
        cluster_data[cat_feature].value_counts(normalize=True).plot(kind='pie', 
                                                                  autopct='%1.1f%%',
                                                                  title=f'Cluster {i}: {cat_feature}')
    plt.tight_layout()
    plt.savefig(f'cluster_{cat_feature}_distribution.png')
    plt.show()
```

#### 6. Interprétation des clusters et recommandations de produits

```python
# Analyse détaillée de chaque cluster
cluster_profiles = pd.DataFrame(index=range(n_clusters))

# Taille des clusters
cluster_profiles['Taille'] = bank_df['Cluster'].value_counts().sort_index().values

# Caractéristiques moyennes par cluster
for col in selected_numeric:
    cluster_profiles[col] = bank_df.groupby('Cluster')[col].mean().values

# Modes des catégories par cluster
for col in selected_categorical:
    for i in range(n_clusters):
        cluster_profiles.loc[i, f'{col}_principal'] = bank_df[bank_df['Cluster'] == i][col].mode()[0]

print(cluster_profiles)

# Exemple d'interprétation des clusters et définition de stratégies de recommandation
# À ajuster en fonction des résultats réels
clusters_interpretation = {
    0: "Jeunes actifs avec prêts: Solutions d'épargne progressive",
    1: "Clients établis avec équilibre financier: Investissements et épargne retraite",
    2: "Clients avec besoins de financement: Consolidation de prêts et assurances",
    3: "Clients seniors avec patrimoine: Services premium et gestion de patrimoine"
}

# Affichage des interprétations
for cluster, interpretation in clusters_interpretation.items():
    print(f"Cluster {cluster}: {interpretation}")
    
# Création d'une fonction de recommandation basique
def recommend_products(cluster_id):
    recommendations = {
        0: ["Épargne progressive", "Applications bancaires mobiles", "Cartes à cashback"],
        1: ["Fonds d'investissement", "Assurance vie", "Épargne retraite"],
        2: ["Consolidation de prêts", "Assurance emprunteur", "Refinancement hypothécaire"],
        3: ["Gestion de patrimoine", "Services bancaires premium", "Planification successorale"]
    }
    return recommendations.get(cluster_id, "Cluster non reconnu")

# Test de la fonction de recommandation
for i in range(n_clusters):
    print(f"Recommandations pour le cluster {i}:")
    print(recommend_products(i))
    print("---")
```

## Visualisation finale intégrée

```python
# Ce code assume que vous avez déjà les deux modèles créés précédemment
import matplotlib.pyplot as plt
import numpy as np

# Créer une figure avec deux sous-graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Premier graphique: Matrice de confusion (cas 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bon payeur', 'Mauvais payeur'], 
            yticklabels=['Bon payeur', 'Mauvais payeur'], ax=ax1)
ax1.set_title('Prédiction de défaut de paiement')
ax1.set_xlabel('Prédiction')
ax1.set_ylabel('Réel')

# Second graphique: Clusters (cas 2)
scatter = ax2.scatter(principal_components[:, 0], principal_components[:, 1], 
                     c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
ax2.set_title('Segmentation des clients')
ax2.set_xlabel('Composante principale 1')
ax2.set_ylabel('Composante principale 2')
legend = ax2.legend(*scatter.legend_elements(), title="Clusters")
ax2.add_artist(legend)

# Ajouter un titre global
fig.suptitle('Analyse financière: Prédiction de défaut et Segmentation clients', fontsize=16)
plt.tight_layout()
plt.savefig('finance_ml_combined_analysis.png', dpi=300)
plt.show()
```

## Note importante sur les datasets

Les datasets utilisés dans ce projet sont accessibles via scikit-learn:

1. Pour le dataset `credit-g` (German Credit Data):
```python
from sklearn.datasets import fetch_openml
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True)
```

2. Pour le dataset `bank-marketing`:
```python
from sklearn.datasets import fetch_openml
bank_data = fetch_openml(name='bank-marketing', version=1, as_frame=True)
```

Ces datasets seront automatiquement téléchargés lors de la première exécution.

## Conclusion

Ces deux cas pratiques démontrent l'application du machine learning à des problématiques financières:

1. **Apprentissage supervisé**: Nous avons développé un modèle de classification pour prédire les défauts de paiement, permettant à l'institution financière d'évaluer le risque associé à chaque client.

2. **Apprentissage non supervisé**: Nous avons segmenté les clients en clusters distincts pour faciliter la personnalisation des offres et recommandations de produits.

Les étudiants peuvent adapter ces exemples en modifiant les paramètres des modèles ou en explorant d'autres algorithmes (comme les SVM, les réseaux de neurones pour la classification, ou DBSCAN pour le clustering).

## Pour aller plus loin

- Implémenter d'autres algorithmes de classification (logistic regression, SVM, etc.)
- Explorer des techniques de feature engineering spécifiques aux données financières
- Mettre en place un système de recommandation basé sur les clusters identifiés
- Combiner les deux approches pour une stratégie marketing ciblée (par ex: identifier les produits à faible risque pour les clients à haut risque de défaut)
- Utiliser des méthodes d'évaluation plus avancées comme la validation croisée stratifiée
- Explorer d'autres méthodes de clustering comme DBSCAN ou l'analyse de clustering hiérarchique
