# Cas Pratiques de Machine Learning en Finance

Ce projet propose deux cas d'usage de machine learning appliqués au domaine financier:
1. **Prédiction de défaut de paiement** (apprentissage supervisé)
2. **Segmentation de clients pour recommandation de produits** (apprentissage non supervisé)

## Prérequis

```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Cas 1: Prédiction de défaut de paiement (Supervised Learning)

### Objectif
Développer un modèle qui prédit la probabilité qu'un client ne rembourse pas son prêt en fonction de ses caractéristiques financières et personnelles.

### Données
Nous utiliserons un dataset contenant les informations suivantes pour chaque client:
- Âge
- Revenu annuel
- Montant du prêt
- Durée du prêt
- Taux d'endettement
- Historique de crédit (nombre de défauts précédents)
- Ancienneté professionnelle
- Variable cible: défaut de paiement (1 = défaut, 0 = remboursement complet)

### Approche étape par étape

#### 1. Chargement et exploration des données

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Charger les données
df = pd.read_csv("credit_data.csv")  # Remplacer par le chemin de vos données

# Afficher les premières lignes
print(df.head())

# Statistiques descriptives
print(df.describe())

# Vérifier les valeurs manquantes
print(df.isnull().sum())
```

#### 2. Prétraitement des données

```python
# Traiter les valeurs manquantes (exemple)
df.fillna(df.mean(), inplace=True)

# Séparer les features et la variable cible
X = df.drop('default', axis=1)  # 'default' est la colonne cible
y = df['default']

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardisation des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### 3. Entraînement du modèle

```python
# Initialiser le modèle RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
model.fit(X_train_scaled, y_train)

# Prédictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
```

#### 4. Évaluation du modèle

```python
# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Remboursé', 'Défaut'], 
            yticklabels=['Remboursé', 'Défaut'])
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
# Importance des features
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Importance des caractéristiques dans la prédiction de défaut')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
```

#### 6. Optimisation du modèle (Bonus)

```python
from sklearn.model_selection import GridSearchCV

# Définir les hyperparamètres à tester
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Recherche par grille
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                           param_grid, 
                           cv=5, 
                           scoring='roc_auc', 
                           n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Meilleurs paramètres
print("Meilleurs paramètres:", grid_search.best_params_)
print("Meilleur score:", grid_search.best_score_)

# Modèle optimisé
best_model = grid_search.best_estimator_
```

## Cas 2: Segmentation de clients pour recommandation de produits (Unsupervised Learning)

### Objectif
Segmenter les clients de la banque en groupes distincts pour faciliter la recommandation de produits financiers adaptés à chaque segment.

### Données
Nous utiliserons un dataset contenant les informations suivantes pour chaque client:
- Âge
- Revenu annuel
- Solde moyen des comptes
- Nombre de produits détenus
- Montant des investissements
- Montant d'épargne
- Nombre de transactions mensuelles
- Durée de la relation avec la banque

### Approche étape par étape

#### 1. Chargement et exploration des données

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Charger les données
df = pd.read_csv("customer_data.csv")  # Remplacer par le chemin de vos données

# Afficher les premières lignes
print(df.head())

# Statistiques descriptives
print(df.describe())

# Vérifier les valeurs manquantes
print(df.isnull().sum())
```

#### 2. Prétraitement des données

```python
# Traiter les valeurs manquantes (exemple)
df.fillna(df.mean(), inplace=True)

# Standardisation des features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```

#### 3. Détermination du nombre optimal de clusters (méthode du coude)

```python
# Méthode du coude pour déterminer le nombre optimal de clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
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
cluster_labels = kmeans.fit_predict(df_scaled)

# Ajouter les labels des clusters au dataframe original
df['Cluster'] = cluster_labels
```

#### 5. Visualisation et analyse des clusters

```python
# Réduction de dimension pour visualisation (PCA)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels

# Visualisation des clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100)
plt.title('Visualisation des clusters de clients')
plt.savefig('customer_clusters.png')
plt.show()

# Analyse des caractéristiques par cluster
cluster_analysis = df.groupby('Cluster').mean()
print(cluster_analysis)

# Visualisation des caractéristiques moyennes par cluster
plt.figure(figsize=(14, 10))
cluster_analysis.T.plot(kind='bar', figsize=(14, 8))
plt.title('Caractéristiques moyennes par cluster')
plt.ylabel('Valeur moyenne')
plt.xlabel('Caractéristiques')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('cluster_characteristics.png')
plt.show()
```

#### 6. Interprétation des clusters et recommandations de produits

```python
# Exemple d'interprétation des clusters et définition de stratégies de recommandation
clusters_interpretation = {
    0: "Clients à faible engagement: Produits simples et accessibles",
    1: "Clients fortunés: Produits d'investissement premium",
    2: "Clients actifs à revenu moyen: Produits d'épargne et assurances",
    3: "Clients fidèles: Programmes de fidélité et services personnalisés"
}

# Affichage des interprétations
for cluster, interpretation in clusters_interpretation.items():
    print(f"Cluster {cluster}: {interpretation}")
    
# Création d'une fonction de recommandation basique
def recommend_products(cluster_id):
    recommendations = {
        0: ["Compte courant basique", "Carte de débit standard", "Application mobile simplifiée"],
        1: ["Gestion de patrimoine", "Investissements internationaux", "Assurance-vie premium"],
        2: ["Épargne programmée", "Crédit immobilier", "Assurance multirisque"],
        3: ["Carte premium avec avantages", "Conseiller personnel dédié", "Produits d'épargne retraite"]
    }
    return recommendations.get(cluster_id, "Cluster non reconnu")

# Test de la fonction de recommandation
for i in range(n_clusters):
    print(f"Recommandations pour le cluster {i}:")
    print(recommend_products(i))
    print("---")
```

## Visualisation finale intégrée

Voici un exemple de visualisation qui combine les résultats des deux modèles:

```python
# Ce code assume que vous avez déjà les deux modèles créés précédemment
import matplotlib.pyplot as plt
import numpy as np

# Créer une figure avec deux sous-graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Premier graphique: Matrice de confusion (cas 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Remboursé', 'Défaut'], 
            yticklabels=['Remboursé', 'Défaut'], ax=ax1)
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
