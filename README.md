# 🦟 Prédiction des cas de Paludisme au Cameroun

Pipeline de Machine Learning pour la prédiction des cas **graves** et **simples** de paludisme au Cameroun, par population (femmes enceintes, enfants de moins de 5 ans, population générale), à partir de données climatiques, démographiques et d'interventions de santé publique.



---

## 📌 Contexte

Le jeu de données couvre **9 504 observations** (198 districts de santé × 4 années × 12 mois, 2021–2024) et combine trois types de variables :

- **Climatiques** : précipitations, température (moyenne/min/max), humidité relative, vent (NASA POWER)
- **Démographiques** : population par district, par tranche d'âge, femmes enceintes
- **Interventions de santé publique** : doses de TPI (traitement préventif intermittent), distribution de MILDA (moustiquaires imprégnées)

**Objectif** : construire le meilleur modèle de prédiction possible pour 4 cibles :

| Population | Cible(s) |
|---|---|
| Enfants < 5 ans | `severe_under5` (cas graves) |
| Femmes enceintes | `severe_pregnant` (cas graves) |
| Population générale | `severe_5plus` (cas graves) + `simple_cases` (cas simples, non désagrégés) |

---

## 🧠 Approche

Le notebook suit un pipeline complet en 15 étapes :

1. **Audit de qualité des données** — valeurs manquantes, doublons, complétude du panel, détection d'outliers
2. **Prétraitement & feature engineering** — encodage cyclique du mois, décomposition de la zone agro-écologique, variables de couverture d'intervention (dose/population), amplitude thermique, interactions climatiques
3. **Variables décalées (lags)** — `_lag1`, `_lag2`, moyennes glissantes 3/6 mois sur le climat, pour refléter le cycle biologique du vecteur anophèle
4. **EDA approfondie** — inflation de zéros (~61 %), non-stationnarité temporelle, saisonnalité, hétérogénéité régionale, corrélations inter-cibles
5. **Anti-fuite de données** — vérification systématique qu'aucun agrégat de cible ne fuite dans les features
6. **Stratégie de validation** — split stratifié par année (80/20) + `GroupKFold` par district pour éviter la fuite inter-districts
7. **Transformation `log1p`** des cibles pour corriger l'asymétrie et l'inflation de zéros
8. **Modélisation comparative** — Dummy, Ridge, ExtraTrees, Random Forest, HistGradientBoosting, Stacking (RF + HGB → méta-Ridge), avec recherche d'hyperparamètres (`RandomizedSearchCV`)
9. **Prédiction multi-cible** — `MultiOutputRegressor` et `RegressorChain`, motivés par des corrélations inter-cibles de 0.78–0.91
10. **Analyse de stabilité** — variance du R² sur 4 plis (`GroupKFold` par district)
11. **Robustesse temporelle** — entraînement sur 2021–2023, test sur 2024, pour mesurer la généralisation
12. **Analyse fine des erreurs** — résidus par région, par mois, réel vs prédit
13. **Interprétabilité** — importance MDI et Permutation Importance, comparées et interprétées par population
14. **Quantification d'incertitude** — intervalles de prédiction à 80 % via régression quantile (`HistGradientBoostingRegressor`, loss quantile)
15. **Synthèse, limites et recommandations pour la production**

---

## 📊 Résultats

| Cible | Meilleur modèle | R² (test) | Facteur dominant |
|---|---|---|---|
| `severe_under5` | Stacking / RF | ~0.83–0.88 | Couverture MILDA nourrissons |
| `severe_pregnant` | Stacking / RF | ~0.88–0.92 | Couverture TPI femmes enceintes |
| `severe_5plus` | RF / HistGB | ~0.91–0.93 | Agrégats démographiques du district |
| `simple_cases` | RF / Stacking | ~0.84–0.88 | Taille de la population |

Le stacking apporte un gain marginal mais systématique sur toutes les cibles ; la prédiction multi-cible surpasse les approches mono-cible en exploitant la corrélation entre catégories de cas graves.

### Limites identifiées

- `simple_cases` n'est pas désagrégé par population dans les données sources (un seul modèle global)
- Non-stationnarité forte (×~500 en 4 ans) probablement liée à la montée en charge du système de surveillance DHIS2 plutôt qu'à une réalité épidémiologique — fragilise l'extrapolation temporelle
- Confusion possible entre taille de district et couverture d'intervention (pas de causalité directe)
- 61 % de zéros non modélisés explicitement ; une approche de comptage (Poisson, binomiale négative, zero-inflated) serait théoriquement plus adaptée

---

## 📁 Structure du dépôt

```
.
├── Challenge_Paludisme.ipynb   # Notebook principal (pipeline complet)
├── Data_Palu.xlsx                 # Jeu de données brut
├── Dictionnaire_Palu.docx         # Dictionnaire des variables
├── Challenge_Palu_v1.docx # Énoncé du challenge
└── README.md
```

---

## 🛠️ Stack technique

- **Python** : `pandas`, `numpy`
- **Visualisation** : `matplotlib`, `seaborn`
- **Machine Learning** : `scikit-learn` (RandomForest, ExtraTrees, HistGradientBoosting, Stacking, MultiOutputRegressor, RegressorChain, RandomizedSearchCV, GroupKFold, régression quantile, permutation importance)

---

## ▶️ Reproduire les résultats

```bash
git clone https://github.com/lethyciamelong/Prediction-Paludisme-Cameroun.git
cd Prediction-Paludisme-Cameroun
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
jupyter notebook Challenge_Paludisme.ipynb
```
