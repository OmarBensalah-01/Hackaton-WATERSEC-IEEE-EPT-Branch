# 💧 WaterSec Hackathon — Gestion Intelligente de la Consommation d'Eau

## Introduction
Ce projet regroupe le travail réalisé pour le hackathon WaterSec. Il vise à construire une plateforme capable de surveiller, analyser et expliquer les consommations d'eau à partir de jeux de données hétérogènes.

## Objectifs du hackathon
- Structurer et explorer des données réelles de consommation d'eau.
- Construire des modèles de détection d'anomalies robustes.
- Enrichir le contexte métier avec des signaux externes (météo, horaires de prière).
- Déployer un agent conversationnel capable de répondre à des questions métier.

## Notre solution
WaterSec combine plusieurs couches techniques pour produire une analyse fiable :
- Prétraitement et nettoyage des jeux de données clients.
- Feature engineering sur les séries temporelles.
- Détection d'anomalies multicanal (statistique, machine learning, règles métiers).
- Orchestration d'un agent conversationnel RAG + LLM pour interpréter les résultats.

## Organisation du projet
1. **Exploration des données** : compréhension des fichiers, nettoyage, et enrichissement.
2. **Modélisation** : scores statistiques, isolation forest, prévision, détection de patterns.
3. **Architecture** : pipeline de traitement, stockage sémantique et agent LangGraph.
4. **Démo et export** : visualisation, rapports CSV, et tests de requêtes.

## Exploration du dataset
Le dossier `data/` contient :
- `customerA_consumption.csv`
- `customerB_consumption.csv`
- `customerC_consumption.csv`
- `gym_consumption_data.csv`

Chaque dataset représente un profil d'utilisation différent :
- customer A : usages de bureaux
- customer B : bloc sanitaire
- customer C : usages résidentiels
- gym : cabines de douche et gestion eau chaude/froide

La notebook `notebooks/EPT_Beginners1.ipynb` présente :
- lecture et validation des timestamps
- nettoyage des anomalies structurelles
- consolidation des colonnes communes
- enrichissement des données par heure, jour, week-end et conditions externes

## Modèles et architecture
### Modèles utilisés
- **IsolationForest** : détection d'anomalies multivariées.
- **Z-Score / IQR** : détection statistique simple et interprétable.
- **Prophet** : prévision de consommation sur 7 jours.
- **Analyse séquentielle (Markov)** : détection de chaînes d'usages.
- **Clustering** : segmentation des profils clients.

### Architecture technique
- **Couche données** : pandas, nettoyage et feature engineering.
- **Couche détection** : anomalies basées sur statistiques et apprentissage.
- **Couche contextuelle** : météo, horaires de prière, règles métiers.
- **Couche conversationnelle** : LangGraph + Groq LLM + RAG semantique.

## Exécution
1. Ouvrir le notebook `notebooks/EPT_Beginners1.ipynb`.
2. Exécuter les cellules dans l'ordre.
3. Si nécessaire, installer les dépendances listées dans `requirements.txt`.

## Structure du dépôt
- `data/` : jeux de données sources
- `notebooks/` : notebook principal et analyses
- `presentation/` : supports de présentation
- `README.md` : documentation du projet
- `requirements.txt` : dépendances Python

## À noter
Le notebook a été organisé pour séparer clairement la partie exploration des données de la partie modélisation et architecture. Les résultats sont exportables en CSV pour faciliter les reportings.

```
┌─────────────────────────────────────────────────────────────────┐
│                  REQUÊTE LANGAGE NATUREL                        │
│         (Ex: "Analyser les fuites sur Client A cette semaine")  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │  LangGraph Router   │
                │ (Classification IA) │
                └──┬───┬──────┬───────┘
                   │   │      │
        ┌──────────┘   │      └──────────────┐
        │              │                     │
    QUERY      ANALYSIS_STAT           ANOMALY_DETECT
        │              │                     │
        └──────────────┼─────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   COUCHE TRAITEMENT DONNÉES │
        ├──────────────────────────────┤
        │ • Pandas SQL-like queries    │
        │ • Feature engineering        │
        │ • Time series analysis       │
        │ • Statistical inference      │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼──────────────┐
        │    COUCHE MODÈLES ML        │
        ├──────────────────────────────┤
        │ • Isolation Forest           │
        │ • Z-Score / IQR              │
        │ • Prophet forecasting        │
        │ • Pattern detection (Markov) │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   GROQ LLM (Raisonnement)   │
        │  llama-3.3-70b-versatile    │
        │   Contexte: RAG + Mémoire   │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼──────────────┐
        │    RÉPONSE FORMATÉE + KPI   │
        │   Dashboard HTML Exportable  │
        └──────────────────────────────┘
```

---

## 📊 Approches Techniques Détaillées

### 1️⃣ **Nettoyage et Unification des Données**

#### Défis Multi-Source
- **4 sources hétérogènes** : Gym (hot/cold), CustomerA (offices), CustomerB (bloc sanitaire), CustomerC (résidentiel)
- **Formats divergents** : séparateurs (`,` vs `;`), colonnes mal nommées, structures différentes
- **Timestamps corrompus** : mélange epoch Unix (secondes) + ISO8601 + dates hors plage

#### Pipeline Robuste
```python
def safe_parse_timestamp(series: pd.Series) -> pd.Series:
    """
    1. Parse ISO8601 standard → pd.to_datetime(..., errors='coerce')
    2. Détecte Unix valide → numeric > 1.5e9 et < 1.9e9 (2020-2030)
    3. Filtre cohérence temporelle → drop NaT et dates aberrantes
    4. Retour : Series avec timestamps validés
    """
```

**Résultat** : dataset unifié `(customer, device, consumption, timestamp, tag, category, sub_category)`

---

### 2️⃣ **Feature Engineering Avancé**

#### Features Temporelles
| Feature | Formule | Utilité |
|---------|---------|---------|
| **Hour-of-day** | `timestamp.dt.hour` | Capture profil horaire client |
| **Day-of-week** | `timestamp.dt.dayofweek` | Distinction semaine/weekend |
| **Rolling mean (24h)** | `consumption.rolling(24).mean()` | Lissage + tendance |
| **Rolling std (24h)** | `consumption.rolling(24).std()` | Volatilité locale |

#### Features Hydrauliques
| Feature | Calcul | Interprétation |
|---------|--------|-----------------|
| **Water Demand Index (WDI)** | Normalisé par heure & profil | Demande relative vs baseline |
| **Nocturnal Flow Ratio** | `débit_nuit / débit_jour` | Indicateur de fuite |
| **Efficiency Score** | `consommation_réelle / benchmark` | Performance vs normes |

#### Features Comportementales
- **Sequence Detection** : Chaînes d'événements (Flush→Sink→Tap) modélisées en **Markov states**
- **Periodicity Strength** : FFT sur série temporelle pour détecter cycles cachés
- **User Signature** : Profil unique par device (patterns récurrents)

---

### 3️⃣ **Détection d'Anomalies Multi-Couche**

#### Couche 1 : Statistique
```python
# Z-Score standard (< -3σ ou > +3σ)
z_scores = np.abs((X - X.mean()) / X.std())
anomalies_zscore = z_scores > 3

# IQR Tukey (< Q1-1.5×IQR ou > Q3+1.5×IQR)
Q1, Q3 = X.quantile([0.25, 0.75])
IQR = Q3 - Q1
anomalies_iqr = (X < Q1 - 1.5*IQR) | (X > Q3 + 1.5*IQR)
```

#### Couche 2 : Isolation Forest
```python
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_scores = iso_forest.fit_predict(features_scaled)
# Retour : -1 (anomalie) ou +1 (normal)
```

**Avantage** : Détecte anomalies multivariées, insensible aux outliers extrêmes.

#### Couche 3 : Détection de Fuites Persistantes
```python
# Fuite = débit nocturne CONTINU > threshold pendant N minutes
night_flow = consumption[(timestamp.hour >= 0) & (timestamp.hour < 6)]
persistent_flow = night_flow[night_flow > 0.1].groupby(
    (night_flow.index.to_period('H')).diff().ne(0).cumsum()
).size() > 30  # durée minimale

leaks = persistent_flow[persistent_flow].index
```

---

### 4️⃣ **Clustering et Segmentation**

#### K-Means + Silhouette Score
```python
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    score = silhouette_score(features_scaled, kmeans.labels_)
    # Sélection k optimal à coude du graphe score(k)

clusters = kmeans.labels_
# Profils : "Heavy Users", "Efficient", "Erratic", etc.
```

#### DBSCAN (géométrique)
- **eps** = 0.5 (rayon recherche)
- **min_samples** = 5 (densité minimale)
- Détecte **clusters de formes quelconques** + **outliers isolés**

---

### 5️⃣ **Pipeline LLM Conversationnel (LangGraph)**

#### Architecture État-Machine

**États** :
1. `"input"` → parsing & classification d'intention
2. `"retrieve"` → RAG sémantique (ChromaDB) + query dataset
3. `"analyze"` → calculs ML (stats, clustering, prédictions)
4. `"format"` → génération réponse textuelle via Groq
5. `"output"` → retour utilisateur + contexte mémorisé

#### Classification d'Intention (LLM)
```
Groq classify: "Quels clients ont des fuites cette semaine ?"
→ Intent: ANOMALY_DETECT
→ Params: scope=clients, anomaly_type=leaks, time_window=7j
```

#### RAG Sémantique
- **Encoder** : `sentence-transformers` (multilingual)
- **Vectorstore** : ChromaDB (persistant local)
- **Documents indexés** : descriptions clients, seuils, patterns connus
- **Retrieval** : Top-3 similaires à requête encodée

#### Mémoire Conversationnelle
```python
conversation_history = deque(maxlen=5)
# Contexte des 5 derniers échanges → injecté dans prompt Groq
```

---

### 6️⃣ **Forecasting Prophet**

#### Modèle Time Series
```python
from prophet import Prophet

df_prophet = pd.DataFrame({
    'ds': timestamps,
    'y': consumption
})
model = Prophet(
    changepoint_prior_scale=0.05,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True
)
model.fit(df_prophet)

# Prédiction 7 jours
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)
```

**Capabilités** :
- Décomposition trend + seasonalité (yearly, weekly, daily)
- Intervalles de confiance (80%, 95%)
- Robustesse aux valeurs manquantes

---

## 📁 Structure du Dépôt

```
hackaton-watersec/
├── data/
│   ├── customerA_consumption.csv     # Offices (100+ appareils)
│   ├── customerB_consumption.csv     # Bloc sanitaire
│   ├── customerC_consumption.csv     # Résidentiel
│   └── gym_consumption_data.csv      # Gym (4 cabines × hot/cold)
│
├── notebooks/
│   ├── EPT_Beginners1.ipynb          # Pipeline IA complet
│   │   ├── Section 0: Dépendances
│   │   ├── Section 1: Config (Groq API, chemins)
│   │   ├── Section 2: Couche données (load + unification)
│   │   ├── Section 3: Feature engineering
│   │   ├── Section 4: Anomaly detection multi-couche
│   │   ├── Section 5: Clustering K-Means + DBSCAN
│   │   ├── Section 6: Prophet forecasting
│   │   ├── Section 7: LangGraph + Groq LLM
│   │   ├── Section 8: Dashboard KPI HTML
│   │   └── Section 9: Démonstration end-to-end
│   │
│   └── eda-part.ipynb                # Exploration & diagnostic
│       ├── Chargement données multi-source
│       ├── Distributions & outliers
│       ├── Profils horaires (heatmaps)
│       ├── Comparaisons inter-clients
│       ├── Corrélations capteurs
│       └── Visualisations Plotly
│
├── presentation/
│   └── WaterSec_AI_Presentation.pptx # Slides compétition
│
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
└── .gitignore                         # Exclusions Git
```

---

## 🚀 Installation et Utilisation

### Prérequis
- **Python 3.12+**
- **Git**
- **API Key Groq** (gratuite) : https://console.groq.com

### Setup Local

```bash
# 1. Cloner
git clone https://github.com/OmarBensalah-01/hackaton-watersec.git
cd hackaton-watersec

# 2. Environnement virtuel
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Dépendances
pip install -r requirements.txt

# 4. Configurer Groq API
$env:GROQ_API_KEY = "gsk_your_api_key_here"  # PowerShell
# ou export GROQ_API_KEY="..." sous Linux

# 5. Lancer Jupyter
jupyter lab notebooks/EPT_Beginners1.ipynb
```

### Utilisation

**Notebook Principal (EPT_Beginners1.ipynb)**
- Exécuter **Section 0** : Installation dépendances (une seule fois)
- Exécuter **Section 1** : Configuration (adapter chemins si nécessaire)
- Exécuter **Section 2-9** : Pipeline complet (5-10 min)

**Notebook EDA (eda-part.ipynb)**
- Pour exploration rapide des données
- Visualisations Plotly interactives
- Diagnostic avant modélisation

---

## 📈 Résultats Clés

### Statistiques Dataset
| Métrique | Valeur |
|----------|--------|
| **Période totale** | Jan 2024 - Avr 2026 |
| **Enregistrements** | 500K+ (après nettoyage) |
| **Clients** | 4 (Gym, A, B, C) |
| **Appareils/capteurs** | 100+ (variable par client) |
| **Taux anomalies détectées** | 3-5% (IsoForest) |

### Exemples Détections
- **Fuites persistantes** : Débits nocturnes > 0.1 L/min pendant 30+ min
- **Patterns comportementaux** : Séquences Flush→Sink→Tap sur 5-10 min
- **Anomalies volumétriques** : Usage 3σ au-delà historique client

---

## 🔬 Méthodologie Avancée

### Validation Croisée
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # Fit sur train, éval sur test temporellement disjoints
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

### Normalisation Robuste
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()  # Insensible aux outliers extrêmes
features_scaled = scaler.fit_transform(features)
```

### Hyperparamètre Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'contamination': [0.03, 0.05, 0.1]
}
grid = GridSearchCV(IsolationForest(), param_grid, cv=5)
grid.fit(features_scaled)
best_model = grid.best_estimator_
```

---

## 🔐 Sécurité & Bonnes Pratiques

- **Groq API Key** : Stockée en variable d'environnement (jamais en clair)
- **Data Privacy** : Anonymisation clients (A, B, C, Gym)
- **Validation Entrées** : Checks timestamp, consumption > 0
- **Logging** : Traçabilité des requêtes conversationnelles

---

## 📚 Références Techniques

| Technique | Source | Application |
|-----------|--------|------------|
| **Isolation Forest** | Liu et al., 2008 | Détection anomalies multivariées |
| **Prophet** | Facebook Research, 2017 | Time series forecasting |
| **LangGraph** | LangChain, 2024 | Orchestration état-machine IA |
| **Markov Chains** | Andrey Markov, 1913 | Pattern sequencing |
| **RAG** | Lewis et al., 2020 | Retrieval-Augmented Generation |

---

## 🤝 Contributions

Ce projet a été développé comme solution au **Hackathon WaterSec** avec l'objectif d'optimiser et monitorer la gestion de la consommation d'eau.

**Équipe** : WaterSec Hackathon Participants

---

## 📄 License

MIT License — Libre d'utilisation à usage éducatif et commercial.

---

## ❓ FAQ

**Q: Comment adapter le modèle à mon contexte (pays, périodes d'usage) ?**  
R: Modifiez `SITE_LAT`, `SITE_LON`, `PRAYER_METHOD` en Section 1 du notebook pour ajuster les horaires de prière et métadonnées géographiques.

**Q: Peut-on utiliser un autre LLM que Groq ?**  
R: Oui, remplacez l'intégration Groq par OpenAI, Claude, ou Mistral en Section 7 (requiert clé API respective).

**Q: Quels seuils pour déclarer une anomalie ?**  
R: Par défaut Z-score > 3σ, IQR Tukey, contamination IsoForest = 5%. À adapter selon seuil métier.

---

**Dernière mise à jour** : Mai 2026  
**Version** : 1.0 (Hackathon)
