# Silenzio
C’est un projet d'école. L’idée est simple : une application de recommandation de films pour la Creuse, où l'on privilégie la qualité du moment plutôt que la quantité de titres.

# Pourquoi ce projet ?
Le projet est né d'une étude de marché basée sur les données du CNC et de l'INSEE. L'objectif était de comprendre la consommation réelle des Creusois (salles et streaming).

Le constat : Un fort attachement aux films d'Art & Essai et aux comédies françaises. Silenzio répond à ce besoin en s'éloignant des standards des blockbusters poussés par les algorithmes classiques.

# Stack Technique
J'ai utilisé une approche orientée Data Science pour construire ce moteur de recommandation :

Data & Analyse : Python (Pandas, NumPy) pour le traitement des données INSEE/CNC.
Visualisation : Matplotlib & Seaborn pour l'étude de marché préliminaire.
Acquisition : Web Scraping (AFCAE) & API (TMDB) pour construire la base de données films.
Intelligence (NLP) : Utilisation de la Similitude Cosinus (Cosine Similarity) pour recommander des films basés sur la proximité sémantique des synopsis.

Interface : Déploiement d'une application web interactive via Streamlit.


