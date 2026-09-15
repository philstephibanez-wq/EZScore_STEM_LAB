# EZScore Stem Lab R7

R7 ajoute une **vue Blocs + paroles**, inspirée d'EZScore.

Le pipeline reste :

```text
other.wav  → harmonie principale
bass.wav   → indice secondaire de fondamentale
drums.wav  → tempo / beats
vocals.wav → Whisper / mots horodatés
```

Puis :

```text
motifs harmoniques répétés
+
paroles répétées
→ candidats structurels
→ blocs visuels
→ affichage des paroles dans chaque bloc
```

## Vue Blocs + paroles

Chaque bloc affiche :

```text
Bloc A · mesures 1–...
timestamps
motifs d'accords par mesure
paroles appartenant à l'intervalle temporel du bloc
```

Les paroles sont présentées dans un panneau visuel avec une barre bleue,
proche du principe de lecture d'EZScore.

## Important

Les blocs sont **strictement visuels** :

```text
aucun timestamp de mot déplacé
aucun timestamp d'accord déplacé
aucune modification du fichier audio
```

Les frontières sont proposées à partir des motifs harmoniques forts.

Les sections harmoniquement similaires reçoivent le même identifiant :

```text
Bloc A
Bloc B
Bloc A
...
```

## Compatibilité avec R6

Si `structure_analysis.json` a été généré avec R6 et ne contient pas encore
`visual_blocks`, R7 les reconstruit automatiquement à l'affichage.

Il n'est donc pas obligatoire de refaire immédiatement l'analyse.

## Installation

```powershell
python -m pip install -r .\requirements.txt
python -m py_compile .\structure_lab.py
python -m py_compile .\stem_lab.py
python -m streamlit run .\stem_lab.py --server.port 8502
```

## Fichiers

```text
stem_lab.py
structure_lab.py
requirements.txt
readme.md
```

Ce prototype ne modifie toujours pas EZScore principal.
