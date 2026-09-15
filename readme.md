# EZScore Stem Lab R9 — architecture modulaire

R9 ne change pas volontairement le principe validé de R8.1.  
Le but est de transformer le prototype en composants réutilisables par EZScore.

## Architecture

```text
stem_lab.py
│
└── stemlab/
    ├── config.py
    ├── io.py
    │
    ├── models/
    │   └── timelines.py
    │
    ├── analysis/
    │   ├── separation.py
    │   ├── lyrics.py
    │   ├── rhythm.py
    │   ├── harmony.py
    │   ├── lyrics_structure.py
    │   ├── structure.py
    │   └── structure_engine.py
    │
    └── player/
        └── webaudio.py
```

## Responsabilités

### `analysis/separation.py`

```text
audio original
→ Demucs
→ vocals / drums / bass / other
```

Le cache reste indexé par hash audio.

### `analysis/lyrics.py`

```text
source audio
→ Whisper
→ texte
→ mots horodatés
```

L'API `transcribe_audio()` est générique et prépare la prochaine étape :

```text
original + vocals
→ comparaison / fusion
```

La fonction actuelle `transcribe_vocals()` reste compatible avec R8.

### `analysis/harmony.py`

Expose l'analyse harmonique validée :

```text
other = harmonie principale
bass  = indice secondaire de fondamentale
drums = beat tracking
```

### `analysis/lyrics_structure.py`

```text
mots
→ phrases
→ motifs textuels répétés
```

### `analysis/structure.py`

Fusion non destructive :

```text
motifs harmoniques
+
motifs textuels
→ candidats structurels
→ blocs visuels
```

Ce module ne déplace aucun timestamp.

### `models/timelines.py`

Contrats communs pour la future intégration EZScore.

Règle fondamentale :

```text
TOUS les timestamps sont exprimés en secondes
sur l'audio ORIGINAL.
```

Les timelines sont indépendantes.

### `player/webaudio.py`

Lecteur WebAudio à horloge unique :

```text
original + vocals + drums + bass + other
```

Aucun micro-seek de resynchronisation périodique.

## Simplification R9

La traduction et eSpeak/phonemizer ont été retirés du chemin principal.

Ils n'apportaient rien au moteur structurel actuellement validé.

Le pipeline reste concentré sur :

```text
stems
harmonie
rythme
paroles
répétitions
blocs
```

## Compatibilité des données

R9 conserve :

```text
stem_lab_data/<audio_hash>/
```

et relit les stems, previews, caches Whisper et `structure_analysis.json`
déjà produits par R8.1.

Il n'est donc pas nécessaire de refaire Demucs.

## Lancement

```powershell
python -m pip install -r .\requirements.txt
python -m compileall .\stemlab .\stem_lab.py
python -m streamlit run .\stem_lab.py --server.port 8502
```

## Étape suivante

Une fois ce checkpoint validé :

```text
Whisper original
+
Whisper vocals
→ fusion de la meilleure timeline paroles
```

Puis cette timeline alimentera le même moteur `lyrics_structure.py`.

Cette séparation est volontairement alignée sur la future migration vers EZScore.
