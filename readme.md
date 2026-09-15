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


## Correctif R9.1

Correction d'une dépendance oubliée lors du découpage modulaire :

```python
from stemlab.config import STEM_NAMES
```

dans :

```text
stemlab/player/webaudio.py
```

Le bug provoquait :

```text
NameError: name 'STEM_NAMES' is not defined
```

au rendu du lecteur.

Aucun changement algorithmique.


## R9.2 — checkpoint validé : chant avant accompagnement

Invariant confirmé par test :

```text
début du chant ≠ début de l'accompagnement
```

Le moteur paroles peut commencer à `t=0.0s` même si la première mesure / le
premier beat détecté commence plus tard.

Les blocs restent purement visuels :

```text
premier mot chanté à 0.0s
↓
enveloppe visuelle du premier bloc étendue jusqu'au chant
↓
première mesure détectée plus tard
```

Aucun timestamp de parole, d'accord, de beat ou de mesure n'est déplacé.

Ce comportement devra être conservé lors de la convergence vers EZScore.


# R10 — paroles sur l'audio original

R10 conserve le workflow R9 mais change une seule responsabilité majeure :

```text
AVANT
vocals.wav (Demucs)
→ Whisper
→ paroles

R10
audio original MP3/WAV
→ Whisper
→ paroles
```

## Pourquoi

Le test `Tombe la neige` a montré que le stem `vocals.wav` dégrade fortement
la reconnaissance lexicale (consonnes et mots altérés).

EZScore utilise déjà Whisper sur l'audio original complet et obtient de
meilleurs résultats.

## Pipeline R10

```text
Original → Whisper SMALL → paroles + mots horodatés
Vocals   → réservé à la future mélodie / F0
Drums    → rythme / beats
Bass     → fondamentale auxiliaire
Other    → harmonie / accords
```

Le modèle Whisper par défaut passe à `small`, comme dans EZScore.

## Cache

Les nouvelles transcriptions sont isolées :

```text
speech_analysis/whisper_original_small.json
speech_analysis/whisper_original_base.json
...
```

Les anciens caches construits depuis `vocals.wav` peuvent rester sur disque :
ils ne sont plus sélectionnés silencieusement par le lecteur ni par l'analyse
structurelle.

## Invariant temporel

Tous les timestamps Whisper restent relatifs à l'audio original.

Le chant peut commencer avant la première mesure détectée. Le pré-roll reste
conservé sans déplacer les beats, mesures, accords ou paroles.

## Test recommandé

Pour `Tombe la neige` :

1. charger le morceau ;
2. sélectionner `small` ;
3. cliquer `Analyser les paroles` ;
4. vérifier le texte exporté ;
5. seulement ensuite relancer `Analyser les motifs structurels`.
