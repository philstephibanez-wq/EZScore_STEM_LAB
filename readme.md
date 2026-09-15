# EZScore Stem Lab R8

R8 consolide la détection structurelle avant d'ajouter de nouveaux signaux.

## 1. Pré-roll / post-roll corrigés

Les paroles peuvent commencer avant la première mesure détectée.

R8 étend uniquement l'enveloppe visuelle du premier bloc :

```text
premier mot chanté
↓
Bloc A
↓
mesure 1
```

Aucun timestamp canonique n'est déplacé.

Même principe pour un éventuel post-roll après la dernière mesure.

## 2. Timeline paroles complète

R7 pouvait sélectionner un mauvais cache Whisper si plusieurs modèles existaient.

R8 choisit désormais automatiquement le cache le plus complet selon :

```text
1. timestamp du dernier mot
2. nombre de mots utilisables
3. date de modification
```

Les mots vides et les mots à durée nulle sont exclus de l'analyse structurelle.

L'interface affiche la timeline réellement utilisée :

```text
nombre de mots
couverture jusqu'à X secondes
nom du fichier cache
```

## 3. Répétitions textuelles plus tolérantes

R7 comparait essentiellement phrase contre phrase.

R8 compare aussi des fenêtres de plusieurs phrases :

```text
1 à 4 phrases consécutives
```

Cela permet de reconnaître un refrain même si Whisper découpe :

```text
occurrence 1 : ligne A + ligne B + ligne C
occurrence 2 : ligne A + ligne B | ligne C
```

Les petites erreurs lexicales Whisper restent tolérées via similarité floue.

## 4. Blocs + paroles

La vue conserve :

```text
Bloc A / Bloc B / ...
mesures
timestamps
motifs d'accords
paroles
```

Les paroles sont présentées sur plusieurs lignes lisibles au lieu d'une seule
ligne continue.

## 5. Principes inchangés

```text
harmonie = signal principal
paroles répétées = confirmation structurelle
blocs = visualisation non destructive
```

Aucun timestamp d'accord, de mot ou d'audio n'est modifié.

## Lancement

```powershell
python -m pip install -r .\requirements.txt
python -m py_compile .\structure_lab.py
python -m py_compile .\stem_lab.py
python -m streamlit run .\stem_lab.py --server.port 8502
```


## Correctif R8.1

Correction d'un oubli d'import Python dans `stem_lab.py` :

```python
import re
```

Cet oubli provoquait :

```text
NameError: name 're' is not defined
```

dans la vue `Blocs + paroles`.

Aucun changement algorithmique par rapport à R8.
