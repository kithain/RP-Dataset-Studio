# RP Dataset Studio — Éditeur local de datasets RP

Un éditeur graphique **100 % local** pour préparer les datasets de jeu de rôle destinés au fine-tuning Unsloth/QLoRA.
Il fonctionne sur Windows 10/11 avec Python et s'ouvre dans un navigateur à l'adresse `http://127.0.0.1:8501`.

**Version actuelle : 0.02a (alpha) — 21 octobre 2026.**

Le code et son historique sont disponibles sur [GitHub](https://github.com/kithain/RP-Dataset-Studio).
Le dépôt contient le programme, ses tests et un petit dataset de démonstration. Les datasets personnels, projets `.rpwork.json`, exports et environnements Python locaux sont exclus du suivi Git par `.gitignore`.

## Changements de la version 0.02a

Cette version ajoute la fusion de datasets et la conversion des conversations récupérables pour Unsloth.

### Nouveautés

- **Fusion de plusieurs fichiers** depuis des fichiers importés ou des chemins locaux, avec bilan par source, réparations JSONL, retrait des doublons entre fichiers et mélange reproductible. Le JSONL Unsloth est préparé dès la fin de la fusion ; le bilan peut être téléchargé séparément.
- **Conversion dans Qualité**, avec aperçu des entrées récupérables et des cas nécessitant une correction manuelle. La conversion peut retirer les messages vides, normaliser les rôles et champs reconnus, décoder les tableaux de messages stockés en JSON et reconnaître des couples explicites demande/réponse. Elle est annulable et peut restaurer les entrées de la corbeille.
- **Conversion automatique pendant la fusion**, avant le dédoublonnage, activée par défaut et désactivable.
- **Provenance des conversations** affichée dans l'inspecteur, sauvegardée dans les projets `.rpwork.json`, puis restaurée à l'ouverture ou lors d'une nouvelle fusion. Les noms des fichiers et les numéros de ligne sont conservés, y compris pour les entrées écartées.

### Corrections

- La sélection des lignes est remise à zéro lorsque les identifiants affichés ou leur ordre changent, pour que la suppression et l'attribution de langue concernent les conversations choisies.
- Les pièces jointes `image`, `images`, `audio`, `video`, `attachments` et les appels d'outils directement portés par un message sont détectés par la fusion. Par défaut, ces conversions incomplètes sont écartées dans la corbeille.
- Le contexte `input` est conservé dans le message utilisateur lors de la conversion `instruction/response`. Un contexte non textuel nécessite une correction manuelle.
- Les projets mal formés sont refusés avant de remplacer le dataset ouvert : validation des entrées, identifiants, numéros de ligne, états de suppression, langues et métadonnées de provenance. Les anciens projets valides sans provenance supplémentaire restent lisibles.
- L'affichage des doublons construit leur ensemble une seule fois et calcule au maximum 100 aperçus ; la suppression continue de concerner tous les doublons.
- La consigne `system` est conservée lors de la normalisation des formats prompt/réponse.

**Validation : 67 tests réussis**, dont 12 tests de régression dédiés aux six points de la revue. Le détail des corrections et des vérifications est disponible dans [le compte rendu de revue](COMPTE_RENDU_REVIEW.md).

**Version de référence 0.01a :** état du dépôt GitHub avant cette mise à jour, au [commit `f49cc23`](https://github.com/kithain/RP-Dataset-Studio/commit/f49cc23df0ef9949f54336d593e86d695fa0849b). Il comprend l'import, les réparations JSONL simples, l'exploration et l'édition des conversations, les filtres, les langues, la corbeille, les exports et le découpage ZIP. Cette référence sert de point de départ à la numérotation des versions.

## Installation Windows

1. Installer [Python 3.11 ou 3.12](https://www.python.org/downloads/windows/) si nécessaire (cocher « Add Python to PATH »).
2. Dézipper ce dossier à un emplacement simple, par exemple `C:\RP-Dataset-Studio`.
3. Double-cliquer sur **`install.bat`**. Cette opération télécharge Streamlit, pandas, pyarrow et Lingua depuis PyPI ; une connexion Internet est nécessaire une seule fois.
4. Double-cliquer sur **`lancer.bat`**. Le navigateur ouvre une interface locale.
5. Laisser la console ouverte pendant l'utilisation. Pour fermer, presser `Ctrl+C` dans cette console.

Si Windows bloque l'exécution d'un fichier `.bat`, ouvrir un terminal dans le dossier et exécuter son contenu après vérification.

## Utilisation

- **Importer** un fichier `.jsonl`, `.json`, `.csv`, `.tsv`, `.parquet` ou `.rpwork.json` depuis le panneau latéral.
- **Fusionner** : ouvrir « 🧩 Fusionner », sélectionner plusieurs fichiers ou coller leurs chemins locaux (un par ligne), puis analyser le bilan. Par défaut, les doublons sont retirés entre tous les fichiers, les entrées incompatibles ou les conversions qui omettraient des messages/images/outils sont écartées et les conversations sont mélangées avec une graine reproductible. Le bouton de fusion ouvre le dataset dans l'éditeur et prépare un JSONL Unsloth dans Exporter. Les formats sources, métadonnées et entrées écartées restent dans le projet ; sauvegarder un `.rpwork.json` pour les conserver. Les fichiers sources ne sont pas modifiés. Le bilan de fusion est téléchargeable séparément.
- **Réparer à l'import** : la case « Réparer les erreurs JSONL simples », activée par défaut, sépare les objets JSON collés sur une ligne et retire les virgules superflues hors des chaînes de texte. Le bilan précise les lignes réparées ; les erreurs restantes sont signalées. Décoche la case avant de réimporter pour utiliser le mode strict. Exporte le JSONL au format original pour conserver une copie corrigée, sans écraser la source. Les contenus tronqués ou ambigus ne sont pas devinés.
- **Explorer** : choisir des langues, inclure/exclure des mots-clés (séparer les expressions avec `;`), utiliser éventuellement des regex ; lire les conversations en entier dans « Lecture » et les éditer dans « Modifier les messages », sans manipuler du JSON ; cocher les lignes d'une page pour les supprimer ou leur attribuer une langue. Enregistrer avant de changer d'entrée ou de vue. « JSON avancé » reste disponible pour les rôles, métadonnées et formats spécifiques. L'édition des textes conserve le format source et les autres champs, y compris les blocs non textuels.
- **Supprimer par lot** : sélectionner tous les résultats filtrés, y compris sur les autres pages, puis confirmer. Les données passent dans la corbeille et restent récupérables.
- **Qualité** : détecter les langues (localement), inspecter les doublons, évaluer la conversion ChatML, supprimer les doublons.
- **Convertir les incompatibles** : dans Qualité, analyser les entrées incompatibles puis appliquer les conversions récupérables. Les messages vides sont retirés, les rôles et champs connus normalisés, les tableaux de messages stockés comme JSON décodés et certains couples explicites demande/réponse reconnus. Les textes, métadonnées et blocs textuels sont conservés ; aucune réponse ni rôle inconnu n'est deviné. Les images et appels d'outils nécessitent une correction manuelle pour un export texte. L'opération est annulable. Cocher « Inclure les entrées de la corbeille » permet aussi de convertir et restaurer celles écartées lors d'une fusion. La fusion applique ces conversions automatiquement par défaut, avant le dédoublonnage.
- **Exporter** : générer un JSONL conservant le format original, un JSONL normalisé `messages` pour Unsloth, ou un fichier `.rpwork.json` pour reprendre le travail plus tard.
- **Découper** : dans Exporter → « Découper en plusieurs fichiers », choisir le nombre d'entrées par fichier (2 000 par défaut) et le format original ou ChatML. Préparer puis télécharger le ZIP contenant les fichiers numérotés. L'ordre et les conversations complètes sont conservés ; le dernier fichier contient le reste. La corbeille est exclue et le format ChatML exclut également les entrées incompatibles.

### Formats compris

- Conversations `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`
- ShareGPT `{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}`
- Alpaca `{"instruction":"...","input":"...","output":"..."}`
- Prompt / completion ou question / answer

La conversion de Qualité et la fusion automatique reconnaissent aussi les couples `user/assistant`, `human/gpt`, `instruction/response` (avec contexte `input` textuel facultatif) et `question/response`, ainsi que les tableaux de messages sous `conversation`, `dialogue` ou `chat`.

L'export Unsloth conserve exclusivement les conversations ayant au moins un message `user` et un message `assistant`, dans le format `{"messages":[...]}`. Les entrées incompatibles sont exclues de **cet export uniquement**.

## Aperçus des nouvelles fonctions

### Préparation de la fusion

![Options de fusion et bilan des conversations retenues](fusion_datasets.jpg)

### Export Unsloth après fusion

![Export JSONL Unsloth et bilan de la fusion](fusion_export_unsloth.jpg)

### Conversion des entrées incompatibles

![Analyse des entrées récupérables et des corrections manuelles](conversion_incompatibles.jpg)

## Notes importantes

- **Les fichiers d'origine ne sont jamais écrasés.** Tout changement reste en mémoire jusqu'au téléchargement d'un export ou d'un projet.
- **Sauvegarder régulièrement** un fichier `.rpwork.json`. Un rafraîchissement de la page ou une fermeture de l'application peut perdre les modifications non exportées.
- Une correction manuelle de langue est sauvegardée dans `.rpwork.json` et ajoutée sous `language` dans l'export JSONL original ; elle n'est pas incluse dans l'export ChatML.
- Les projets `.rpwork.json` conservent aussi les noms des fichiers sources et les numéros de ligne des conversations fusionnées, y compris celles de la corbeille. Les anciens projets restent lisibles, sans provenance supplémentaire lorsqu'elle n'a pas été sauvegardée. Un projet mal formé est refusé avant de remplacer le dataset ouvert.
- La langue détectée est une estimation, particulièrement fragile pour les dialogues très courts, les univers avec noms inventés ou les conversations multilingues.
- Ce logiciel ne classe pas automatiquement les scènes selon leur degré de violence ou d'intimité ; le filtrage par mots-clés est littéral (ou regex).
- L'export ChatML normalise le contenu textuel des messages et peut omettre des champs spécifiques à certains jeux de données (images, outils, métadonnées). Tester une portion du dataset avant un entraînement.
- Les CSV/TSV doivent être encodés en UTF-8. Pour un CSV avec un champ `messages`, ce champ doit contenir un tableau JSON valide.
- Pour des fichiers gigantesques, les imports peuvent consommer beaucoup de RAM : le programme conserve les conversations en mémoire pour permettre l'édition.
- Aucun upload distant : **toutes les données restent sur ton PC**. Le navigateur communique avec un serveur Streamlit local, limité à `127.0.0.1`.

## Test technique (facultatif)

Dans un terminal ouvert dans le dossier :

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Instructions du projet

Les consignes de travail sont enregistrées dans [AGENTS.md](AGENTS.md) : versionnement alpha, mise à jour du README, vérifications, commit et push sur GitHub. Les nouvelles fonctionnalités font progresser la partie numérique (`0.02a` → `0.03a`), les corrections seules la lettre (`0.02a` → `0.02b`). Les changements purement documentaires conservent la version de l'application.

## Fichiers

- `app.py` : interface Streamlit.
- `core.py` : import, conversion, filtres, doublons, export.
- `merge_datasets.py` et `merge_ui.py` : analyse et interface de fusion de plusieurs sources.
- `conversation_conversion.py` et `conversion_ui.py` : analyse, aperçu et conversion des conversations récupérables.
- `split_export.py` : découpage des exports JSONL dans une archive ZIP.
- `examples/exemple_rp.jsonl` : six conversations de démonstration, dont une en anglais et un doublon.
- `examples/exemple_conversion.jsonl` : exemples de conversations récupérables et de cas à corriger manuellement.
- `install.bat` et `lancer.bat` : installation et lancement Windows.
- `tests/` : tests unitaires et scénarios d'interface Streamlit.
- `COMPTE_RENDU_REVIEW.md` : revue de code, corrections et vérifications.
- `AGENTS.md` : instructions de contribution, de versionnement et de publication.
