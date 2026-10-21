# RP Dataset Studio — Éditeur local de datasets RP

Un éditeur graphique **100 % local** pour préparer les datasets de jeu de rôle destinés au fine-tuning Unsloth/QLoRA.
Il fonctionne sur Windows 10/11 avec Python et s'ouvre dans un navigateur à l'adresse `http://127.0.0.1:8501`.

Le code et son historique sont disponibles sur [GitHub](https://github.com/kithain/RP-Dataset-Studio).
Le dépôt contient le programme, ses tests et un petit dataset de démonstration. Les datasets personnels, projets `.rpwork.json`, exports et environnements Python locaux sont exclus du suivi Git par `.gitignore`.

## Installation Windows

1. Installer [Python 3.11 ou 3.12](https://www.python.org/downloads/windows/) si nécessaire (cocher « Add Python to PATH »).
2. Dézipper ce dossier à un emplacement simple, par exemple `C:\RP-Dataset-Studio`.
3. Double-cliquer sur **`install.bat`**. Cette opération télécharge Streamlit, pandas, pyarrow et Lingua depuis PyPI ; une connexion Internet est nécessaire une seule fois.
4. Double-cliquer sur **`lancer.bat`**. Le navigateur ouvre une interface locale.
5. Laisser la console ouverte pendant l'utilisation. Pour fermer, presser `Ctrl+C` dans cette console.

Si Windows bloque l'exécution d'un fichier `.bat`, ouvrir un terminal dans le dossier et exécuter son contenu après vérification.

## Utilisation

- **Importer** un fichier `.jsonl`, `.json`, `.csv`, `.tsv`, `.parquet` ou `.rpwork.json` depuis le panneau latéral.
- **Réparer à l'import** : la case « Réparer les erreurs JSONL simples », activée par défaut, sépare les objets JSON collés sur une ligne et retire les virgules superflues hors des chaînes de texte. Le bilan précise les lignes réparées ; les erreurs restantes sont signalées. Décoche la case avant de réimporter pour utiliser le mode strict. Exporte le JSONL au format original pour conserver une copie corrigée, sans écraser la source. Les contenus tronqués ou ambigus ne sont pas devinés.
- **Explorer** : choisir des langues, inclure/exclure des mots-clés (séparer les expressions avec `;`), utiliser éventuellement des regex ; lire les conversations en entier dans « Lecture » et les éditer dans « Modifier les messages », sans manipuler du JSON ; cocher les lignes d'une page pour les supprimer ou leur attribuer une langue. Enregistrer avant de changer d'entrée ou de vue. « JSON avancé » reste disponible pour les rôles, métadonnées et formats spécifiques. L'édition des textes conserve le format source et les autres champs, y compris les blocs non textuels.
- **Supprimer par lot** : sélectionner tous les résultats filtrés, y compris sur les autres pages, puis confirmer. Les données passent dans la corbeille et restent récupérables.
- **Qualité** : détecter les langues (localement), inspecter les doublons, évaluer la conversion ChatML, supprimer les doublons.
- **Exporter** : générer un JSONL conservant le format original, un JSONL normalisé `messages` pour Unsloth, ou un fichier `.rpwork.json` pour reprendre le travail plus tard.
- **Découper** : dans Exporter → « Découper en plusieurs fichiers », choisir le nombre d'entrées par fichier (2 000 par défaut) et le format original ou ChatML. Préparer puis télécharger le ZIP contenant les fichiers numérotés. L'ordre et les conversations complètes sont conservés ; le dernier fichier contient le reste. La corbeille est exclue et le format ChatML exclut également les entrées incompatibles.

### Formats compris

- Conversations `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`
- ShareGPT `{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}`
- Alpaca `{"instruction":"...","input":"...","output":"..."}`
- Prompt / completion ou question / answer

L'export Unsloth conserve exclusivement les conversations ayant au moins un message `user` et un message `assistant`, dans le format `{"messages":[...]}`. Les entrées incompatibles sont exclues de **cet export uniquement**.

## Notes importantes

- **Les fichiers d'origine ne sont jamais écrasés.** Tout changement reste en mémoire jusqu'au téléchargement d'un export ou d'un projet.
- **Sauvegarder régulièrement** un fichier `.rpwork.json`. Un rafraîchissement de la page ou une fermeture de l'application peut perdre les modifications non exportées.
- Une correction manuelle de langue est sauvegardée dans `.rpwork.json` et ajoutée sous `language` dans l'export JSONL original ; elle n'est pas incluse dans l'export ChatML.
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

## Fichiers

- `app.py` : interface Streamlit.
- `core.py` : import, conversion, filtres, doublons, export.
- `examples/exemple_rp.jsonl` : six conversations de démonstration, dont une en anglais et un doublon.
- `install.bat` et `lancer.bat` : installation et lancement Windows.
- `tests/` : tests unitaires.
