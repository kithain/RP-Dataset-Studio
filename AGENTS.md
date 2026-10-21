# Instructions du projet — RP Dataset Studio

Ces consignes s'appliquent à tout le dépôt. Les demandes explicites de l'utilisateur priment.

## Publication des modifications

Par défaut, terminer chaque tâche de modification par la documentation, la validation, un commit et un push sur `origin/main`, sans redemander une autorisation déjà donnée. Cette règle ne s'applique pas à une simple question, un audit en lecture seule, un travail inachevé ou une demande explicite de rester en local. Respecter une autre branche ou un autre dépôt si l'utilisateur le demande.

1. Vérifier `git status`, la branche, le dépôt distant et les modifications locales avant de travailler. Récupérer les références distantes avec `git fetch origin --tags` avant de préparer la publication. Conserver les changements préexistants et inclure uniquement les fichiers relevant de la tâche ; ne pas écraser le travail de l'utilisateur.
2. Déterminer la version à partir du `README.md` publié sur GitHub, de l'historique et des éventuels tags ou releases. Ne pas se fier uniquement au numéro affiché dans l'interface. La référence historique est **0.01a**, correspondant au commit `f49cc23df0ef9949f54336d593e86d695fa0849b` ; la première mise à jour selon cette convention est **0.02a**.
3. Appliquer la convention de version ci-dessous, puis mettre à jour le README et le numéro affiché dans l'application dans le même commit.
4. Exécuter les vérifications adaptées, contrôler les fichiers à publier et créer un commit explicite en français. Pour une version applicative, inclure le numéro dans le sujet, par exemple `feat: publier la version 0.03a avec ...` ou `fix: publier la version 0.02b avec ...`.
5. Pousser sur `origin/main` sans forcer. Si la branche distante a avancé, intégrer ses changements en préservant le travail local et résoudre les conflits avant de pousser. Respecter les protections de branche ; si elles imposent une pull request, utiliser ce parcours.
6. Vérifier après le push que le SHA de la branche distante correspond au commit local, et vérifier l'état du répertoire de travail. Ne déclarer le push réussi qu'après confirmation. En cas d'échec, conserver le commit local et expliquer précisément le blocage.
7. Terminer par une réponse courte en français indiquant la version, le résultat des vérifications, la branche, le lien du commit GitHub et les éventuels changements locaux restants.

## Convention de version

- Conserver le format alpha utilisé par ce projet : `0.01a`, `0.02a`, etc., sauf demande de l'utilisateur de changer de convention.
- **Nouvelle fonctionnalité ou ensemble fonctionnel :** augmenter la partie numérique et repartir à `a`, par exemple `0.02a` → `0.03a`.
- **Corrections seules :** avancer la lettre de révision, par exemple `0.02a` → `0.02b`. Après `z`, augmenter la partie numérique et repartir à `a`.
- **Documentation ou consignes uniquement :** conserver la version de l'application ; faire un commit `docs:` et le pousser selon le même parcours.
- Ne pas réutiliser ni diminuer une version déjà publiée. Une tâche regroupant fonctionnalités et corrections suit l'incrément des fonctionnalités. Ne pas modifier le marqueur de format des projets `.rpwork.json` pour un simple changement de version applicative.
- Ne créer ni tag Git ni release GitHub automatiquement : le modèle actuel est une publication par commit sur `main`, avec la version documentée dans le README et affichée dans `app.py`.

## README et vérifications

- Pour chaque version applicative, mettre à jour la version actuelle, la date, les nouveautés et corrections, l'utilisation et les formats concernés, ainsi que le résultat des tests. Conserver l'historique des versions précédentes, en plaçant la plus récente en premier. Ajouter ou actualiser les captures lorsqu'elles aident à comprendre une nouvelle fonction.
- Aligner le numéro du README avec celui de la barre latérale dans `app.py`. Documenter uniquement des comportements implémentés et des vérifications réellement exécutées.
- Pour une modification du code, lancer depuis la racine du dépôt :

  ```powershell
  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
  ```

- Ajouter des tests de régression pour les bogues corrigés quand ils protègent un comportement utile. Après réussite, ne relancer la suite que si de nouveaux changements ou un doute concret le justifient.
- Pour une modification documentaire seule, vérifier le contenu, les liens locaux et les différences ; la suite applicative n'est pas nécessaire.
- Avant le commit, sélectionner explicitement les fichiers concernés et exécuter `git diff --cached --check`. Consulter les différences préparées et ne publier aucun dataset personnel, export, secret ou environnement local ; respecter `.gitignore`.
