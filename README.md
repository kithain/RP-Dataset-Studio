# RP Dataset Studio

**Un atelier local pour préparer tes datasets de narration, de dialogues et de jeu de rôle.**

RP Dataset Studio permet d’explorer, corriger, organiser et transformer des textes en données structurées pour le fine-tuning d’un modèle de langage.

Le projet privilégie la liberté d’écriture : des personnages nuancés, des dialogues naturels, des univers cohérents et des histoires qui peuvent aborder des émotions fortes, des contradictions ou des sujets difficiles. Tu choisis les sources, les thèmes et la direction éditoriale de ton dataset.

**Version actuelle : 0.10b — alpha — 24 octobre 2026.**

[Voir le projet sur GitHub](https://github.com/kithain/RP-Dataset-Studio)

## L’esprit du projet

La préparation d’un dataset doit améliorer sa structure et sa qualité technique tout en préservant la voix des textes.

Dans le préparateur de fine-tuning, aucun thème ou genre ne provoque une exclusion automatique. Les doublons sont écartés et les erreurs de structure sont signalées. Les autres contrôles fournissent des avertissements que tu peux examiner, corriger ou accepter.

Tu gardes la décision sur les données à conserver et à exporter.

## Ce que tu peux faire

- **Explorer et modifier** des conversations dans une interface lisible.
- **Importer et fusionner** plusieurs datasets.
- **Repérer les doublons**, les erreurs de format et les problèmes de conversion.
- **Organiser les entrées** avec des tags, des langues et des filtres.
- **Créer des exemples depuis un livre TXT**, en conservant les passages sources.
- **Préparer un dataset de narration française** avec l’aide d’un modèle local.
- **Comparer les propositions aux textes d’origine** et les valider manuellement.
- **Exporter les données** pour l’entraînement ou sauvegarder un projet modifiable.

L’application prépare les données. L’entraînement du modèle cible se réalise ensuite dans un environnement adapté, comme Unsloth.

## Installation sur Windows

1. Récupère le projet et décompresse le dossier dans un emplacement accessible en écriture.
2. Lance **`install.bat`**.
3. Attends la confirmation de fin d’installation.
4. Lance **`lancer.bat`** pour ouvrir l’application.

L’installateur prépare Python portable, les dépendances, un modèle Qwen 2B au format GGUF et le moteur llama.cpp. Une connexion Internet est nécessaire pour cette première installation.

L’interface s’ouvre à l’adresse :

```text
http://127.0.0.1:8501
```

Laisse la console ouverte pendant l’utilisation. Pour arrêter l’application, utilise `Ctrl+C` dans cette console.

Le traitement peut ensuite fonctionner hors ligne. L’option `install.bat -CpuOnly` permet de choisir le moteur CPU.

**Avant de relancer `lancer.bat`, sauvegarde ton projet : le lanceur redémarre la session de l’application.**

## Commencer avec un dataset existant

Depuis la barre latérale, importe un fichier ou indique son chemin local. Tu peux aussi charger le dataset de démonstration.

Les principales pages permettent de suivre ce parcours :

1. **Explorer** pour lire les entrées et corriger leurs messages.
2. **Qualité** pour examiner les langues, les doublons et les formats.
3. **Tags** pour organiser les conversations et retrouver des sous-ensembles.
4. **Exporter** pour enregistrer le résultat ou sauvegarder le travail.

Les filtres peuvent combiner du texte, des langues et des tags. Tu peux exporter uniquement les résultats filtrés sans retirer les autres entrées du dataset ouvert.

Les suppressions ordinaires passent par une corbeille. Sa purge définitive demande une confirmation.

### Réparer et fusionner

L’import peut réparer certaines erreurs JSONL simples, comme des objets collés sur une même ligne ou des virgules superflues. Les réparations sont indiquées dans le bilan ; les contenus incomplets ou ambigus ne sont pas devinés.

La page **🧩 Fusionner** rassemble plusieurs fichiers, analyse leur compatibilité et permet de retirer les doublons. Le projet conserve la provenance et les entrées mises de côté afin de pouvoir les examiner.

## Préparer un dataset de narration française

La page **🛠️ Fine-tuning FR** transforme des textes bruts ou des paires consigne/réponse en nouveaux exemples structurés.

Le modèle local propose une consigne française, puis vérifie sa cohérence avec le passage. La réponse destinée à l’entraînement conserve le texte source, avec le nettoyage de forme choisi.

### Parcours conseillé

1. Ouvre ton dataset.
2. Renseigne le modèle GGUF et le moteur dans **Modèle GGUF et mémoire**.
3. Clique sur **Préparer / reprendre le dataset**.
4. Lance **Générer un aperçu (3 exemples)**.
5. Compare les consignes proposées aux réponses sources.
6. Continue avec **Traiter tout / reprendre** lorsque le résultat te convient.
7. Prépare et télécharge les exports.

**La préparation seule ne charge pas le modèle et ne lance aucune génération.**

Pour éviter également une analyse à l’import, décoche auparavant **Ajouter les tags automatiquement à l’import**.

### Des contrôles que tu peux examiner

Les doublons sont écartés automatiquement. Les textes vides et les structures non prises en charge restent à corriger, avec leur motif.

Les autres observations — longueur inhabituelle, langue, forme ou cohérence de la consigne — sont présentées comme des avertissements. Tu peux corriger puis valider un exemple manuellement.

L’option **Inclure les propositions et les avertissements non relus** permet de les exporter sans validation individuelle. Le rapport conserve leurs motifs pour permettre une relecture ultérieure.

### Pause et reprise

Une passe peut être mise en pause après l’exemple en cours. Les résultats terminés sont sauvegardés localement.

Après fermeture, recharge le même dataset avec les mêmes réglages, puis clique sur **Préparer / reprendre le dataset**.

Un changement de données, de modèle ou de paramètres peut créer une progression distincte. Les anciennes sauvegardes restent conservées.

## Créer des exemples depuis un livre TXT

La page **📖 Depuis un livre** propose deux types de préparation :

- **RP et personnages** : sélectionner des échanges dans les dialogues originaux.
- **Narration et style** : construire une consigne dont la réponse reprend un passage du texte.

Le découpage privilégie les paragraphes et les fins de phrases. Un aperçu permet d’examiner le résultat avant de poursuivre.

Chaque exemple conserve des informations de provenance pour retrouver son passage d’origine. Les consignes et les attributions de voix restent à relire.

Les exemples peuvent être téléchargés ou ouverts dans l’éditeur pour correction. Le fichier TXT d’origine reste intact.

## Choisir un modèle local et gérer la mémoire

L’installation fournit un modèle Qwen 2B pour commencer. Les chemins restent modifiables dans l’interface.

Le préparateur de narration dispose de réglages propres pour utiliser un autre GGUF compatible avec le moteur local. Un moteur FP8 utilisant une installation ComfyUI compatible est également disponible pour les fonctions qui le prennent en charge.

Pour un modèle volumineux, tu peux régler :

- le nombre de couches placées sur le GPU ;
- la taille du contexte ;
- le batch et le microbatch ;
- le cache des préfixes ;
- les réserves de RAM et de VRAM libres.

Un seul modèle est conservé à la fois dans le moteur partagé. Le bouton de déchargement permet de libérer ses ressources.

La surveillance mémoire peut arrêter le processus de modèle appartenant à l’application lorsque la réserve choisie est atteinte. Ce contrôle périodique ne constitue pas un plafond strict. La mesure de VRAM nécessite `nvidia-smi`.

Le comportement des réponses dépend du modèle choisi. Commence par un aperçu avant de lancer une longue passe.

## Organiser les données avec les tags

Les tags permettent de retrouver des genres, des thèmes, des personnages, des ambiances ou des styles de narration.

Tu peux :

- demander des suggestions au modèle local ou utiliser les règles rapides ;
- modifier les tags d’une entrée ;
- renommer, fusionner ou retirer des tags par lot ;
- combiner plusieurs critères avec les modes **ET** ou **OU** ;
- exporter un sous-ensemble correspondant à tes critères.

Les suggestions automatiques restent modifiables. Les annotations servent à organiser le corpus ; les sélections et les corrections restent sous ton contrôle.

## Formats pris en charge

| Usage | Formats |
| --- | --- |
| Import de datasets | JSONL, JSON, CSV, TSV, Parquet |
| Sauvegarde du travail | Projet `.rpwork.json` |
| Import de livre | TXT |
| Structures conversationnelles | Messages, ShareGPT, Alpaca, prompt/réponse |
| Préparateur de narration | Texte brut, paire utilisateur/assistant, système initial facultatif |

Les conversations compatibles peuvent être exportées sous la forme :

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Écris en français."
    },
    {
      "role": "user",
      "content": "Décris une gare à l’aube."
    },
    {
      "role": "assistant",
      "content": "La lumière du matin glissait entre les verrières de la gare."
    }
  ]
}
```

Chaque ligne d’un fichier JSONL contient un objet JSON complet.

Le préparateur de narration traite une paire utilisateur/assistant, avec système initial facultatif. Les conversations à plusieurs tours et les contenus nécessitant une conversion particulière restent à examiner.

## Exports et sauvegardes

L’éditeur propose :

- un **JSONL au format original**, avec les annotations prévues par l’export ;
- un **JSONL conversationnel `messages`** pour les outils d’entraînement compatibles ;
- un **projet `.rpwork.json`** pour retrouver les corrections, les tags et la provenance ;
- des **archives découpées** pour répartir un export en plusieurs fichiers.

Le préparateur de narration fournit aussi une archive contenant le dataset, les partitions d’entraînement et de validation, les éléments à revoir, les exclusions et le bilan.

La séparation entraînement/validation utilise des groupes de passages et une marge entre voisins. Sans provenance détaillée, elle repose sur l’ordre des lignes : une séparation par livre ou chapitre reste à vérifier selon le corpus.

Un export peut être partiel. Consulte son bilan avant de l’utiliser pour un entraînement.

## Données locales et conservation du travail

Les fichiers sources ne sont pas écrasés. Les modifications de l’éditeur restent en mémoire jusqu’à l’export ; les générateurs disposent de leurs propres sauvegardes de progression.

**Sauvegarde régulièrement un projet `.rpwork.json`**, notamment avant de fermer ou de redémarrer l’application.

Le traitement et l’inférence s’effectuent sur ton ordinateur. Les datasets personnels, les exports, les configurations locales et les environnements d’exécution sont exclus du suivi Git.

Pour déplacer l’installation, ferme l’application et copie le dossier complet, y compris les modèles, les moteurs et les configurations. Les chemins externes personnalisés devront éventuellement être ajustés.

## Vérifications et limites

Les contrôles automatiques facilitent la préparation, mais la relecture reste utile : un modèle peut proposer une consigne inexacte, attribuer une mauvaise voix ou suggérer un tag discutable.

Les exports conversationnels textuels peuvent omettre des champs propres à certains formats, notamment les images ou les appels d’outils. Les incompatibilités doivent être examinées avant conversion.

Les gros imports restent en mémoire et peuvent consommer une quantité importante de RAM.

La version 0.10b remplace l’ancien paramètre de largeur de Streamlit par son API actuelle, en conservant la largeur des boutons et des tableaux. Les messages répétés dans la console lors de l’actualisation des motifs d’exclusion sont ainsi supprimés. L’application demande désormais Streamlit 1.49 ou plus récent.

**Dernière validation de la version 0.10b : 228 tests automatisés réussis.** Un test vérifie l’absence de ces avertissements sur les boutons et les tableaux d’exclusion après plusieurs affichages. Les tests ciblés d’interface et de rechargement passent aussi avec le Python portable. Le rechargement des modules conserve le dataset et les ressources Qwen déjà ouverts. Les tests courants simulent les appels au modèle ; les essais réels sont documentés séparément dans le dépôt.

Pour lancer la suite avec le Python portable :

```powershell
.runtimes\python\python.exe -m unittest discover -s tests -v
```

## Historique résumé

| Version | Évolution principale |
| --- | --- |
| **0.10b** | Compatibilité Streamlit : suppression des avertissements répétés liés à la largeur des éléments. |
| **0.10a** | Choix éditoriaux laissés à l’utilisateur, contrôles techniques et export volontaire des avertissements. |
| **0.09c** | Motifs d’exclusion détaillés et consultables. |
| **0.09b** | Prise en charge du message système dans le préparateur. |
| **0.09a** | Préparation de narration française, pause/reprise et réglages mémoire. |
| **0.08b** | Amélioration de la reconnaissance des dialogues et des contrôles du générateur TXT. |
| **0.08a** | Création d’exemples depuis un livre TXT. |
| **0.07c** | Redémarrage plus fiable du lanceur Windows. |
| **0.07b** | Rechargement des modules après une mise à jour. |
| **0.07a** | Analyse ciblée des entrées sans tags et purge de la corbeille. |
| **0.06a** | Catalogue de tags, filtres combinés et corrections par lot. |
| **0.05a** | Installation autonome avec Python portable et modèle local. |
| **0.04a** | Analyse sémantique des tags avec Qwen. |
| **0.03a** | Tags automatiques et édition des annotations. |
| **0.02a** | Renforcement des validations et de la conservation des données. |
| **0.01a** | Import, exploration, édition, filtres et exports. |

L’historique complet du développement reste disponible dans les commits du dépôt GitHub.

Le README précédent est conservé dans [l’archive du 24 octobre 2026](README.archive-0.10a-2026-10-24.md).
