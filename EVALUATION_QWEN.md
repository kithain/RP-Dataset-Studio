# Évaluation locale Qwen — 22 octobre 2026

Essai de fonctionnement sur **huit exemples courts en français**, exécutés par `evaluate_qwen.py` avec la même consigne, génération sans échantillonnage et raisonnement désactivé. Ces exemples servent à contrôler quelques comportements ; ils ne mesurent pas la précision sur un dataset réel et ne garantissent pas l'absence de refus ou d'erreurs.

## Modèles et environnement

| Modèle | Fichier | Moteur |
| --- | --- | --- |
| [Qwen3.5-2B Heretic Q4_K_M](https://huggingface.co/darrellbest/Qwen3.5-2B-Heretic-GGUF) | 1 312 164 832 octets, environ 1,31 Go | llama.cpp `b11819`, Vulkan x64 |
| Qwen3.5-4B Heretic FP8 fourni localement | 5 513 517 624 octets, environ 5,51 Go | ComfyUI, PyTorch `2.14.1+cu130`, Transformers `5.19.0` |

Machine : Windows, NVIDIA GeForce RTX 5080 avec 16 Go de VRAM. Les empreintes SHA-256 du téléchargement GGUF et de l'archive llama.cpp ont été vérifiées avec les métadonnées de leurs dépôts. Aucun projecteur vision n'a été téléchargé pour la 2B.

## Résultats observés

| Exemple | Classement attendu | 2B | 4B |
| --- | --- | --- | --- |
| Romance implicite, fleurs et maladresse au café | SFW, Romance, Slice of Life | Conforme | Conforme |
| Interdiction de sexe/nudité/gore, discussion neutre autour du thé | SFW, sans thèmes sexuels ni relationnels ajoutés | SFW correct ; ajoute Romance, Teasing et Playful | Conforme |
| Thèmes adultes déclarés : BDSM, nudité et contenu sexuel explicite | NSFW, BDSM | Conforme, sans refus | Conforme, sans refus |
| Thèmes adultes déclarés : Vore et digestion | NSFW, Vore | Conforme, sans refus | Conforme, sans refus |
| Instruction parasite dans les données et enquête spatiale | SFW, Mystery, Sci-Fi | Conforme ; instruction parasite ignorée sur ce cas | Conforme ; instruction parasite ignorée sur ce cas |
| Sentiments et mains tenues, sans nommer le genre | SFW, Romance | Conforme | Conforme |
| Courses, trajet en bus et tâches du quotidien | SFW, Slice of Life, sans romance ajoutée | Conforme | Conforme |
| Horreur, gore et torture, sans contenu sexuel | NSFW, Horror, Violence | Conforme | Conforme |

- **Classification SFW/NSFW : 8/8 pour les deux modèles.**
- **Critères de tags vérifiés : 7/8 pour la 2B, 8/8 pour la 4B.** Les critères vérifient quelques tags attendus et quelques ajouts indésirables ; ils ne valident pas exhaustivement chaque tag libre généré.
- **Temps médians observés par entrée :** environ 0,12 s pour la 2B et 0,73 s pour la 4B. Plages sur ces exemples : 0,09–0,17 s et 0,58–1,91 s. Chargement : environ 1,63 s et 8,52 s, avec les fichiers déjà présents dans le cache du système après les premiers essais. Un premier démarrage ou de longues conversations peuvent prendre davantage de temps.
- Les premiers essais ont révélé un faux positif NSFW sur une romance innocente et une confusion entre absence de sexe et absence de violence. La consigne finale précise ces distinctions ; les résultats ci-dessus concernent cette consigne finale.

La **2B est proposée comme choix rapide**, avec ses limites visibles. La **4B constitue une alternative plus précise sur cet échantillon**. Les corrections manuelles restent nécessaires, particulièrement pour les sous-entendus, les négations et les thèmes ambigus.

## Reproduire

Les chemins par défaut correspondent au PC utilisé pour le développement et restent modifiables. Le test utilise les vrais poids locaux ; il est distinct de la suite automatisée sans modèle.

```powershell
.\.venv\Scripts\python.exe evaluate_qwen.py --backend gguf --model "D:\I.A\GGUF\Qwen3.5-2B-Heretic-Q4_K_M.gguf" --server ".runtimes\llama-b11819-vulkan\llama-server.exe" --output "evaluation-2b.json"
.\.venv\Scripts\python.exe evaluate_qwen.py --backend comfy --model "D:\I.A\ComfyUI_windows_portable\ComfyUI\models\text_encoders\Qwen3.5-4B-heretic-fp8.safetensors" --output "evaluation-4b.json"
```

Le script affiche chaque résultat, le temps d'analyse et le total des contrôles satisfaits. Il ferme son processus à la fin. Les rapports JSON générés sont ignorés par Git. La 4B utilise par défaut le Python et le dossier ComfyUI locaux définis dans `llm_tagging.py`.

Un import réel dans l'interface Streamlit avec la 2B a également été vérifié : une romance implicite reçoit `SFW` et `Romance`, et une fiche de thèmes sexuels adultes reçoit `NSFW`. Les 109 tests automatisés passent séparément ; leurs appels au modèle sont simulés pour ne pas exiger de GPU ni de téléchargement.
