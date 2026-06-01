# Template Python + CI pour Quantum ÉTS

Ce dépôt sert de **base de départ minimale** pour les projets Python du club, en particulier pour les membres débutants ou intermédiaires.

Il est pensé pour deux cas fréquents:

1. Hackathon avec sujet déterminé, mais sans code fourni: vous pouvez partir de ce template et développez votre code/tests à partir de zéro.
2. Hackathon avec dépôt de départ fourni: vous réutilisez la structure CI/tests de ce template et ajoutez les documents, codes et dépendances spécifiques au projet.

>[!NOTE]
> Dans le cas d'un hackathon avec dépôt de départ fourni, il vaudra peut-être mieux utiliser un autre template agnostique des outils de développement (pytest, pre-commit, etc.) pour éviter les conflits de configuration. Par exemple, un template avec des instructions pour des agents sera peut-être plus adapté.

Ce template est conçu pour être utilisé comme **template GitHub d'organisation** afin de créer rapidement de nouveaux dépôts cohérents au sein du club.

## Ce que contient ce template

- `src/`: code source Python, où se trouve le code d'application.
- `tests/`: tests unitaires, suivant la convention ["src layout"](https://docs.pytest.org/en/stable/explanation/goodpractices.html#tests-outside-application-code).
- `pyproject.toml`: configuration du projet, des dépendances et des groupes de développement.
- `uv.lock`: verrouillage des versions pour des installations reproductibles avec `uv`.
- `.github/workflows/unit-tests.yml`: exécution automatique des tests sur GitHub Actions.
- `.pre-commit-config.yaml`: configuration des hooks `pre-commit` (Ruff).

L'exemple de code est volontairement minimal (`sum`) pour rester lisible et servir de squelette.

## Outils et standard d'équipe

- **GitHub Actions** (`.github/workflows/unit-tests.yml`): exécute automatiquement les tests à chaque push/PR pour garantir que tout le monde valide le même niveau de qualité avant fusion.
- **pytest** (groupe `dev` dans `pyproject.toml`): framework de tests unitaires, utilisé pour vérifier le comportement du code de manière reproductible entre les membres.
- **pre-commit** (`.pre-commit-config.yaml`): au moment de commit, l'outil `pre-commit` exécute les hooks configurés, pour garantir que le code respecte les standards de qualité avant même d'être poussé.
- **Ruff** (`pre-commit` + groupe `dev` dans `pyproject.toml`): linter et formateur de code Python qui remplace le duo historique Black/isort dans ce template, pour garder un code lisible et uniforme.
- **Séparation des dépendances** (`pyproject.toml` + `uv.lock`): distingue les dépendances nécessaires pour exécuter le projet de celles réservées au développement, tout en gardant un verrouillage reproductible des versions.

## Première configuration

> [!IMPORTANT]
> Ce template est prévu pour `uv`. Utilisez `uv sync` pour installer les dépendances et `uv run` pour exécuter les commandes du projet.

### Configuration avec [`uv`](https://docs.astral.sh/uv/)

<details>
<summary>Instructions pour uv</summary>

#### 1) Installer les dépendances

```bash
uv sync --group dev
```

#### 2) Exécuter les tests

```bash
uv run pytest tests/
```

#### 3) Lancer les hooks qualité

```bash
uv tool install pre-commit
uvx pre-commit install
uvx pre-commit run --all-files
```

#### 4) Lancer le script d'exemple

```bash
uv run python src/main.py
```

> [!TIP]
> Si vous préférez ne pas installer `pre-commit` globalement, vous pouvez remplacer `uvx pre-commit ...` par une commande équivalente dans un environnement où `pre-commit` est déjà installé.

</details>

## Usage courant

- **Développement**: travaillez dans `src/` pour le code d'application, et dans `tests/` pour les tests unitaires.
- **Tests**: lancez les tests avec pytest via `uv run pytest tests/` pour vérifier que votre code fonctionne comme prévu.
- **Qualité**: les hooks `pre-commit` s'exécutent automatiquement au moment du commit, avec Ruff comme formateur et linter.
- **CI**: à chaque push sur `main` ou `dev`, et à chaque Pull Request, les tests sont exécutés automatiquement sur GitHub Actions pour garantir que le code fusionné passe les tests.
- **Documentation**: ajoutez des commentaires et des docstrings (Google Style) pour expliquer le fonctionnement de votre code, surtout si vous travaillez en équipe.
- **Type-hints**: utilisez des annotations de type pour faciliter la lecture par les pairs et l'IA si vous en utilisez.

**Puis-je utiliser uv ?**

Oui. Ce template est configuré pour `uv` et s'appuie sur `pyproject.toml` ainsi que `uv.lock` pour gérer les dépendances et reproduire les installations.

## Ressources supplémentaires

### Outils de développement

- [Documentation pytest](https://docs.pytest.org/en/stable/)
- [Documentation pre-commit](https://pre-commit.com/)
- [Documentation Ruff](https://docs.astral.sh/ruff/)
- [Documentation pre-commit Ruff hooks](https://github.com/astral-sh/ruff-pre-commit)
- [PEP 8 - Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

### Gestion d'environnement

- [uv - Documentation](https://docs.astral.sh/uv/)
