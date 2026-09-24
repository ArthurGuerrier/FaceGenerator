# 🎭 Face Generator (DDPM)

Un générateur de visages basé sur l'architecture **DDPM** (Denoising Diffusion Probabilistic Models) implémenté from scratch avec PyTorch. Ce projet permet de générer des visages réalistes à partir de bruit aléatoire en utilisant un modèle U-Net pré-entraîné.

## ✨ Fonctionnalités

- 🧠 **Architecture U-Net personnalisée** avec blocs résiduels et attention
- ⚡ **Génération optimisée** : Utilisation des poids EMA pour une meilleure qualité
- 💾 **Léger et rapide** : Pas de dépendances lourdes comme matplotlib
- 🖼️ **Visualisation native** : Ouverture automatique des images sous Windows
- 🚀 **Simple d'utilisation** : Un seul script pour tout lancer

## 📋 Prérequis

- Python 3.10+
- CUDA compatible GPU (recommandé pour la vitesse de génération)
- Un fichier checkpoint `.pth` issu de l'entraînement


## 🛠️ Installation

1. Cloner le projet

```sh
git clone 
cd FaceGenerator
```

2. Créer un environnement virtuel

```sh
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Installer les dépendances
Nous utilisons un miroir rapide pour PyTorch et installons les bibliothèques nécessaires en une fois.

```sh
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install diffusers accelerate transformers tqdm pillow
```


## Génération d'images

Assure-toi d'avoir placé ton fichier de checkpoint (ex: checkpoint.pth) à la racine du projet ou d'avoir mis à jour le chemin dans generate.py.

```sh
python generate.py
```

Options de configuration (generate.py) :
num_images: Nombre d'images à générer (défaut: 1).
checkpoint_path: Chemin vers ton fichier .pth.
seed: Pour la reproductibilité (optionnel).

🖼️ Sous Windows, la première image générée s'ouvrira automatiquement dans l'application Photos.

📂 Structure du projet

FaceGenerator/
├── generate.py       # Script de génération d'images
├── README.md         # Documentation
├── venv/             # Environnement virtuel
└── generated/        # Dossier de sortie des images
    ├── faces_img1.png
    └── ...


## ⚙️ Configuration

Les paramètres principaux sont centralisés dans le dictionnaire CONFIG au début du script :

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `image_size` | Résolution des images | 64x64 |
| `num_timesteps` | Étapes de diffusion | 1000 |
| `time_embed_dim` | Dimension de l'embedding temporel | 256 |


## 🤝 Contribution
Les contributions sont les bienvenues ! N'hésite pas à ouvrir une issue ou une pull request si tu souhaites améliorer l'architecture ou ajouter des fonctionnalités.

## 📄 Licence
Ce projet est open-source. Voir le fichier LICENSE pour plus de détails.