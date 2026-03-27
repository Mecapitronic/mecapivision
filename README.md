# MecapiVision
Détection et localisation de tags ArUco sur le terrain avec une caméra, pour le robot holonome de l'équipe **Mecapitronic** (Coupe de France de Robotique).

## Vue d'ensemble

```
Caméra (webcam / Pi Camera)
        │
        ▼
  Calibration (chessboard ou charuco)
        │
        ▼
  Détection ArUco (DICT_6X6_250)
        │
        ▼
  Estimation de pose (solvePnP)
        │
        ▼
  Position x, y du tag dans le repère terrain (mm)
```

## Commandes disponibles (entry points)

| Commande     | Description                                               |
|--------------|-----------------------------------------------------------|
| `record`     | Enregistre des photos du chessboard pour la calibration   |
| `calibrate`  | Calibre la caméra depuis les photos enregistrées          |
| `caliblive`  | Calibre la caméra en direct (livestream)                  |
| `charuco`    | Calibre via une mire ChArUco                              |
| `detect`     | Lance la détection ArUco en live (avec estimation de pose)|
| `vision`     | Point d'entrée principal (démo)                           |


## 1. Installation

Il est fortement recommandé d'utiliser un environnement virtuel Python pour exécuter ce projet afin d'isoler les dépendances de **mecapivision**.

### 1.1 Prérequis

- Python **3.10+** (vérifier avec `python --version`)
- Webcam disponible

### 1.2 Cloner le projet

```bash
git clone https://github.com/Mecapitronic/mecapivision.git
cd mecapivision
```

### 1.3 Créer l'environnement virtuel
Ouvrez votre terminal à la racine du projet et exécutez la commande correspondant à votre système :

```bash
  python -m venv .venv
```

### 1.4 Activer l'environnement virtuel
```bash
.\.venv\Scripts\activate
```
Une fois activé, vous devriez voir le préfixe (venv) apparaître au tout début de la ligne de votre terminal.

### 1.5 Installer les dépendances
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1.6 Installer le package mecapivision
```bash
pip install -e .
```
> L'option `-e` installe le package en mode *editable* : les modifications du code sont prises en compte immédiatement, sans réinstaller.






</br></br></br></br></br></br></br></br></br></br>
## Installation on RPi
[source : https://pyimagesearch.com/2018/09/19/pip-install-opencv/](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)

Install pre-requisites
```bash
sudo apt install -y \
libatlas-base-dev \
libhdf5-103 \
libhdf5-dev \
libhdf5-serial-dev \
libjasper-dev \
libqt4-test  \
libqtgui4  \
libqtwebkit4  \
python3-pyqt5 \
```

Install pip
```bash
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```

Install opencv from PiWheel
```bash
pip install opencv
```

Install PiCamera lib
```bash
pip install "picamera[array]"
```


## Sources and Docs

* [OpenCV documentation](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)

