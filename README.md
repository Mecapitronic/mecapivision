# mecapivision
detection et analyse des éléments du terrain avec une camera

## Quick start
ouvrez dans le devcontainer avec vscode pour que ça soit plus simple.

```python
uv run main.py
```

## 1. Installation

Il est fortement recommandé d'utiliser un environnement virtuel Python pour exécuter ce projet afin d'isoler les dépendances de **mecapivision**.

### 1.1 Créer l'environnement virtuel
Ouvrez votre terminal à la racine du projet et exécutez la commande correspondant à votre système :

```bash
  python -m venv venv
```

### 1.2 Activer l'environnement virtuel
```bash
.\.venv\Scripts\activate
```
Une fois activé, vous devriez voir le préfixe (venv) apparaître au tout début de la ligne de votre terminal.

### 1.3 Installer les dépendances
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1.4 Installer le package mecapivision
```bash
pip install .
```

### 1.5 Utiliser le package mecapivision

## Installation on RPi
[source](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)

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

