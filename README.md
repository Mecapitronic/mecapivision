# mecapivision
detection et analyse des éléments du terrain avec une camera

## Quick start
ouvrez dans le devcontainer avec vscode pour que ça soit plus simple.

```python
uv run main.py
```

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

