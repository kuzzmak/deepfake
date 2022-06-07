# deepfake

Simple implementation of the deepfake creation process in Python.

## Installing steps

1. Download code.

2. Make a virtual environment.

    ```python
    python -m venv env
    ```

3. Activate this virtual environment.

    ```bash
    env\Scripts\activate (Windows) 
    ```

4. Install [PyTorch and Torchvision](https://pytorch.org/). Version used here was 1.9.0.

    ```python
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    ```

5. Install python dependencies with

    ```python
    pip install -r requirements.txt
    ```

6. Install [visual studio build tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019).

7. python setup.py build_ext --inplace


## Resolving errors

1. DirectShowPlayerService::doRender: Unresolved error code 0x80040266 (IDispatch error #102)
 
    If you get this error, this means that codecs for video playback are missing, for Windows you can install them [here](https://www.codecguide.com/download_kl.htm).
