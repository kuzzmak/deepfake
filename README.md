# deepfake

Simple implementation of the deepfake creation process in Python.

## Installing steps

1. Download code.
2. Install [PyTorch and Torchvision](https://pytorch.org/). Version used here was 1.9.0.

    ```python
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    ```

3. Install python dependencies with

    ```python
    pip install -r requirements.txt
    ```

4. Install [visual studio build tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019).
5. python setup.py build_ext --inplace