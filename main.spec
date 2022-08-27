# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None




a = Analysis(
    ['main.py'],
    pathex=[str(Path('.').absolute())],
    binaries=[],
    datas=[('config.json', '.'), ('README.md', '.'), ('logging_config.yaml', '.')],
    hiddenimports=['sklearn.utils._typedefs', 'logger'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

MISSING_LIBS = (
    # torch for cpu
    Path('').absolute() / 'env' / 'Lib' / 'site-packages' / 'torch' / 'lib' / 'torch_cpu.dll',
)

a.binaries += TOC([(lib.name, str(lib.resolve()),'BINARY') for lib in MISSING_LIBS])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
