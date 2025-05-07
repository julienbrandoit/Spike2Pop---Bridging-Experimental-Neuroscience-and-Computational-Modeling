# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['spike2pop.py'],
    pathex=[],
    binaries=[],
    datas = [
    ('logo.png', '.'),  # Bundling the logo
    ('config.json', '.'),  # Bundling the config file
    ('models', 'models'),  # Bundling the models folder
    ('script', 'script'),  # Bundling the script folder
    ],
    hiddenimports=['PIL._tkinter_finder'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='spike2pop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='spike2pop',
)
