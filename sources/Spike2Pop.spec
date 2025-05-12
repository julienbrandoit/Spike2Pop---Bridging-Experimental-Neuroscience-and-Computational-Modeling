# Spike2Pop.spec

block_cipher = None

a = Analysis(
    ['spike2pop.py'],  # Replace with your main entry script
    pathex=['.'],
    binaries=[],
    datas = [
    ('logo.png', '.'),  # Bundling the logo
    ('config.json', '.'),  # Bundling the config file
    ('models', 'models'),  # Bundling the models folder
    ('script', 'script'),  # Bundling the script folder
    ],
    hiddenimports=[
        'torch.nn.utils.rnn',  # Dynamic imports like padding
        'PIL._tkinter_finder'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'scipy.fftpack',  # Unused parts of SciPy
        'scipy.linalg',    # Unused parts of SciPy
        'scipy.sparse',    # Unused parts of SciPy
        'scipy.optimize',  # If not used directly
        'scipy.stats',     # If not used directly
        'PyQt5',           # May be pulled in by matplotlib
        'PySide2',         # Same as above
        'matplotlib.backends.backend_pdf',  # Unused backends
        'matplotlib.backends.backend_ps',   # Unused backends
        'matplotlib.tests',  # Testing modules
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Spike2Pop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Enable UPX if installed
    console=False,  # False if you have a GUI
    icon='logo.ico'  # Replace with your app's icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='Spike2Pop'
)
