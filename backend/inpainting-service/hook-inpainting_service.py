import pkgutil
import site

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = []
datas = []

for _, name, _ in pkgutil.iter_modules(site.getsitepackages()):
    hiddenimports += collect_submodules(name)
    datas += collect_data_files(name)
