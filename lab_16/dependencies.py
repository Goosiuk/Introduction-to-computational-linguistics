import importlib
import subprocess
import sys

def check_and_install_libraries():
    required_libraries = {
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }

    for module_name, package_name in required_libraries.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            except subprocess.CalledProcessError as e:
                sys.exit(1)