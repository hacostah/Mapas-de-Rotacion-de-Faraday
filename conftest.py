import os
import sys

# Los tests usan el perfil "rapido" (malla 64³): el exhaustivo, el de por
# defecto, es para las corridas de producción en GPU (ver config_fisica.py).
os.environ.setdefault("FARADAYMR_PERFIL_RESOLUCION", "rapido")

_RAIZ_REPO = os.path.dirname(os.path.abspath(__file__))
if _RAIZ_REPO not in sys.path:
    sys.path.insert(0, _RAIZ_REPO)
