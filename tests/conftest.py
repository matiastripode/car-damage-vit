import sys
from pathlib import Path

# Agrega src/ al path para que los módulos de vit sean importables sin instalación
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
