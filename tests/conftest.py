import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
# src/ para módulos de vit
sys.path.insert(0, str(ROOT / "src"))
# raíz para app/
sys.path.insert(0, str(ROOT))
