"""Reproduce manuscript result plots from bundled saved measurements."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for name in ['figures','tables']:(root/name).mkdir(exist_ok=True)
for script in ['rebuild_tradeoff_figure.py','rebuild_listening_focus.py','rebuild_listening_per_emotion.py','rebuild_appendix.py']:
    subprocess.run([sys.executable,str(root/script)],cwd=root,check=True)
print('Figures:',root/'figures')
