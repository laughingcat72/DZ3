import argparse
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--checkpoints',type=Path,default=Path('checkpoints'))
a=p.parse_args();root=Path(__file__).resolve().parents[1];os.chdir(root)
os.environ.setdefault('TORCH_HOME',str(a.checkpoints.resolve()/'torch'))
for stage in ['analyse','retinanet','fasterrcnn','summarize']:
    subprocess.run([sys.executable,'src/experiment.py','--data',str(a.data.resolve()),'--stage',stage],check=True)
cmd=[sys.executable,'src/segmentation.py','--data',str(a.data.resolve()),'--checkpoint',str((a.checkpoints/'sam2.1_hiera_tiny.pt').resolve())]
manual=root/'manual_prompts.json'
if manual.exists():cmd+=['--manual',str(manual)]
subprocess.run(cmd,check=True)
subprocess.run([sys.executable,'scripts/make_summary.py'],check=True)
print('Experiments finished. Open plankton_experiments.ipynb to inspect tables and figures.')
