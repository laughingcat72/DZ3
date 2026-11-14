import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('checkpoints'));a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
url='https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt'
target=a.output/'sam2.1_hiera_tiny.pt'
if not target.exists():
    partial=target.with_suffix('.partial');urllib.request.urlretrieve(url,partial);partial.replace(target)
digest=hashlib.sha256(target.read_bytes()).hexdigest()
reference=Path(__file__).resolve().parents[1]/'results/weights_manifest.json'
if reference.exists():
    items=json.loads(reference.read_text())
    expected=next((r['sha256'] for r in items if r['filename']==target.name),None)
    if expected and digest!=expected:raise RuntimeError('SAM checkpoint checksum mismatch')
print(f'{target}: SHA256={digest}')
print('Torchvision weights are downloaded automatically on first model construction with hash validation.')
