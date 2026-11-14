import argparse
from pathlib import Path, PurePosixPath
import zipfile

p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('data'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.zip) as z:
    members=[i for i in z.infolist() if not i.is_dir() and PurePosixPath(i.filename).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}]
    names=[PurePosixPath(i.filename).name for i in members]
    if len(set(names))!=len(names):raise ValueError('Duplicate image basenames')
    for i,name in zip(members,names):
        target=a.output/name
        if target.exists():raise FileExistsError(target)
        target.write_bytes(z.read(i))
print(f'Extracted {len(members)} images to {a.output.resolve()}')
