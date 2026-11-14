"""Restore saved image artifacts from independent ZIPs; no model execution."""
from pathlib import Path

def unpack_results(root):
    import zipfile
    root=Path(root).resolve()
    restored=0
    for archive in sorted((root/'results_archives').glob('*.zip')):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if member.is_dir(): continue
                target=(root/member.filename).resolve()
                if not target.is_relative_to(root/'results'):
                    raise ValueError(f'Unsafe archive member: {member.filename}')
                if target.exists(): continue
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(z.read(member))
                restored+=1
    return restored

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    print(f'Restored {unpack_results(root)} saved files; no models executed.')
