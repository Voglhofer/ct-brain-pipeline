#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

FILES = [
    ("data.zip", "1buISR_b3HQDU4KeNc_DmvKTYJ1gvj5-3", "aux"),
    ("csv.zip", "1qYi4k-DuOLJmyZ7uYYrnomU2U7MrYRBV", "aux"),
    ("feature_samples.zip", "1lJgzZoHFu6HI4JBktkGY3qMk--28IUkC", "aux"),
    ("seresnext101_256.zip", "18Py5eW1E4hSbTT6658IAjQjJGS28grdx", "pre"),
    ("densenet169_256.zip", "1vCsX12pMZxBmuGGNVnjFFiZ-5u5vD-h6", "pre"),
    ("densenet121_512.zip", "1o0ok-6I2hY1ygSWdZOKmSD84FsEpgDaa", "pre"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-root', required=True)
    parser.add_argument('--pretrained-root', required=True)
    args = parser.parse_args()

    download_root = Path(args.download_root)
    pretrained_root = Path(args.pretrained_root)
    download_root.mkdir(parents=True, exist_ok=True)
    pretrained_root.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except Exception:
        print('gdown mangler. Installer miljøet først.', file=sys.stderr)
        raise

    failed = []
    for name, file_id, target_type in FILES:
        target_dir = pretrained_root if target_type == 'pre' else download_root
        out_path = target_dir / name
        if out_path.exists():
            print(f'Skipper {out_path} (findes allerede)')
            continue
        url = f'https://drive.google.com/uc?id={file_id}'
        print(f'Downloader {name} ...')
        try:
            result = gdown.download(url, str(out_path), quiet=False, fuzzy=True)
            if result is None:
                raise RuntimeError('gdown returnerede None')
        except Exception as e:
            print(f'[warn] Kunne ikke downloade {name} med gdown API: {e}', file=sys.stderr)
            # Fjern evt. tom/ufuldstændig fil
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
            failed.append(name)

    if failed:
        print(f'\n[advarsel] Følgende filer kunne ikke downloades: {", ".join(failed)}')
        print('Download dem manuelt fra Google Drive:')
        for name, file_id, target_type in FILES:
            if name in failed:
                target_dir = pretrained_root if target_type == 'pre' else download_root
                print(f'  https://drive.google.com/uc?id={file_id}&export=download')
                print(f'    -> {target_dir / name}')


if __name__ == '__main__':
    main()