"""Audit original photos and freeze photo-disjoint protocols before model evaluation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'T1k_dataset'
for sub in ['audit', 'cache', 'protocol', 'results', 'predictions', 'checkpoints', 'logs']:
    (OUT / sub).mkdir(exist_ok=True)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-audit', action='store_true')
    args = parser.parse_args()
    started = time.time()
    classes = sorted(item['folder'] for item in json.loads(
        (DATA / '_download_support/fabric_classes.json').read_text(encoding='utf-8')))
    assert len(classes) == len(set(classes)) == 355
    expected = {}
    for path in (DATA / '_download_support').glob('*.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(value, dict) and 'members' in value:
            assert value['complete'], path
            for member in value['members'].values():
                expected[(member['folder'], member['filename'])] = member
    paths = [path for name in classes for path in sorted((DATA / 'fabric_images' / name).iterdir())
             if path.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    assert len(paths) == len(expected) == 1792
    spatial = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

    def inspect(item):
        index, path = item
        blob = path.read_bytes()
        checksum = sha(blob)
        record = expected[(path.parent.name, path.name)]
        assert len(blob) == record['size'] and checksum == record['sha256'], str(path)
        with Image.open(path) as source:
            exif = source.getexif()
            metadata = {key: str(exif.get(tag, '')) for key, tag in [
                ('camera_make', 271), ('camera_model', 272), ('exif_datetime', 306),
                ('exif_orientation', 274)]}
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.load()  # Full decode, not only JPEG magic bytes.
        width, height = image.size
        pixel_hash = sha(width.to_bytes(4, 'little') + height.to_bytes(4, 'little') + image.tobytes())
        resized = np.asarray(spatial(image)).copy()
        return dict(index=index, class_name=path.parent.name, label=classes.index(path.parent.name),
                    relative_path=str(path.relative_to(DATA / 'fabric_images')).replace('\\', '/'),
                    bytes=len(blob), sha256=checksum, decoded_pixel_sha256=pixel_hash,
                    width=width, height=height, **metadata), resized

    if args.reuse_audit:
        rows = json.loads((OUT / 'audit/image_manifest.json').read_text(encoding='utf-8'))
        assert len(rows) == 1792
        assert np.load(OUT / 'cache/center_pixels.npy', mmap_mode='r').shape == (1792,224,224,3)
    else:
        pixels = np.lib.format.open_memmap(OUT / 'cache/center_pixels.npy', mode='w+',
                                          dtype=np.uint8, shape=(1792, 224, 224, 3))
        rows = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for row, image in pool.map(inspect, enumerate(paths)):
                rows.append(row)
                pixels[row['index']] = image
                if len(rows) % 100 == 0:
                    print('audited', len(rows), '/', len(paths), 'seconds', round(time.time()-started), flush=True)
        pixels.flush()
        write_csv(OUT / 'audit/image_manifest.csv', rows)
        (OUT / 'audit/image_manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    duplicates = {}
    for field in ['sha256', 'decoded_pixel_sha256']:
        buckets = collections.defaultdict(list)
        for row in rows:
            buckets[row[field]].append(row['relative_path'])
        duplicates[field] = [value for value in buckets.values() if len(value) > 1]
    (OUT / 'audit/exact_duplicate_groups.json').write_text(json.dumps(duplicates, indent=2), encoding='utf-8')
    ambiguous = sorted({name.split('/')[0] for group in duplicates['decoded_pixel_sha256'] for name in group})
    if ambiguous:
        # Verified here: a452/a456 are complete copies, not partially overlapping collections.
        assert ambiguous == ['a452', 'a456']
        assert ({row['decoded_pixel_sha256'] for row in rows if row['class_name']=='a452'} ==
                {row['decoded_pixel_sha256'] for row in rows if row['class_name']=='a456'})
    assignments = []
    for class_name in classes:
        photos = [row for row in rows if row['class_name'] == class_name]
        # Fixed, prespecified per-class hash shuffle; filenames do not define support roles.
        photos.sort(key=lambda row: sha(('T1K355|20261029|' + row['decoded_pixel_sha256']).encode()))
        assert len(photos) >= 4
        for order, row in enumerate(photos):
            assignments.append(dict(**row, role='reference' if order == 0 else 'validation' if order == 1 else 'test',
                                    within_class_order=order,
                                    paired_two_reference_role='reference' if order in [0, 2] else 'validation' if order == 1 else 'test'))
    assignments.sort(key=lambda row: row['index'])
    write_csv(OUT / 'protocol/photo_splits.csv', assignments)
    (OUT / 'protocol/photo_splits.json').write_text(json.dumps(assignments, ensure_ascii=False, indent=2), encoding='utf-8')
    roles = collections.Counter(row['role'] for row in assignments)
    assert roles == {'reference': 355, 'validation': 355, 'test': 1082}
    for field in ['sha256','decoded_pixel_sha256']:
        hash_roles = collections.defaultdict(set)
        for row in assignments:
            hash_roles[row[field]].add(row['role'])
        assert all(len(value)==1 for value in hash_roles.values()), 'Exact duplicate crosses photo roles'
    # Alternative gallery choices are sensitivity results, not independently collected replications.
    reference_sensitivity = []
    for seed in [101, 102, 103, 104, 105]:
        support, query = [], []
        for name in classes:
            photos = [row for row in rows if row['class_name'] == name]
            photos.sort(key=lambda row: sha(f'T1K355|sensitivity|{seed}|{row["decoded_pixel_sha256"]}'.encode()))
            support.append(photos[0]['index'])
            query.extend(row['index'] for row in photos[1:])
        reference_sensitivity.append(dict(seed=seed, reference=support, query=query))
    (OUT / 'protocol/reference_sensitivity.json').write_text(json.dumps(reference_sensitivity, indent=2), encoding='utf-8')
    ordered_classes = sorted(set(classes)-set(ambiguous), key=lambda name: sha(f'T1K355|open-set|20261029|{name}'.encode()))
    open_protocol = dict(known=ordered_classes[:282], calibration_unknown=ordered_classes[282:317],
                         test_unknown=ordered_classes[317:], excluded_ambiguous_classes=ambiguous,
                         threshold_selection='Maximize balanced known/unknown accuracy using known validation photos and all calibration-unknown photos only; highest threshold on ties.',
                         limitation='Unknown Fabric instances, not professionally verified visually similar unknown leather; no new devices/sessions certified.')
    (OUT / 'protocol/open_set_class_splits.json').write_text(json.dumps(open_protocol, indent=2), encoding='utf-8')
    summary = dict(classes=355, photos=1792, bytes=sum(row['bytes'] for row in rows),
                   counts=dict(roles), min_photos=min(collections.Counter(row['class_name'] for row in rows).values()),
                   max_photos=max(collections.Counter(row['class_name'] for row in rows).values()),
                   exact_file_duplicate_groups=len(duplicates['sha256']),
                   exact_decoded_duplicate_groups=len(duplicates['decoded_pixel_sha256']),
                   ambiguous_classes=ambiguous, unique_source_photos=len({r['sha256'] for r in rows}),
                   no_exact_file_or_pixel_overlap_across_roles=True,
                   main_analysis='355 official labels retained, duplicate pair uses synchronized photo roles; strict ID tie-breaking; clean-353 sensitivity excludes this pair, without claiming new labels.',
                   every_source_matches_download_manifest=True, every_image_fully_decoded=True,
                   exif_orientation='EXIF transpose before all features, augmentations and pixel hashes',
                   paired_two_reference_queries=sum(row['paired_two_reference_role']=='test' for row in assignments),
                   metadata_limit='Public collection documents separate acquisitions; neither exact-duplicate checks nor EXIF metadata establish cross-session/device independence.',
                   elapsed_seconds=time.time()-started)
    (OUT / 'audit/dataset_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import torch, torchvision, PIL, scipy
    environment = dict(python=sys.executable, python_version=sys.version, platform=platform.platform(),
                       torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
                       scipy=scipy.__version__, pillow=PIL.__version__, cpu_threads=6, device='cpu',
                       command='prepare_dataset.py', source_script_sha256=sha(Path(__file__).read_bytes()))
    (OUT / 'audit/environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    print('COMPLETE DATA AUDIT', json.dumps(summary), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
