"""
Standalone profiler for the per-sample corruption path.

Replays exactly what one DataLoader worker does per sample during a corrupted
pass -- DetectionDataset.__getitem__ (disk open + resize + ToTensor) followed by
CorruptedDataset's tensor<->uint8 thrash around corr_func_sample -- and times
each sub-step in isolation, single-threaded, over N real BCCD images.

Nothing in the plugin is modified; this only imports and calls it.

Usage:
    python profile_corruption.py [N_SAMPLES]
"""
import sys
import time
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, ".")  # so `import odaugvm_algorithm.*` resolves from here

from odaugvm_algorithm.augmentations_class import make_augmentation_dict  # noqa: E402
from albumentations.pytorch import ToTensorV2  # noqa: E402
import albumentations as A  # noqa: E402

IMG_DIR = "/home/bjieyong/aiverify/cvrob/bccd/BCCD/JPEGImages"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
MIN_SIZE = 500

to_tensor = transforms.ToTensor()


def resize_up(image, min_size=MIN_SIZE):
    W, H = image.size
    s = min(W, H)
    if s >= min_size:
        return image
    scale = min_size / s
    return image.resize((round(W * scale), round(H * scale)), Image.BILINEAR)


def profile_aug(aug_class, image_paths, n):
    """Time each per-sample stage for one augmentation, averaged over n images."""
    severities = aug_class.severities
    severity = severities[len(severities) // 2]  # a mid severity
    aug_class.set_seed(42)

    # dummy target with one box (needed by geometric augs); label 0
    def make_target():
        return {
            "boxes": torch.tensor([[50.0, 50.0, 200.0, 200.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
        }

    acc = {k: 0.0 for k in
           ["open", "resize", "totensor", "to_uint8", "corrupt", "back"]}

    for path in image_paths[:n]:
        target = make_target()

        t = time.perf_counter()
        image = Image.open(path).convert("RGB")
        acc["open"] += time.perf_counter() - t

        t = time.perf_counter()
        image = resize_up(image)
        acc["resize"] += time.perf_counter() - t

        t = time.perf_counter()
        img_t = to_tensor(image)  # CHW float [0,1]
        acc["totensor"] += time.perf_counter() - t

        t = time.perf_counter()
        image_np = img_t.mul(255).byte().cpu().numpy().transpose(1, 2, 0)  # HWC uint8
        acc["to_uint8"] += time.perf_counter() - t

        t = time.perf_counter()
        corrupted_image, _ = aug_class.corr_func_sample(image_np, target, severity)
        acc["corrupt"] += time.perf_counter() - t

        t = time.perf_counter()
        _ = torch.from_numpy(
            np.ascontiguousarray(corrupted_image.transpose(2, 0, 1))
        ).float().div_(255.0)
        acc["back"] += time.perf_counter() - t

    per = {k: v / n * 1000.0 for k, v in acc.items()}  # ms per sample
    total = sum(per.values())
    return severity, per, total


def measure_compose_overhead(aug_class, image_paths, n):
    """Isolate: cost of REBUILDING A.Compose per image vs a prebuilt one.

    Only meaningful for photometric albumentations augs (the path that goes
    through corrupt_func_album_reduced, which rebuilds Compose every call).
    """
    if aug_class.requires_bbox_transform or str(aug_class.aug_func) == "None":
        return None
    if "albumentations" not in str(aug_class.aug_func):
        return None

    severity = aug_class.severities[len(aug_class.severities) // 2]
    aug_class.set_seed(42)
    params = {k: v for k, v in aug_class.param_dict[severity].items()
              if k != "random_seed"}
    aug_func = aug_class.aug_func

    # one representative HWC uint8 image
    img = np.array(resize_up(Image.open(image_paths[0]).convert("RGB"))).astype(np.uint8)

    t = time.perf_counter()
    for _ in range(n):
        _ = A.Compose([aug_func(**params), ToTensorV2()])
    build_ms = (time.perf_counter() - t) / n * 1000.0

    prebuilt = A.Compose([aug_func(**params), ToTensorV2()])
    t = time.perf_counter()
    for _ in range(n):
        _ = prebuilt(image=img)["image"]
    apply_ms = (time.perf_counter() - t) / n * 1000.0

    return build_ms, apply_ms


def main():
    import glob
    image_paths = sorted(glob.glob(f"{IMG_DIR}/*.jpg"))
    if not image_paths:
        print(f"No images found in {IMG_DIR}")
        return
    n = min(N, len(image_paths))
    print(f"Profiling over {n} real images ({IMG_DIR.split('/')[-1]}), single-threaded\n")

    album = make_augmentation_dict("albumentations")
    nrtk = make_augmentation_dict("nrtk")

    # representative picks: photometric, geometric, JPEG-recompress, NRTK-optical
    picks = []
    for name in ["Rain"]:
        if name in album:
            picks.append(album[name])

    for aug in picks:
        severity, per, total = profile_aug(aug, image_paths, n)
        geo = "geometric" if aug.requires_bbox_transform else "photometric"
        print(f"=== {aug.name}  ({geo}, severity={severity}) ===")
        for k in ["open", "resize", "totensor", "to_uint8", "corrupt", "back"]:
            print(f"  {k:>9}: {per[k]:7.3f} ms/img  ({per[k]/total*100:4.1f}%)")
        print(f"  {'TOTAL':>9}: {total:7.3f} ms/img\n")

        ov = measure_compose_overhead(aug, image_paths, n)
        if ov:
            build_ms, apply_ms = ov
            print(f"    within 'corrupt': A.Compose REBUILD = {build_ms:.3f} ms/img, "
                  f"apply = {apply_ms:.3f} ms/img  "
                  f"(rebuild is {build_ms/(build_ms+apply_ms)*100:.0f}% of the album op)\n")


if __name__ == "__main__":
    main()
