from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import interpolate, ndimage
from skimage.transform import resize


class PreprocessingService:
    IMG_SIZE = (128, 128)
    MIDDLE_SLICE_RATIO = 0.5

    STANDARD_LANDMARKS = np.array(
        [
            1.11170489e-41,
            5.18809652e-25,
            3.05301042e-17,
            4.70776470e-12,
            6.94896974e-07,
            5.00224718e-03,
            1.75081036e-01,
            5.40189319e-01,
            7.13654902e-01,
            8.37299432e-01,
            9.57721521e-01,
        ],
        dtype=np.float32,
    )
    PERCENTILES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

    def remove_artifacts(self, volume: np.ndarray, percentile: int = 99) -> np.ndarray:
        nz = volume[volume > 0]
        if nz.size == 0:
            return volume
        thr = np.percentile(nz, percentile)
        return np.clip(volume, 0, thr)

    def normalize01(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mn, mx = float(x.min()), float(x.max())
        if mx > mn:
            return (x - mn) / (mx - mn)
        return x

    def resize_128(self, x2d: np.ndarray) -> np.ndarray:
        return resize(
            x2d,
            self.IMG_SIZE,
            order=3,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

    def center_slice(self, slice_img: np.ndarray, target_center=(64, 64)) -> np.ndarray:
        if np.count_nonzero(slice_img) < 100:
            return slice_img
        cy, cx = ndimage.center_of_mass(slice_img > 0.1)
        shift_y = target_center[0] - cy
        shift_x = target_center[1] - cx
        return ndimage.shift(slice_img, [shift_y, shift_x], order=1, mode="constant", cval=0)

    def nyul_normalize(self, img_01: np.ndarray) -> np.ndarray:
        img = img_01.astype(np.float32)
        mask = img > 0
        if np.sum(mask) < 100:
            return img

        curr_landmarks = np.percentile(img[mask], self.PERCENTILES).astype(np.float32)

        x = np.concatenate(([0.0], curr_landmarks, [float(img.max())])).astype(np.float32)
        y = np.concatenate(([0.0], self.STANDARD_LANDMARKS, [float(self.STANDARD_LANDMARKS[-1])])).astype(np.float32)

        x_unique, idx = np.unique(x, return_index=True)
        y_unique = y[idx]
        if x_unique.size < 2:
            return img

        f = interpolate.interp1d(x_unique, y_unique, bounds_error=False, fill_value="extrapolate")
        out = f(img).astype(np.float32)
        out[~mask] = 0.0
        return np.clip(out, 0.0, 1.0)

    def pick_middle_slice_index(self, vol: np.ndarray) -> int:
        z = vol.shape[2]
        start = int(z * (0.5 - self.MIDDLE_SLICE_RATIO / 2))
        end = int(z * (0.5 + self.MIDDLE_SLICE_RATIO / 2))
        start = max(0, start)
        end = min(z, end)
        if end <= start:
            return z // 2

        best_i = (start + end) // 2
        best_score = -1
        for i in range(start, end):
            sl = vol[:, :, i]
            score = int(np.count_nonzero(sl))
            if score > best_score:
                best_score = score
                best_i = i
        return best_i

    def get_slice_indices_around(self, center_idx: int, depth: int, k: int) -> list[int]:
        k = int(k)
        if k <= 1:
            return [int(center_idx)]
        if k % 2 == 0:
            k += 1

        half = k // 2
        idxs = [center_idx + i for i in range(-half, half + 1)]
        idxs = [max(0, min(depth - 1, i)) for i in idxs]

        idxs_unique = []
        for i in idxs:
            if i not in idxs_unique:
                idxs_unique.append(i)
        return idxs_unique

    def preprocess_single_slice_from_volume(
        self,
        vol01: np.ndarray,
        idx: int,
        apply_nyul: bool,
        apply_center: bool,
    ) -> np.ndarray:
        sl = vol01[:, :, idx]
        sl = self.resize_128(sl)
        if apply_center:
            sl = self.center_slice(sl)
        sl = np.clip(sl, 0, 1).astype(np.float32)
        if apply_nyul:
            sl = self.nyul_normalize(sl)
        return sl

    def preprocess_any(self, file_bytes: bytes, filename: str, apply_nyul: bool, apply_center: bool):
        name = filename.lower()
        debug = {}

        if name.endswith(".nii") or name.endswith(".nii.gz"):
            suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
            tmp_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp_path = tmp.name
                tmp.write(file_bytes)
                tmp.close()

                img_obj = nib.load(tmp_path)
                img_obj = nib.as_closest_canonical(img_obj)
                vol = img_obj.get_fdata()
            finally:
                if tmp_path is not None:
                    try:
                        import gc

                        gc.collect()
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            if vol.ndim == 4:
                vol = vol[..., 0]
            if vol.ndim != 3:
                raise ValueError(f"Expected 3D NIfTI. Got shape {vol.shape}")

            vol = self.remove_artifacts(vol, percentile=99)
            vol = self.normalize01(vol)

            idx = self.pick_middle_slice_index(vol)
            sl = vol[:, :, idx]

            sl = self.resize_128(sl)
            if apply_center:
                sl = self.center_slice(sl)
            sl = np.clip(sl, 0, 1).astype(np.float32)
            if apply_nyul:
                sl = self.nyul_normalize(sl)

            nz_ratio = float(np.count_nonzero(sl) / sl.size)
            debug.update(
                {
                    "type": "nifti",
                    "orig_shape": tuple(vol.shape),
                    "selected_slice": int(idx),
                    "nonzero_ratio": nz_ratio,
                    "min": float(sl.min()),
                    "max": float(sl.max()),
                }
            )
            if nz_ratio < 0.05:
                debug["warning"] = "Sparse slice (<5% non-zero) - might be mask/empty slice."

            return sl, debug

        if name.endswith(".npy"):
            arr = np.load(io.BytesIO(file_bytes), allow_pickle=False)
            arr = np.array(arr)

            if arr.ndim == 3:
                if arr.shape[0] < 32 and arr.shape[2] >= 32:
                    arr = np.transpose(arr, (1, 2, 0))
                idx = self.pick_middle_slice_index(arr)
                sl = arr[:, :, idx]
                debug["selected_slice"] = int(idx)
            elif arr.ndim == 2:
                sl = arr
            else:
                raise ValueError(f"Unsupported npy shape: {arr.shape}")

            sl = sl.astype(np.float32)
            if np.any(sl > 0):
                sl = np.clip(sl, 0, np.percentile(sl[sl > 0], 99))
            sl = self.normalize01(sl)

            sl = self.resize_128(sl)
            if apply_center:
                sl = self.center_slice(sl)
            sl = np.clip(sl, 0, 1).astype(np.float32)
            if apply_nyul:
                sl = self.nyul_normalize(sl)

            debug.update(
                {
                    "type": "npy",
                    "orig_shape": tuple(arr.shape),
                    "nonzero_ratio": float(np.count_nonzero(sl) / sl.size),
                    "min": float(sl.min()),
                    "max": float(sl.max()),
                }
            )
            return sl, debug

        from PIL import Image

        img = Image.open(io.BytesIO(file_bytes)).convert("L")
        sl = np.array(img).astype(np.float32)

        if np.any(sl > 0):
            sl = np.clip(sl, 0, np.percentile(sl[sl > 0], 99))
        sl = self.normalize01(sl)

        sl = self.resize_128(sl)
        if apply_center:
            sl = self.center_slice(sl)
        sl = np.clip(sl, 0, 1).astype(np.float32)
        if apply_nyul:
            sl = self.nyul_normalize(sl)

        debug.update(
            {
                "type": "image",
                "orig_shape": tuple(np.array(img).shape),
                "nonzero_ratio": float(np.count_nonzero(sl) / sl.size),
                "min": float(sl.min()),
                "max": float(sl.max()),
            }
        )
        return sl, debug

    def load_nifti_volume_from_bytes(self, file_bytes: bytes, filename: str) -> np.ndarray:
        name = filename.lower()
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            raise ValueError("Expected NIfTI file")

        suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_path = tmp.name
            tmp.write(file_bytes)
            tmp.close()

            img_obj = nib.load(tmp_path)
            img_obj = nib.as_closest_canonical(img_obj)
            vol = img_obj.get_fdata()
        finally:
            if tmp_path is not None:
                try:
                    import gc

                    gc.collect()
                    os.unlink(tmp_path)
                except Exception:
                    pass

        if vol.ndim == 4:
            vol = vol[..., 0]
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI. Got shape {vol.shape}")

        vol = self.remove_artifacts(vol, percentile=99)
        vol = self.normalize01(vol)
        return vol
