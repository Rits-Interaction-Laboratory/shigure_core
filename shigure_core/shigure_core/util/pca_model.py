"""Build pca_model.pkl from face_models/*.npy (frontal + profile)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from shigure_core.util.face_models_dir import get_face_models_dir


@dataclass(frozen=True)
class PcaBuildResult:
    success: bool
    output_path: Path
    num_samples: int = 0
    num_users: int = 0
    message: str = ''


def count_user_dirs(face_models_dir: Path) -> int:
    if not face_models_dir.is_dir():
        return 0
    return sum(1 for p in face_models_dir.glob('user_*') if p.is_dir())


def iter_embedding_paths(face_models_dir: Path) -> Iterator[Path]:
    """Yield frontal and profile .npy paths under user_* directories."""
    if not face_models_dir.is_dir():
        return
    for user_dir in sorted(face_models_dir.glob('user_*')):
        if not user_dir.is_dir():
            continue
        for npy_path in sorted(user_dir.glob('*.npy')):
            yield npy_path
        profile_dir = user_dir / 'profile'
        if profile_dir.is_dir():
            for npy_path in sorted(profile_dir.glob('*.npy')):
                yield npy_path


def user_id_for_npy(npy_path: Path) -> str:
    if npy_path.parent.name == 'profile':
        return npy_path.parent.parent.name
    return npy_path.parent.name


def load_embeddings_matrix(face_models_dir: Path) -> tuple[np.ndarray, int]:
    vectors: list[np.ndarray] = []
    user_ids: set[str] = set()
    for npy_path in iter_embedding_paths(face_models_dir):
        user_ids.add(user_id_for_npy(npy_path))
        raw = np.squeeze(np.load(npy_path)).astype(np.float32).reshape(-1)
        if raw.size == 0:
            continue
        vectors.append(raw)
    if not vectors:
        return np.empty((0, 0), dtype=np.float32), 0
    lengths = {v.shape[0] for v in vectors}
    if len(lengths) != 1:
        dim = max(lengths)
        vectors = [v for v in vectors if v.shape[0] == dim]
    if not vectors:
        return np.empty((0, 0), dtype=np.float32), len(user_ids)
    return np.vstack(vectors), len(user_ids)


def default_pca_model_path(face_models_dir: Path) -> Path:
    return face_models_dir / 'pca_model.pkl'


def build_pca_model(
    face_models_dir: Path,
    output_path: Optional[Path] = None,
    *,
    n_components: int = 2,
) -> PcaBuildResult:
    """Fit sklearn PCA on all embeddings and write pca_model.pkl."""
    from sklearn.decomposition import PCA

    face_models_dir = Path(face_models_dir)
    out = Path(output_path) if output_path is not None else default_pca_model_path(face_models_dir)

    matrix, num_users = load_embeddings_matrix(face_models_dir)
    if matrix.shape[0] < n_components:
        return PcaBuildResult(
            success=False,
            output_path=out,
            num_samples=int(matrix.shape[0]),
            num_users=num_users,
            message=f'need at least {n_components} samples, got {matrix.shape[0]}',
        )

    pca = PCA(n_components=n_components)
    pca.fit(matrix.astype(np.float32))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(pca, f)

    return PcaBuildResult(
        success=True,
        output_path=out,
        num_samples=int(matrix.shape[0]),
        num_users=num_users,
        message=f'wrote {out} ({matrix.shape[0]} samples, {num_users} users, dim={matrix.shape[1]})',
    )


def main() -> None:
    import argparse

    default_face_models_dir = get_face_models_dir()

    parser = argparse.ArgumentParser(description='Rebuild pca_model.pkl from face_models')
    parser.add_argument(
        '--face-models-dir',
        type=Path,
        default=default_face_models_dir,
    )
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    result = build_pca_model(args.face_models_dir, args.output)
    print(result.message)
    if not result.success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
