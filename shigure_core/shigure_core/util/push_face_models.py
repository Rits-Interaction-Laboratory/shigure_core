"""face_models を別 git リポジトリへスナップショット push する.

コード用 shigure_core には載せない。個人データ（顔画像・埋め込み）専用の
非公開リポジトリへ、rsync 後に commit/push する。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from shigure_core.util.face_models_dir import get_face_models_dir

DEFAULT_REPO_DIR = Path.home() / 'face_models_after_Adaface'
DEFAULT_REMOTE = 'origin'
# 旧 OpenCV モデル等は顔辞書ではないので同期しない
DEFAULT_EXCLUDES = (
    '.git',
    'README.md',
    '*.onnx',
    '*.download',
    '*.bak*',
    '__pycache__',
)


def _run(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """サブプロセスを実行する."""
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        capture_output=True,
        text=True,
    )


def _ensure_local_git_identity(repo_dir: Path) -> None:
    """作業コピーにだけ commit 用 identity を設定する（global は触らない）."""
    name = _run(
        ['git', 'config', '--local', 'user.name'],
        cwd=repo_dir,
        check=False,
    )
    if name.returncode != 0 or not name.stdout.strip():
        _run(
            ['git', 'config', '--local', 'user.name', 'face_models_push'],
            cwd=repo_dir,
        )
    email = _run(
        ['git', 'config', '--local', 'user.email'],
        cwd=repo_dir,
        check=False,
    )
    if email.returncode != 0 or not email.stdout.strip():
        _run(
            [
                'git', 'config', '--local', 'user.email',
                'face_models_push@localhost',
            ],
            cwd=repo_dir,
        )


def _path_matches_exclude(rel_path: Path, excludes: Sequence[str]) -> bool:
    """相対パスが exclude パターンに当たるなら True."""
    name = rel_path.name
    as_posix = rel_path.as_posix()
    for pattern in excludes:
        if pattern == '.git' and (name == '.git' or as_posix.startswith('.git/')):
            return True
        if Path(name).match(pattern) or Path(as_posix).match(pattern):
            return True
    return False


def _sync_face_models(
    src: Path,
    repo_dir: Path,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
) -> None:
    """face_models をバックアップ用作業コピーへ同期する.

    rsync があれば使う。無ければ shutil で同等のコピーを行う
    （Docker 素のイメージでも動くようにする）。
    """
    if shutil.which('rsync') is not None:
        cmd: List[str] = ['rsync', '-a', '--delete']
        for pattern in excludes:
            cmd.extend(['--exclude', pattern])
        cmd.extend([f'{src}/', f'{repo_dir}/'])
        result = _run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f'rsync failed: '
                f'{result.stderr.strip() or result.stdout.strip()}'
            )
        return

    # shutil フォールバック: 宛先の余分なファイルを消してからコピー
    keep_roots = {'.git', 'README.md'}
    for child in list(repo_dir.iterdir()):
        if child.name in keep_roots:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for path in src.rglob('*'):
        rel = path.relative_to(src)
        # 親ディレクトリが除外対象ならスキップ（'.' 自身は見ない）
        skip = False
        for parent in list(rel.parents)[:-1]:
            if _path_matches_exclude(parent, excludes):
                skip = True
                break
        if skip or _path_matches_exclude(rel, excludes):
            continue
        dest = repo_dir / rel
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


def push_face_models(
    repo_dir: Path | str | None = None,
    *,
    source_dir: Path | str | None = None,
    remote: str = DEFAULT_REMOTE,
    dry_run: bool = False,
) -> str:
    """変更があれば commit して push し、結果メッセージを返す.

    :param repo_dir: バックアップ用 git 作業コピー
    :param source_dir: face_models 実体（未指定なら get_face_models_dir）
    :param remote: push 先 remote 名
    :param dry_run: True なら rsync/commit/push せず差分有無だけ返す
    """
    src = Path(source_dir) if source_dir is not None else get_face_models_dir()
    dest = Path(repo_dir) if repo_dir is not None else Path(
        os.environ.get(
            'SHIGURE_FACE_MODELS_BACKUP_REPO',
            str(DEFAULT_REPO_DIR),
        )
    ).expanduser()
    dest = dest.resolve()

    if not src.is_dir():
        raise FileNotFoundError(f'face_models が無い: {src}')
    if not (dest / '.git').is_dir():
        raise FileNotFoundError(
            f'git リポジトリではない: {dest} '
            f'(先に clone または git init してください)'
        )

    if dry_run:
        return f'dry-run ok src={src} repo={dest}'

    _ensure_local_git_identity(dest)
    _sync_face_models(src, dest)
    readme = dest / 'README.md'
    if not readme.is_file():
        readme.write_text(
            '# face_models_after_Adaface\n\n'
            '**Private backup** of Shigure face_models '
            '(face crops + AdaFace embeddings).\n\n'
            '- Contains personal / biometric-related data.\n'
            '- **Do not make this repository public.**\n',
            encoding='utf-8',
        )

    _run(['git', 'add', '-A'], cwd=dest)
    diff = _run(['git', 'diff', '--cached', '--quiet'], cwd=dest, check=False)
    if diff.returncode == 0:
        return 'no changes'

    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    commit = _run(
        ['git', 'commit', '-m', f'face_models snapshot {stamp}'],
        cwd=dest,
        check=False,
    )
    if commit.returncode != 0:
        raise RuntimeError(
            f'git commit failed: {commit.stderr.strip() or commit.stdout.strip()}'
        )

    push = _run(['git', 'push', remote, 'HEAD'], cwd=dest, check=False)
    if push.returncode != 0:
        raise RuntimeError(
            f'git push failed: {push.stderr.strip() or push.stdout.strip()}'
        )
    return f'pushed ({stamp})'


def main() -> None:
    """コマンドラインから face_models をバックアップリポジトリへ push する."""
    parser = argparse.ArgumentParser(
        description='Push face_models snapshot to a separate private git repo',
    )
    parser.add_argument(
        '--repo-dir',
        type=Path,
        default=None,
        help=f'バックアップ用 git 作業コピー（既定: {DEFAULT_REPO_DIR}）',
    )
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=None,
        help='face_models 実体（未指定なら get_face_models_dir）',
    )
    parser.add_argument(
        '--remote',
        default=DEFAULT_REMOTE,
        help='push 先 remote 名（既定: origin）',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='同期・commit・push せずパス確認だけする',
    )
    args = parser.parse_args()
    message = push_face_models(
        args.repo_dir,
        source_dir=args.source_dir,
        remote=args.remote,
        dry_run=args.dry_run,
    )
    print(message)


if __name__ == '__main__':
    main()
