import fnmatch
import os
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime


ZIP_URL = "https://github.com/python-igarashi/WORK.KENDO/archive/refs/heads/main.zip"

EXCLUDE_DIRS = {
    ".git",
    "bak",
    "__pycache__",
    "DownloadTSV",
    "TournamentFiles",
}

EXCLUDE_FILES = {
    "tournament_settings.json",
}

EXCLUDE_PATTERNS = [
    "credentials_*.json",
    "*.pyc",
    "*.tsv",
    "*.xlsx",
    "*.svg",
    "*.log",
]

TEXT_FILE_EXTENSIONS = {
    ".py",
    ".md",
    ".json",
    ".txt",
}

TEXT_FILE_NAMES = {
    ".gitattributes",
    ".gitignore",
}


def should_skip(relative_path):
    parts = relative_path.split(os.sep)
    if any(part in EXCLUDE_DIRS for part in parts):
        return True

    name = os.path.basename(relative_path)
    if name in EXCLUDE_FILES:
        return True

    return any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDE_PATTERNS)


def find_extracted_root(extract_dir):
    entries = [
        os.path.join(extract_dir, name)
        for name in os.listdir(extract_dir)
        if os.path.isdir(os.path.join(extract_dir, name))
    ]
    if len(entries) != 1:
        raise RuntimeError("zipファイル内のフォルダ構成を確認できませんでした。")
    return entries[0]


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def backup_existing_file(dst_path, repo_dir, backup_dir):
    if not os.path.exists(dst_path):
        return False

    relative_path = os.path.relpath(dst_path, repo_dir)
    backup_path = os.path.join(backup_dir, relative_path)
    ensure_parent_dir(backup_path)
    shutil.copy2(dst_path, backup_path)
    return True


def is_text_file(relative_path):
    name = os.path.basename(relative_path).lower()
    _, ext = os.path.splitext(name)
    return name in TEXT_FILE_NAMES or ext in TEXT_FILE_EXTENSIONS


def read_bytes(path):
    with open(path, "rb") as file:
        return file.read()


def normalize_newlines(data):
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def files_are_same(src_path, dst_path, relative_path):
    if not os.path.exists(dst_path):
        return False

    src_data = read_bytes(src_path)
    dst_data = read_bytes(dst_path)

    if src_data == dst_data:
        return True

    if is_text_file(relative_path):
        return normalize_newlines(src_data) == normalize_newlines(dst_data)

    return False


def copy_latest_file(src_path, dst_path, relative_path):
    if is_text_file(relative_path):
        data = normalize_newlines(read_bytes(src_path)).replace(b"\n", b"\r\n")
        with open(dst_path, "wb") as dst_file:
            dst_file.write(data)
        shutil.copystat(src_path, dst_path)
        return

    shutil.copy2(src_path, dst_path)


def copy_latest_files(src_root, repo_dir, backup_dir):
    copied = 0
    backed_up = 0
    skipped = 0
    unchanged = 0

    for root, dirs, files in os.walk(src_root):
        dirs[:] = [dirname for dirname in dirs if dirname not in EXCLUDE_DIRS]

        for filename in files:
            src_path = os.path.join(root, filename)
            relative_path = os.path.relpath(src_path, src_root)
            dst_path = os.path.join(repo_dir, relative_path)

            if should_skip(relative_path):
                skipped += 1
                continue

            if files_are_same(src_path, dst_path, relative_path):
                unchanged += 1
                continue

            if backup_existing_file(dst_path, repo_dir, backup_dir):
                backed_up += 1

            ensure_parent_dir(dst_path)
            copy_latest_file(src_path, dst_path, relative_path)
            copied += 1

    return copied, backed_up, skipped, unchanged


def download_zip(zip_path):
    print("最新版をダウンロードしています...")
    urllib.request.urlretrieve(ZIP_URL, zip_path)
    print("ダウンロードが完了しました。")


def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(repo_dir, "bak", f"latest_{timestamp}")

    print("最新プログラムに更新します。")
    print(f"更新元: {ZIP_URL}")
    print(f"更新先: {repo_dir}")
    print("")
    print("認証情報、取得済みデータ、出力ファイル、大会設定は更新対象外です。")
    print("")

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "main.zip")
        extract_dir = os.path.join(tmp_dir, "extract")

        download_zip(zip_path)

        print("zipファイルを展開しています...")
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extract_dir)

        src_root = find_extracted_root(extract_dir)

        print("ファイルを更新しています...")
        copied, backed_up, skipped, unchanged = copy_latest_files(
            src_root, repo_dir, backup_dir
        )

    print("")
    print("更新が完了しました。")
    print(f"更新ファイル数: {copied}")
    print(f"バックアップファイル数: {backed_up}")
    print(f"更新対象外ファイル数: {skipped}")
    print(f"変更なしファイル数: {unchanged}")
    if backed_up:
        print(f"バックアップ先: {backup_dir}")
    print("")
    print("PortalGUIを起動中の場合は、いったん閉じてから再起動してください。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("")
        print(f"[エラー] {exc}")
        raise
