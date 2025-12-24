# Repository Guidelines

原則として回答・説明・出力は日本語で行う（コード引用は原文のまま）。
Pythonの実行・テストはWindows(cmd)で行う。Codexは実行せず、必要ならコマンド提示に留める。

## Project Structure & Module Organization
このリポジトリは内部大会向けの Python ツール集です。各大会フォルダに README と実行スクリプトがあります。

- `研修会/`, `春の大会/`, `秋の大会/`: main event toolsets (Google Sheets download, aggregation, and output).
- `テスト/`: ad-hoc development or verification scripts.
- Root `README.md`: setup prerequisites, dependency list, and shared assumptions.

探索の起点と順序は固定します: 「各大会README → `Defines.py` → `01_*.py` → `02_*.py` → `03_*.py` → `04_*.py` → `05_*.py`」。
各大会フォルダには `Defines.py`（設定）と `01_`〜`05_` の手順スクリプトがあり、必要に応じて `Tournament.py`/`Booklet.py` が含まれます。

## Build, Test, and Development Commands
原則 read-only（閲覧のみ）で作業し、変更が必要な場合はまず unified diff を提示し、適用前に確認を取ります。
対象フォルダに credential ファイル（例: `credentials_*.json`）を配置してから実行します。

- `python 01_PrintURL.py`: print Google Sheet URLs for data entry.
- `python 02_DownloadTSV.py`: download all sheets as TSV files.
- `python 03_SumPlayer.py`: aggregate players and write results.
- `python 04_SumStaff.py`: aggregate staff (spring/autumn).
- `python 05_CreateTournament.py`: generate tournament brackets (if present).
- `python XX_RemakeFiles.py`: admin-only reset of sheets（危険操作、実行前に必ず明示確認が必要）.

Dependencies are listed in the root `README.md` and installed via `pip install ...`.

## Coding Style & Naming Conventions
- Match existing indentation and formatting in the file you touch; avoid sweeping reformatting.
- Prefer Python `snake_case` for variables and functions, and keep script names aligned with the numbered workflow (e.g., `01_*.py`, `02_*.py`).
- Keep configuration in `Defines.py` rather than hard-coding in multiple scripts.

## Testing Guidelines
There is no automated test framework. Validate changes by running the relevant scripts end-to-end and checking:
- Downloaded TSV output in `DownloadTSV/`.
- Google Sheet output for aggregates.
- Tournament files in `TournamentFiles/` (if applicable).

Use `テスト/Tournament_SVG.py` only for local experiments.

## Commit & Pull Request Guidelines
Recent history uses short, direct commit messages (often Japanese), e.g., "リファクタ" or "Update README.md". Keep commits concise and scoped.

For PRs, include:
- A brief description of the change and affected event folder.
- Steps you ran (script names).
- Notes about any template or Google Sheet changes.

## Security & Configuration Tips
- `credentials_*.json` などの秘匿情報は出力・ログに含めない（Do not print or log secrets）.
- Credential JSON files are not tracked; keep them local and out of Git.
- Only system administrators should run sheet reset scripts (`XX_RemakeFiles.py`).
