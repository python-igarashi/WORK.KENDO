from google.oauth2 import service_account
from googleapiclient.discovery import build
import csv
import os
import glob

import Defines

SCOPES = [
  'https://www.googleapis.com/auth/drive.readonly',
  'https://www.googleapis.com/auth/spreadsheets'
  ]


# ========= 認証とサービス初期化 ========= #
creds = service_account.Credentials.from_service_account_file(Defines.service_account_file, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)
sheets_service = build('sheets', 'v4', credentials=creds)

# ========= 既存TSVの削除 ========= #
download_dir = Defines.download_folder
for path in glob.glob(os.path.join(download_dir, "*.tsv")):
	try:
		os.remove(path)
	except Exception:
		pass

# ========= 全団体のスプレッドシートの、全てのワークシートのTSVファイルをダウンロード ========= #
for groupname in Defines.l_groupname + ["テンプレート", "記入例"]:
	print(f"{groupname} 処理中...")
	
	# 団体のスプレッドシートファイル名を取得
	filename_ss = Defines.filename_header + "." + groupname
	
	# 指定フォルダ内でファイル名を検索
	query = f"'{Defines.drive_id}' in parents and name = '{filename_ss}' and trashed = false"
	results = drive_service.files().list(q=query, fields="files(id)").execute()
	files = results.get('files', [])
	file_id = files[0]["id"]
	
	# 団体ごとに全ワークシートの表示文字列を一括取得し、
	# ワークシートごとに「{団体名}_{シート名}.tsv」として出力する
	spreadsheet = sheets_service.spreadsheets().get(
		spreadsheetId=file_id,
		fields="sheets(properties(title,gridProperties(columnCount)),data(rowData(values(formattedValue))))",
	).execute(num_retries=3)
	for worksheet in spreadsheet.get("sheets", []):
		properties = worksheet["properties"]
		column_count = properties["gridProperties"]["columnCount"]
		path = os.path.join(Defines.download_folder, f"{groupname}_{properties['title']}.tsv")
		with open(path, "w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f, delimiter="\t", lineterminator="\r\n")
			for grid in worksheet.get("data", []):
				for row in grid.get("rowData", []):
					values = [cell.get("formattedValue", "") for cell in row.get("values", [])]
					writer.writerow(values + [""] * max(0, column_count - len(values)))
