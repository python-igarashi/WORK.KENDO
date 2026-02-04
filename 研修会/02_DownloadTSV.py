from google.oauth2 import service_account
from googleapiclient.discovery import build
import gspread
import requests
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
gc = gspread.authorize(creds)

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
	
	# 団体のスプレッドシートを開き、全てのワークシートを走査
	ss = gc.open_by_key(file_id)
	for worksheet in ss.worksheets():
		# ワークシートの内容を取得
		gid = worksheet.id
		url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=tsv&gid={gid}"
		res = requests.get(url)
		
		# {グループ名}_{ワークシート名}.tsv に出力
		with open(f"{Defines.download_folder}\\{groupname}_{worksheet.title}.tsv", "wb") as f:
			f.write(res.content)

