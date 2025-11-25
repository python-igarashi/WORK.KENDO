from google.oauth2 import service_account
from googleapiclient.discovery import build

import Defines

SCOPES = [
  'https://www.googleapis.com/auth/drive.readonly',
  ]


# ========= 認証とサービス初期化 ========= #
creds = service_account.Credentials.from_service_account_file(Defines.service_account_file, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

for groupname in Defines.l_groupname:
	filename = Defines.filename_header + "." + groupname

	# 指定フォルダ内で団体のファイル名を検索
	query = f"'{Defines.drive_id}' in parents and name = '{filename}' and trashed = false"
	results = drive_service.files().list(q=query, fields="files(id, name, webViewLink)").execute()
	files = results.get('files', [])

	if files:
		file = files[0]
		print(f"[{groupname}]\n{file['webViewLink']}\n")
	else:
		print(f"[{groupname}]\n見つかりませんでした。\n")
