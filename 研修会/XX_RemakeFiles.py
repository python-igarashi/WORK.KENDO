from google.oauth2 import service_account
from googleapiclient.discovery import build
import gspread

import Defines
import datetime

SCOPES = [
 'https://www.googleapis.com/auth/drive',
 'https://www.googleapis.com/auth/spreadsheets'
 ]


# ========= 認証とサービス初期化（GoogleCloudアカウントがオーナーとなるファイルを作成。ストレージ制限[0byte]のためエラーとなる。） ========= #
#creds = service_account.Credentials.from_service_account_file(Defines.service_account_file, scopes=SCOPES)
#drive_service = build('drive', 'v3', credentials=creds)
#gc = gspread.authorize(creds)

# ========= 認証とサービス初期化（個人アカウント[chofuchuou@gmail.com] がオーナーとなるファイルを作成。OAuthという特殊な機能を使う。） ========= #
from google_auth_oauthlib.flow import InstalledAppFlow
flow = InstalledAppFlow.from_client_secrets_file(Defines.clients_secrets_file, scopes=SCOPES)
creds = flow.run_local_server(port=0)
drive_service = build('drive', 'v3', credentials=creds)
gc = gspread.authorize(creds)

# ========= テンプレートファイル情報取得 ========= #
template_ss = gc.open_by_url(Defines.url_template)
sheet_header = f"{Defines.sheet_header}.{datetime.datetime.now().year}.{datetime.datetime.now().month}"

# ========= 各団体ファイル作成処理 ========= #
for groupname in Defines.l_groupname:
	print(f"{groupname}: 処理中...")
	
	# コピー先ファイル名生成
	filename_dst = Defines.filename_header + "." + groupname
	
	# 同名ファイルがあれば削除
	query = f"'{Defines.drive_id}' in parents and name = '{filename_dst}' and trashed = false"
	results = drive_service.files().list(q=query, fields="files(id)").execute()
	for file_old in results.get('files', []):
		drive_service.files().update(fileId=file_old['id'], body={'trashed': True}).execute()
	
	# テンプレートのコピーを作成
	file_dst = drive_service.files().copy(
		fileId = template_ss.id,
		body = {'name': filename_dst, 'parents': [Defines.drive_id]}
		).execute()
	
	# コピーされたスプレッドシートを開き、全シートのB4セルに団体名を設定、A1セルに大会名を設定
	groupname_ss = gc.open_by_key(file_dst['id'])
	sheet = groupname_ss.worksheets()[0]
	sheet.update(range_name="B4", values=[[groupname]])
	sheet.update(range_name="A1", values=[[sheet_header]])
