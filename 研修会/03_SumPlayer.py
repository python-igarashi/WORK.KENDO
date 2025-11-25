from google.oauth2 import service_account
from googleapiclient.discovery import build
import gspread
import time

import Defines

SCOPES = [
 'https://www.googleapis.com/auth/drive.readonly',
 'https://www.googleapis.com/auth/spreadsheets',
 ]


# ========= 認証とサービス初期化 ========= #
global creds, drive_service, gc
creds = service_account.Credentials.from_service_account_file(Defines.service_account_file, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)
gc = gspread.authorize(creds)


# ========= 名称（集計ファイルのシート名）を指定して集計を行うメソッド ========= #
def Summary(summary_name):
	print(f"----- {summary_name}: 集計開始 -----")
	summary = [] # ここに集計データを作成する
	
	indexof_groupname = 2 # 団体名の列インデックス（"ふりがな" の後ろに "団体名" を挿入する）
	col_length = 15 # サマリの列数
	
	
	# ========= テンプレートからヘッダを取得 ========= #
	# テンプレート_審判係員.tsv のデータを取得
	tsv_filename = f"{Defines.download_folder}\\テンプレート_{summary_name}.tsv"
	l_row = Defines.tsv_get_all_values(tsv_filename)
	for row in l_row:
		if row[1] == "氏名":
			header = row[1 : col_length] # 特定の列まで切り取る
			header.insert(indexof_groupname, "団体名") # "ふりがな" の後ろに "団体名" を挿入する
			summary.append(Defines.pad_list(header, col_length)) # サマリの列数（リストの要素数）が col_length に一致するように調整
			break
	
	
	# ========= 各団体ファイルからデータを集計 ========= #
	#for groupname in ["記入例"]: # テスト用
	for groupname in Defines.l_groupname:
		print(f"{groupname}: 処理中...")
		
		# {団体名}_{集計対象}.tsv のデータを取得
		tsv_filename = f"{Defines.download_folder}\\{groupname}_{summary_name}.tsv"
		l_row = Defines.tsv_get_all_values(tsv_filename)
		
		start_summary = False
		for row in l_row:
			if row[1] == "氏名": # ヘッダ行が見つかったところで集計開始
				start_summary = True
				continue
			
			if row[1] == None or row[1] == "": # 氏名の記入が無い場合はcontinue
				continue
			
			if start_summary != True: # 集計開始前はcontinue
				continue
			
			# "ふりがな" の後ろに "団体名" を挿入する
			value = row[1: col_length] # 特定の列まで切り取る
			value.insert(indexof_groupname, groupname) # "ふりがな" の後ろに "団体名" を挿入する
			summary.append(Defines.pad_list(value, col_length)) # サマリの列数（リストの要素数）が col_length に一致するように調整
	
	
	# ========= 集計スプレッドシートへ出力 ========= #
	print(f"{summary_name}: 出力中...")
	sheet_dst = gc.open_by_url(Defines.url_summary).worksheet(summary_name)
	sheet_dst.clear()
	sheet_dst.update(range_name = f"B1:P{len(summary)}", values = summary)
	
	
	# ========= テスト ========= #
	#print("----- " + summary_name + " -----")
	#for row in summary:
	#	print(row)
	
	print(f"{summary_name}: 集計を終了しました。")


# ========= 名称（集計ファイルのシート名）を指定して集計を行う ========= #
Summary("一般")
Summary("小学生")
Summary("中学生")
