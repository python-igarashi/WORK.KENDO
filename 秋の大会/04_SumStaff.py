# coding: cp932
from google.oauth2 import service_account
from googleapiclient.discovery import build
import gspread

import Defines

SCOPES = [
 'https://www.googleapis.com/auth/spreadsheets',
 ]


# ========= 認証とサービス初期化 ========= #
creds = service_account.Credentials.from_service_account_file(Defines.service_account_file, scopes=SCOPES)
gc = gspread.authorize(creds)


# ========= 審判係員集計データを初期化 ========= #
dic_summary = {}
for summary_name in Defines.summary_staff: # 集計名称でループ
	dic_summary[summary_name] = []

sheetname_src = "審判係員"
col_length = 10


# ========= テンプレートからヘッダを取得 ========= #
# テンプレート_審判係員.tsv のデータを取得
tsv_filename = f"{Defines.download_folder}\\テンプレート_{sheetname_src}.tsv"
l_row = Defines.tsv_get_all_values(tsv_filename)
for row in l_row:
	summary_name = row[0]
	if summary_name in dic_summary: # A列の値が集計名称に一致した場合
		summary = dic_summary[summary_name]
		header = ["団体名"] + row[1 : col_length] # 一番左を団体名とし、それ以降に行データを追加
		summary.append(Defines.pad_list(header, col_length)) # サマリの列数（リストの要素数）が col_length に一致するように調整


# ========= 全団体のファイルからデータを集計 ========= #
#for groupname in ["記入例"]: # テスト用
for groupname in Defines.l_groupname:
	print(f"{groupname}: 処理中...")
	
	# {団体名}_審判係員.tsv のデータを取得
	tsv_filename = f"{Defines.download_folder}\\{groupname}_{sheetname_src}.tsv"
	l_row = Defines.tsv_get_all_values(tsv_filename)
	
	# データを加工しながら集計
	summary_name = ""
	for row in l_row:
		summary_name = row[0] if row[0] != None and row[0] != "" else summary_name # A列の値から集計名称を取得
		
		if row[1] == None or row[1] == "" or row[1] == "氏名": # 氏名の記入が無い、もしくはヘッダ行の場合はcontinue
			continue
		
		if not summary_name in dic_summary: # 集計名称が集計対象でなければcontinue
			continue
		
		summary = dic_summary[summary_name]
		value = [groupname] + row[1 : col_length] # 一番左を団体名とし、それ以降に行データを追加
		summary.append(Defines.pad_list(value, col_length)) # サマリの列数（リストの要素数）が col_length に一致するように調整


# ========= 集計スプレッドシートへ出力 ========= #
ss_dst = gc.open_by_url(Defines.url_summary)
for summary_name in Defines.summary_staff:
	print(f"{summary_name}: 出力中...")
	l_row = dic_summary[summary_name]
	sheet_dst = ss_dst.worksheet(summary_name)
	sheet_dst.clear()
	sheet_dst.update(range_name = f"A1:J{len(l_row)}", values = l_row)


# ========= テスト ========= #
#for summary_name in Defines.summary_staff:
#	l_row = dic_summary[summary_name]
#	print("----- " + summary_name + " -----")
#	for row in l_row:
#		print(row)
