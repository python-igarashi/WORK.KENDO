# coding: cp932
from google.oauth2 import service_account
from googleapiclient.discovery import build
import gspread
import time

import Defines
import Booklet

SCOPES = [
 'https://www.googleapis.com/auth/spreadsheets',
 ]


# ========= 認証とサービス初期化 ========= #
global creds, gc
creds = service_account.Credentials.from_service_account_file(Defines.service_account_file, scopes=SCOPES)
gc = gspread.authorize(creds)


# ========= 名称（集計ファイルのシート名）を指定して集計を行うメソッド ========= #
def Summary(summary_name, player_num):
	print(f"----- {summary_name}: 集計開始 -----")
	summary = [] # ここに集計データを作成する
	booklet = Booklet.booklet(player_num, summary_name)
	
	col_length = 10 # サマリの列数
	
	
	# ========= 全団体のファイルからデータを集計 ========= #
	#for groupname in ["記入例"]: # テスト用
	for groupname in Defines.l_groupname:
		print(f"{groupname}: 処理中...")
		
		# {団体名}_{集計対象}.tsv のデータを取得
		tsv_filename = f"{Defines.download_folder}\\{groupname}_{summary_name}.tsv"
		l_row = Defines.tsv_get_all_values(tsv_filename)
		
		# 全行を解析する
		l_team = [] # チームリスト
		team = [] # 解析中のチームデータ
		team_enable = False # 解析中のチームの氏名が記載されていたらTrueにする
		for row in l_row:
			# チームのヘッダ行
			if row[0] != None and row[0] != "" and row[2] == "氏名":
				team_name = f"{groupname} {row[0][:1]}" # "{団体名} {A-Z}" でチーム名を作成
				
				value = row[0: col_length] # 特定の列まで切り取る
				value[0] = team_name # 団体名を付与したチーム名で値を入替
				
				team = [ Defines.pad_list(value, col_length) ] # サマリの列数（リストの要素数）が col_length に一致するように調整
				team_enable = False
				continue
			
			# チームのデータ行
			if team != [] and row[1] != None and row[1] != "": # 監督もしくはポジション名がある場合はデータ行
				if row[2] == None or row[2] == "": # 氏名に記載が無い
					value = row[0: 2] # 氏名の直前列まで切り取る
				else:
					value = row[0: col_length] # 特定の列まで切り取る
				
				value = Defines.pad_list(value, col_length) # サマリの列数（リストの要素数）が col_length に一致するように調整
				if summary_name.find("一般") >= 0: value[7] = value[7].split(' ')[-1] # 一般の部、段位の値を修正。例："[10] 教士七段" を "教士七段" とする。
				
				team.append(value)
				
				# 氏名が記載されていれば集計対象とする
				if row[2] != None and row[2] != "":
					team_enable = True
				
				continue
			
			# チームのデータ行の終了
			if team != [] and (row[1] == None or row[1] == ""):
				# 氏名が記載されていたチームの場合、チームリストに追加する
				if team_enable == True:
					l_team.append(team)
				
				# チームデータを初期化
				team = []
				team_enable = False
				continue
		
		# 全行解析後の処理
		
		# 氏名が記載されていたチームの場合、チームリストに追加する
		if team_enable == True:
			l_team.append(team)
		
		# 団体のチームが1つだけだったら、チーム名末尾のアルファベットを削除
		if len(l_team) == 1:
			team = l_team[0]
			team[0][0] = team[0][0].split(" ")[0] # チーム名の文字列を更新
		
		# チームのデータ行の末尾に空行を追加後、集計対象に追加する
		for team in l_team:
			team.append(Defines.pad_list([], col_length)) # サマリの列数（リストの要素数）が col_length に一致するように調整
			summary = summary + team
			booklet.append_team(team)
	
	
	# ========= 集計スプレッドシートへ出力 ========= #
	print(f"{summary_name}: 出力中...")
	sheet_dst = gc.open_by_url(Defines.url_summary).worksheet(summary_name)
	sheet_dst.clear()
	sheet_dst.update(range_name = f"A1:J{len(summary)}", values = summary)
	
	# ========= テスト ========= #
	#print(f"----- 集計: {summary_name} -----")
	#for row in summary:
	#	print(row)
	
	
	# ========= 大会プログラム冊子の選手一覧を出力 ========= #
	booklet.output_tsv(f"BookletFiles\\{summary_name}.tsv")
	
	# ========= テスト ========= #
	#print(f"----- 大会プログラム: {summary_name} -----")
	#for row in booklet.summary:
	#	print(row)
	
	print(f"{summary_name}: 集計を終了しました。")


# ========= 名称（集計ファイルのシート名）を指定して集計を行う ========= #
Summary("小学生の部", 3)
Summary("中学生の部", 3)
Summary("一般女子の部", 3)
Summary("一般の部", 5)
