import Defines
import Tournament


# ========= 名称（集計ファイルのシート名）を指定してトーナメント作成を行うメソッド ========= #
def Create(summary_name, seed=None, print_match_no=True, match_name=None, match_date="", match_place1="", match_place2="", hide_groupname=True, init_workbook=False):
	print(f"----- {summary_name}: トーナメント作成開始 -----")
	l_participant = [] # ここに集計データを作成する
	
	
	# ========= 全団体のファイルからデータを集計 ========= #
	#for groupname in ["記入例"]: # テスト用
	for groupname in Defines.l_groupname:
		print(f"{groupname}: 処理中...")
		
		# {団体名}_{集計対象}.tsv のデータを取得
		tsv_filename = f"{Defines.download_folder}\\{groupname}_{summary_name}.tsv"
		l_row = Defines.tsv_get_all_values(tsv_filename)
		
		# 全行を解析する
		l_team = [] # チーム名リスト
		team = None # 解析中のチーム名
		team_enable = False # 解析中のチームの氏名が記載されていたらTrueにする
		for row in l_row:
			# チームのヘッダ行
			if row[0] != None and row[0] != "" and row[2] == "氏名":
				team_name = f"{groupname} {row[0][:1]}" # "{団体名} {A-Z}" でチーム名を作成
				team = Tournament.Participant(team_name, None, groupname)
				team_enable = False
				continue
			
			# チームのデータ行
			if team != [] and row[1] != None and row[1] != "": # 監督もしくはポジション名がある場合はデータ行
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
				team = None
				team_enable = False
				continue
		
		# 全行解析後の処理
		
		# 氏名が記載されていたチームの場合、チームリストに追加する
		if team_enable == True:
			l_team.append(team)
		
		# 団体のチームが1つだけだったら、チーム名末尾のアルファベットを削除
		if len(l_team) == 1:
			team = l_team[0]
			l_team[0] = Tournament.Participant(team.name.split(" ")[0], team.kana, team.groupname) # チーム名の文字列を更新
		
		# 集計対象に追加する
		l_participant = l_participant + l_team
	
	rounds = Tournament.build_full_bracket(l_participant, seed=seed)
	
	
	# ========= トーナメントのコンソール出力 ========= #
	Tournament.print_bracket(rounds)
	
	
	# ========= トーナメントのEXCELファイルを出力 ========= #
	Tournament.save_bracket_xlsx(
	    rounds, path = f"{Defines.tournament_folder}\\Tournament_春.xlsx", summary_name = summary_name,
	    match_name = match_name if match_name != None else summary_name, match_date = match_date, match_place1 = match_place1, match_place2 = match_place2,
	    hide_groupname = hide_groupname, init_workbook = init_workbook,
	    playername_formatter = Defines.get_formal_groupname,
	    groupname_formatter = None)
	
	
	print(f"{summary_name}: 集計を終了しました。")
	print("")



# ========= 名称（集計ファイルのシート名）を指定してトーナメント作成を行う ========= #
# seed: 抽選結果番号
#   この値によって、作成されるトーナメント抽選結果が決定される。
#   特定の部門だけ値を変更することもできる。
#   毎回ランダムに作成したい場合はseed=Noneとする。
import random
from datetime import datetime

seed, seed_time = Defines.read_tournament_no_logfile()[-1]
print( "------------------")
print(f"現在の抽選結果番号")
print(f"[{seed}]")
print(f"(発行日時 = {seed_time})")
print( "------------------")
print("")
print("A.現在の抽選結果番号のまま実行する場合は、リターンキーを押してください。")
print("B.新しい番号(ランダム)にする場合は、new あるいは NEW と入力後リターンキーを押してください。")
print("C.抽選結果番号を任意に指定する場合は、数値(9桁以内)を入力後リターンキーを押してください。")

user_input = input("> ").strip()
if user_input == "":
	print("現在の抽選結果番号を使用します。")
elif user_input.lower() == "new":
	print("新しい抽選結果番号（ランダム）を生成します。")
	seed = random.randint(100000000, 999999999)
	seed_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
elif len(user_input) <= 9 and user_input.isdigit():
	seed = int(user_input)
	seed_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
else:
	print("入力が不正です。プログラムを終了します。\n")
	import sys
	sys.exit(0)

print(f"抽選結果番号 [{seed}] で実行します。")
Defines.save_tournament_no_logfile(seed, seed_time) # 抽選結果番号をファイルに保存


# トーナメント設定を読み込む
settings = Defines.load_tournament_settings()
if settings is None:
	print("エラー: トーナメント設定ファイルが見つからないか、読み込みに失敗しました。")
	print("デフォルト設定で実行します...")
	import os
	event_dir = os.path.basename(os.getcwd())
	settings = Defines.get_default_tournament_settings(event_dir)
	if settings is None:
		print("エラー: デフォルト設定の取得に失敗しました。プログラムを終了します。")
		import sys
		sys.exit(1)

match_date = settings["tournament_date"]
categories = settings["categories"]

# 各部門のトーナメント作成
for i, category in enumerate(categories):
	summary_name = category["summary_name"]
	match_name = category.get("match_name") or None
	match_place1 = category.get("match_place1", "")
	match_place2 = category.get("match_place2", "")
	init_workbook = (i == 0)  # 最初の部門のみTrue

	Create(
		summary_name,
		seed=seed,
		match_name=match_name,
		match_date=match_date,
		match_place1=match_place1,
		match_place2=match_place2,
		hide_groupname=True,  # 春の大会は常にTrue（チーム戦）
		init_workbook=init_workbook
	)

print(f"抽選結果番号 [{seed}] での抽選を終了しました。")
