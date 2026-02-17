import Defines
import Tournament


# ========= 名称（集計ファイルのシート名）を指定してトーナメント作成を行うメソッド ========= #
def Create(summary_name, seed=None, print_match_no=True, match_name=None, match_date="", match_place1="", match_place2="", hide_groupname=False, init_workbook=False):
	print(f"----- {summary_name}: トーナメント作成開始 -----")
	summary = [] # ここに集計データを作成する
	
	indexof_groupname = 2 # 団体名の列インデックス（"ふりがな" の後ろに "団体名" を挿入する）
	col_length = 10 # サマリの列数
	
	
	# ========= 各団体ファイルからデータを集計 ========= #
	#for groupname in ["記入例"]: # テスト用
	for groupname in Defines.l_groupname:
		#print(f"{groupname}: 処理中...")
		
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
	
	
	# ========= トーナメントを作成 ========= #
	#print(f"{summary_name}: 出力中...")
	l_participant = []
	for value in summary:
		#groupname = Defines.get_booklet_groupname(value[indexof_groupname], summary_name)
		groupname = value[indexof_groupname]
		l_participant.append(Tournament.Participant(value[0], value[1], groupname))
	
	rounds = Tournament.build_full_bracket(l_participant, seed=seed)
	
	
	# ========= トーナメントのコンソール出力 ========= #
	Tournament.print_bracket(rounds)
	
	
	# ========= トーナメントのEXCELファイルを出力 ========= #
	Tournament.save_bracket_xlsx(
	    rounds, path = f"{Defines.tournament_folder}\\Tournament_秋.xlsx", summary_name = summary_name,
	    match_name = match_name if match_name != None else summary_name, match_date = match_date, match_place1 = match_place1, match_place2 = match_place2,
	    hide_groupname = hide_groupname, init_workbook = init_workbook)
	
	
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
		hide_groupname=False,  # 秋の大会は常にFalse（個人戦）
		init_workbook=init_workbook
	)

print(f"抽選結果番号 [{seed}] での抽選を終了しました。")
