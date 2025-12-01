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
# seed:
#   この値によって作成されるトーナメントが決定されるので、seedの値を変更しないことで、SVGファイルだけを微調整することが可能。
#   毎回ランダムに作成したい場合はseed=Noneとする。

match_date = "2025.10.26"
seed = 20251026 * 10 + 1

Create("小学1･2年生の部",                                        seed=seed, match_date=match_date, match_place1="第一試合場", init_workbook=True)
Create("小学3･4年生の部",                                        seed=seed, match_date=match_date, match_place1="第一試合場")
Create("小学5･6年生の部",                                        seed=seed, match_date=match_date, match_place1="第二試合場")
Create("中学生女子の部",                                         seed=seed, match_date=match_date, match_place1="第二試合場")
Create("中学1年生男子の部",                                      seed=seed, match_date=match_date, match_place1="第三試合場")
Create("中学2･3年生男子の部",                                    seed=seed, match_date=match_date, match_place1="第三試合場")
Create("一般女子5段以下の部", match_name="一般女子五段以下の部", seed=seed, match_date=match_date, match_place1="第一試合場(選手番号1～4)",  match_place2="第二試合場(選手番号5～9)")
Create("一般男子3段以下の部", match_name="一般男子三段以下の部", seed=seed, match_date=match_date, match_place1="第一試合場(選手番号1～16)", match_place2="第二試合場(選手番号17～32)")
Create("一般男子4･5段の部",   match_name="一般男子四･五段の部",  seed=seed, match_date=match_date, match_place1="第三試合場")
Create("一般6･7段の部",       match_name="一般六･七段の部",      seed=seed, match_date=match_date, match_place1="第一試合場(選手番号1～7)",  match_place2="第二試合場(選手番号8～14)")
