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
	    hide_groupname = hide_groupname, init_workbook = init_workbook)
	
	
	print(f"{summary_name}: 集計を終了しました。")
	print("")



# ========= 名称（集計ファイルのシート名）を指定してトーナメント作成を行う ========= #
# seed: 抽選結果番号
#   この値によって、作成されるトーナメント抽選結果が決定される。
#   特定の部門だけ値を変更することもできる。
#   毎回ランダムに作成したい場合はseed=Noneとする。

match_date = "2025.6.8"
seed = 20251026 * 10 + 1

Create("小学生の部",   seed=seed, match_date=match_date, match_place1="第一試合場", init_workbook=True)
Create("中学生の部",   seed=seed, match_date=match_date, match_place1="第二試合場")
Create("一般女子の部", seed=seed, match_date=match_date, match_place1="第一試合場(チーム番号1～4)",  match_place2="第二試合場(チーム番号5～9)")
Create("一般の部",     seed=seed, match_date=match_date, match_place1="第一試合場(チーム番号1～16)", match_place2="第二試合場(チーム番号17～32)")
