# coding: cp932
import Defines


# ========= 大会プログラム冊子の選手一覧作成用クラス ========= #
class booklet:
	def __init__(self, player_num, summary_name):
		self.player_num = player_num
		self.summary_name = summary_name
		self.team_index = -1
		self.current_row = []
		self.summary = []
		self.teams_per_line = 3 # 一覧で横に並べるチーム数
	
	# ========= チームデータを追加する ========= #
	def append_team(self, team):
		self.team_index += 1
		
		team_offset_Y = self.team_index // self.teams_per_line
		team_offset_X = self.team_index % self.teams_per_line
		
		# 1チームあたりの行数列数を取得
		num_rows = 1 + 1 + (self.player_num * 2) # 行数 = 1[団体] + 1[監督] + ({選手数} * 2[選手ふりがな + 選手氏名])
		num_cols = 3 # ポジション + 名前 + 学年|段位
		
		# 1チームの内容保存用行列を初期化
		contents = [[""] * num_cols for i in range(num_rows)] # Empty文字の2次元配列を作成する
		
		# 一覧の左端のチームの場合、summaryに横に並べるチーム数分の行列を追加する
		if team_offset_X == 0:
			self.summary += [[""] * num_cols * self.teams_per_line for i in range(num_rows)]
		
		summary_offset_Y = team_offset_Y * num_rows
		summary_offset_X = team_offset_X * num_cols
		
		# 団体行
		row_cur = 0
		self.summary[row_cur + summary_offset_Y][0 + summary_offset_X] = "団体名"
		self.summary[row_cur + summary_offset_Y][1 + summary_offset_X] = Defines.get_booklet_groupname(team[0][0], self.summary_name)
		
		# 監督行
		row_cur += 1
		self.summary[row_cur + summary_offset_Y][0 + summary_offset_X] = "監督"
		self.summary[row_cur + summary_offset_Y][1 + summary_offset_X] = team[1][2]
		
		# 選手行
		for i in range(self.player_num):
			# ふりがな行
			row_cur += 1
			self.summary[row_cur + summary_offset_Y][1 + summary_offset_X] = team[2 + i][3] # ふりがな
			
			# 氏名行
			row_cur += 1
			self.summary[row_cur + summary_offset_Y][0 + summary_offset_X] = team[2 + i][1] # ポジション(先鋒, ...)
			self.summary[row_cur + summary_offset_Y][1 + summary_offset_X] = team[2 + i][2] # 氏名(先鋒, ...)
			self.summary[row_cur + summary_offset_Y][2 + summary_offset_X] = team[2 + i][7 if self.summary_name.find("一般") >= 0 else 5] + ("" if self.summary_name.find("一般") >= 0 else "年")  # 学年/段位
	
	# ========= 全てのチームをTSV形式で出力する ========= #
	def output_tsv(self, filepath):
		import csv
		with open(filepath, mode="w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f, delimiter="\t")
			writer.writerows(self.summary)
