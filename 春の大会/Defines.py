service_account_file = "credentials_driveaccess.json"
clients_secrets_file = "credentials_GoogleAuthKendoApp.json"

# Googleドライブの共有URLから転載する
drive_id = "1UcYAg2aFUIFhStVTRdSixbmQiIANChx7" # ドライブ
#drive_id = "12xet2DKsUfF1oErE7EJO4q8rU_GqJPzZ" # ドライブ 2025.06.08の大会のバックアップ - テスト用

# テンプレートファイル、集計ファイル
url_template = "https://docs.google.com/spreadsheets/d/1SKbhDT1hgG3GLCg45SVAJZ0LMTkonoeHRA6Yis4I7eo/edit?usp=drive_link" # テンプレート
url_summary  = "https://docs.google.com/spreadsheets/d/1GxuRYHmYUfv8k4dlopaOiezKkDgcpiVV0oAJc9u029Q/edit?usp=drive_link" # 集計

# テンプレートからの各団体用ファイル作成の際、この値に団体名を付与したものがファイル名となる
filename_header = "調布市大会(春)"

# テンプレートからの各団体用ファイル作成の際、この値に西暦を扶余したものをA1列に出力する
sheet_header = "調布市大会"

# 団体名のリスト
l_groupname = [
  "中央会",
  "染地",
  "大町",
  "聖武会",
  "深大寺",
  "文荘館",
  "第七機動隊",
  "調布警察",
  "電通大",
  "調布北高",
  "神代高",
  "明治",
  "桐朋女子",
  "アメリカン",
  "ドルトン",
  "調布中",
  "神代中",
  "狛江高",
  "矢野口",
]

# 係員の集計単位（集計ファイルのシート名）
summary_staff = [
  "審判",
  "係員",
  "接待係",
  "本部その他",
]

# 正式の団体名表記を取得するメソッド
def get_formal_groupname(groupname, summary_name):
	# シート名のままにしたい場合はこちら
	#return groupname
	
	# 長い団体名に変換したい場合はこちら（↑をコメントアウトしてこちらを有効にする）
	dic_1 = {
	  "中央会": "中央剣道会",
	  "染地": "染地剣道会",
	  "大町": "大町剣道倶楽部",
	  "深大寺": "深大寺剣道会",
	  "電通大": "電気通信大学",
	  "調布北高": "調布北高等学校",
	  "神代高": "神代高等学校",
	  "ドルトン": "ドルトン東京学園",
	  "アメリカン": "ｱﾒﾘｶﾝｽｸｰﾙｲﾝｼﾞｬﾊﾟﾝ",
	  "調布中": "調布中学校",
	  "神代中": "神代中学校",
	  "狛江高": "狛江高等学校",
	  "矢野口": "矢野口剣道会",
	}
	
	src = groupname.split(" ")[0] # チーム名末尾のアルファベットを削除
	dst = dic_1.get(src)
	if dst != None:
		return groupname.replace(src, dst)
	
	if src == "明治":
		if summary_name.find("中学") >= 0:
			return groupname.replace(src, "明治中学校")
		elif summary_name.find("一般") >= 0:
			return groupname.replace(src, "明治高等学校")
	
	if src == "桐朋女子":
		if summary_name.find("中学") >= 0:
			return groupname.replace(src, "桐朋女子中学校")
		elif summary_name.find("一般") >= 0:
			return groupname.replace(src, "桐朋女子高等学校")
	
	return groupname

# 略式の団体名表記を取得するメソッド
def get_simple_groupname(groupname, summary_name):
	src = groupname.split(" ")[0] # チーム名末尾のアルファベットを削除
	
	if src == "明治":
		if summary_name.find("中学") >= 0:
			return groupname.replace(src, "明治中")
		elif summary_name.find("一般") >= 0:
			return groupname.replace(src, "明治高")
	
	if src == "桐朋女子":
		if summary_name.find("中学") >= 0:
			return groupname.replace(src, "桐朋女子中")
		elif summary_name.find("一般") >= 0:
			return groupname.replace(src, "桐朋女子高")
	
	if src == "アメリカン":
		return groupname.replace(src, "ASIJ")
	
	return groupname

# TSVファイルのフォルダ
download_folder = ".\\DownloadTSV"

# トーナメントデータファイルのフォルダ
tournament_folder = ".\\TournamentFiles"

# トーナメント抽選結果番号ログファイル
tournament_no_logfile = ".\\tournament_no.log"
def read_tournament_no_logfile():
	import csv
	result = []
	try:
		with open(tournament_no_logfile, newline='', encoding='utf-8') as logfile:
			reader = csv.reader(logfile, delimiter=',')
			for row in reader:
				if len(row) == 2 and row[0].isdigit():
					result.append([ int(row[0]), row[1] ]) # [抽選結果番号, 日時]
	except:
		print("抽選結果番号ログファイルの読込をスキップしました。")
		pass
	if len(result) == 0:
		result.append([100000000, "2000/01/01 00:00:00"])
	return result

def save_tournament_no_logfile(seed, seed_time):
	lines = read_tournament_no_logfile()
	if lines[-1][0] != seed:
		lines.append([ seed, seed_time ])
	lines = lines[-10:]
	with open(tournament_no_logfile, "w", encoding="utf-8") as logfile:
		for line in lines:
			logfile.write(f"{line[0]},{line[1]}\n")

# サマリデータの列数を一定に保つためのメソッド
def pad_list(l_value, length):
	return l_value + [""] * max(0, length - len(l_value))

# TSVファイルから二次元配列を返却する
def tsv_get_all_values(tsv_filename):
	import csv

	result = []
	with open(tsv_filename, newline='', encoding='utf-8') as tsvfile:
		reader = csv.reader(tsvfile, delimiter='\t')
		for row in reader:
			# 空欄が None にならないよう、空文字に統一
			result.append([cell if cell is not None else "" for cell in row])
	return result

# トーナメント設定ファイル
tournament_settings_file = ".\\tournament_settings.json"

def load_tournament_settings():
	"""tournament_settings.jsonを読み込む"""
	import json
	try:
		with open(tournament_settings_file, encoding='utf-8') as f:
			data = json.load(f)
			# バリデーション
			if "tournament_date" not in data or "categories" not in data:
				print(f"警告: {tournament_settings_file} に必須フィールドがありません。")
				return None
			if not isinstance(data["categories"], list):
				print(f"警告: {tournament_settings_file} のcategoriesが配列ではありません。")
				return None
			return data
	except FileNotFoundError:
		print(f"情報: {tournament_settings_file} が見つかりません。")
		return None
	except json.JSONDecodeError as e:
		print(f"エラー: {tournament_settings_file} のJSON解析に失敗しました: {e}")
		return None
	except Exception as e:
		print(f"エラー: {tournament_settings_file} の読込に失敗しました: {e}")
		return None

def save_tournament_settings(tournament_date, categories):
	"""tournament_settings.jsonに保存する"""
	import json
	try:
		data = {
			"version": "1.0",
			"tournament_date": tournament_date,
			"categories": categories
		}
		with open(tournament_settings_file, "w", encoding='utf-8') as f:
			json.dump(data, f, ensure_ascii=False, indent=2)
		print(f"トーナメント設定を保存しました: {tournament_settings_file}")
		return True
	except Exception as e:
		print(f"エラー: {tournament_settings_file} の保存に失敗しました: {e}")
		return False

def get_default_tournament_settings(event_dir):
	"""デフォルトのトーナメント設定を返す"""
	if event_dir == "春の大会":
		return {
			"version": "1.0",
			"tournament_date": "2025.6.8",
			"categories": [
				{
					"summary_name": "小学生の部",
					"match_name": None,
					"match_place1": "第一試合場",
					"match_place2": ""
				},
				{
					"summary_name": "中学生の部",
					"match_name": None,
					"match_place1": "第二試合場",
					"match_place2": ""
				},
				{
					"summary_name": "一般女子の部",
					"match_name": None,
					"match_place1": "第一試合場(チーム番号1～4)",
					"match_place2": "第二試合場(チーム番号5～9)"
				},
				{
					"summary_name": "一般の部",
					"match_name": None,
					"match_place1": "第一試合場(チーム番号1～16)",
					"match_place2": "第二試合場(チーム番号17～32)"
				}
			]
		}
	else:
		return None
