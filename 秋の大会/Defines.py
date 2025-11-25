service_account_file = "credentials_driveaccess.json"
clients_secrets_file = "credentials_GoogleAuthKendoApp.json"

# Googleドライブの共有URLから転載する
drive_id = "12jbd11itYcVmICCJYiFxiXz10oHyANqt" # ドライブ
#drive_id = "1yYZ2HfFYQyozop20_BfcH30w0j8OrxbL" # ドライブ 2024.10.13の大会のバックアップ - テスト用

# テンプレートファイル、集計ファイル
url_template = "https://docs.google.com/spreadsheets/d/1DTvAV4FUbAgvkbKFnBSA4wSmDnLkYDMsw7KkWsT3IiE/edit?usp=drive_link" # テンプレート
url_summary  = "https://docs.google.com/spreadsheets/d/1AKBk7D6UTkqCs40vUJ1lDFX0Ho41ItDrV8RybufYlWo/edit?usp=drive_link" # 集計

# テンプレートからの各団体用ファイル作成の際、この値に団体名を付与したものがファイル名となる
filename_header = "調布市民大会(秋)"

# テンプレートからの各団体用ファイル作成の際、この値に西暦を扶余したものをA1列に出力する
sheet_header = "調布市民大会"

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
  "個人参加",
]

# 係員の集計単位（集計ファイルのシート名）
summary_staff = [
  "審判",
  "係員",
  "接待係",
  "本部その他",
]

# 大会プログラム冊子用の団体名表記を取得するメソッド
def get_booklet_groupname(groupname, summary_name):
	# 団体名を略称のままにしたい場合はこちら
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

# TSVファイルのフォルダ
download_folder = ".\\DownloadTSV"

# トーナメントデータファイルのフォルダ
tournament_folder = ".\\TournamentFiles"

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
