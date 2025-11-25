service_account_file = "credentials_driveaccess.json"
clients_secrets_file = "credentials_GoogleAuthKendoApp.json"

# Googleドライブの共有URLから転載する
drive_id = "1fJyapu7qwsIUkx20yY-rmK3IZBM06srs" # ドライブ
#drive_id = "1x-kXdsHh68GL88ZdVTNq0rdl4g10jC7v" # ドライブ 2025.05.25の研修会のバックアップ - テスト用

# テンプレートファイル、集計ファイル
url_template = "https://docs.google.com/spreadsheets/d/1LL-lTI0U3lv0GQW2CWaYVHxghBKNgLql4Syrw6i_bP8/edit?usp=drive_link" # テンプレート
url_summary  = "https://docs.google.com/spreadsheets/d/1oy19HAJJFw5g016kwF77tdvq24W-llHACtOBWTbBcls/edit?usp=drive_link" # 集計

# テンプレートからの各団体用ファイル作成の際、この値に団体名を付与したものがファイル名となる
filename_header = "研修会"

# テンプレートからの各団体用ファイル作成の際、この値に西暦を扶余したものをA1列に出力する
sheet_header = "研修会参加者"

# 団体名のリスト
l_groupname = [
  "中央会",
  "染地",
  "大町",
  "聖武会",
  "深大寺",
  "文荘館",
  "電通大",
  "狛江",
]

# 係員の集計単位（集計ファイルのシート名）
summary_staff = [
  "一般",
  "小学生",
  "中学生",
]

# TSVファイルのフォルダ
download_folder = ".\\DownloadTSV"

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
