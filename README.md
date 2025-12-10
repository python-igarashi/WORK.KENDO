# 調布市剣道連盟 内部大会支援ツール

調布剣道連盟の各種内部大会イベント（研修会／春の大会／秋の大会）向けの Python ツール集です。

## フォルダ構成
- [研修会](./研修会/)
- [春の大会](./春の大会/)
- [秋の大会](./秋の大会/)
- [テスト](./テスト/)（開発・検証用）

## 前提
- 操作手順は各フォルダの README を参照してください。
- Google認証情報ファイル（credentials_*.json）や入出力データ（tsv/xlsx 等）はリポジトリに含まれていません。

## 事前準備（環境構築）  
1. Python 3.13 以降をインストールします。  
   [https://www.python.org/downloads/](https://www.python.org/downloads/)

2. ライブラリのインストール  
 コマンドプロンプトにて、以下のコマンドを実行します（「C:\Python313\Scripts」は、上記にてPythonをインストールしたフォルダに依存します）。  
   cd C:\Python313\Scripts  
   pip install gspread  
   pip install google-auth  
   pip install google-api-python-client  
   pip install openpyxl  

3. Google認証情報ファイルの入手  
 以下のファイルをシステム管理者から入手して下さい。  
   credentials_driveaccess.json  
   credentials_GoogleAuthKendoApp.json  

## 使い方
各フォルダの README を参照してください。
