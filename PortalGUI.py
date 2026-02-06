import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import subprocess
import queue
import locale
import importlib.util
import webbrowser
from datetime import datetime, timezone
import tkinter.font as tkfont


EVENTS = {
    "研修会": {
        "dir": "研修会",
        "actions": [
            ("GoogleシートURL", "01_PrintURL.py", False),
            ("ダウンロード", "02_DownloadTSV.py", False),
            ("選手集計", "03_SumPlayer.py", False),
            ("(注意)Googleシート再作成", "xx_RemakeFiles.py", True),
        ],
    },
    "春の大会": {
        "dir": "春の大会",
        "actions": [
            ("GoogleシートURL", "01_PrintURL.py", False),
            ("ダウンロード", "02_DownloadTSV.py", False),
            ("選手集計", "03_SumPlayer.py", False),
            ("係員集計", "04_SumStaff.py", False),
            ("トーナメント作成", "05_CreateTournament.py", False),
            ("(注意)Googleシート再作成", "xx_RemakeFiles.py", True),
        ],
    },
    "秋の大会": {
        "dir": "秋の大会",
        "actions": [
            ("GoogleシートURL", "01_PrintURL.py", False),
            ("ダウンロード", "02_DownloadTSV.py", False),
            ("選手集計", "03_SumPlayer.py", False),
            ("係員集計", "04_SumStaff.py", False),
            ("トーナメント作成", "05_CreateTournament.py", False),
            ("(注意)Googleシート再作成", "xx_RemakeFiles.py", True),
        ],
    },
}


class PortalGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KENDO Portal GUI")
        self.geometry("900x650")
        self.minsize(700, 500)

        self.output_queue = queue.Queue()
        self.running = False
        self.last_action_text = ""
        self.last_action_script = ""
        self.last_action_event_name = ""
        self.last_action_event_dir = ""
        self.drive_urls = self._load_drive_urls()
        self.current_drive_url = ""
        self.status_path = ""
        self.pending_output_file_path = ""
        self.latest_update_cache = {}
        self.latest_update_running = False
        self.latest_download_cache = {}
        self.event_image = None

        self._build_ui()
        self._schedule_drive_update()
        self._poll_output()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="大会を選択:").pack(side=tk.LEFT)
        self.event_var = tk.StringVar(value="研修会")
        self.event_radiobuttons = []
        for name in EVENTS.keys():
            rb = ttk.Radiobutton(
                top, text=name, value=name, variable=self.event_var, command=self._refresh_actions
            )
            rb.pack(side=tk.LEFT, padx=6)
            self.event_radiobuttons.append(rb)

        info_frame = ttk.Frame(self, padding=(10, 0, 10, 6))
        info_frame.pack(fill=tk.X)

        self.image_target_width = 120
        self.image_target_height = 120
        self.event_image_canvas = tk.Canvas(
            info_frame,
            width=self.image_target_width,
            height=self.image_target_height,
            highlightthickness=0,
            bd=0,
        )
        self.event_image_canvas.pack(side=tk.LEFT, padx=(0, 10))

        info_right = ttk.Frame(info_frame)
        info_right.pack(side=tk.LEFT, fill=tk.X, expand=True)

        url_frame = ttk.Frame(info_right, padding=(0, 0, 0, 6))
        url_frame.pack(fill=tk.X)
        ttk.Label(url_frame, text="GoogleドライブURL:").pack(side=tk.LEFT)
        self.drive_url_var = tk.StringVar(value="")
        self.drive_url_label = tk.Label(
            url_frame, textvariable=self.drive_url_var, fg="#0078D7", cursor="hand2", anchor="w"
        )
        self.drive_url_label.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)
        self.drive_url_label.bind("<Button-1>", self._open_drive_url)
        self.copy_drive_url_button = ttk.Button(
            url_frame, text="URLコピー", command=self._copy_drive_url
        )
        self.copy_drive_url_button.pack(side=tk.RIGHT)

        update_frame = ttk.Frame(info_right, padding=(0, 0, 0, 6))
        update_frame.pack(fill=tk.X)
        ttk.Label(update_frame, text="Googleシート更新日時:").pack(side=tk.LEFT)
        self.latest_update_var = tk.StringVar(value="")
        self.latest_update_label = ttk.Label(update_frame, textvariable=self.latest_update_var)
        self.latest_update_label.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)

        download_frame = ttk.Frame(info_right, padding=(0, 0, 0, 6))
        download_frame.pack(fill=tk.X)
        ttk.Label(download_frame, text="最終ダウンロード日時:").pack(side=tk.LEFT)
        self.latest_download_var = tk.StringVar(value="")
        self.latest_download_label = ttk.Label(download_frame, textvariable=self.latest_download_var)
        self.latest_download_label.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)

        self.actions_frame = ttk.LabelFrame(self, text="操作", padding=10)
        self.actions_frame.pack(fill=tk.X, padx=10, pady=5)

        output_frame = ttk.LabelFrame(self, text="操作ログ", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        copy_row = ttk.Frame(output_frame)
        copy_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(copy_row, text="コピー", command=self._copy_output).pack(side=tk.RIGHT)
        self.status_label = tk.Label(copy_row, text="実行中…", fg="#0078D7", cursor="")
        self.status_label.pack(side=tk.LEFT)
        self.status_label.pack_forget()
        self.status_label.bind("<Button-1>", self._open_status_path)

        self.output_text = tk.Text(output_frame, wrap=tk.NONE, height=20)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        y_scroll = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.configure(yscrollcommand=y_scroll.set)

        x_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.output_text.xview)
        x_scroll.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.output_text.configure(xscrollcommand=x_scroll.set)

        self.style = ttk.Style(self)
        self.emphasis_style = "Emphasis.TButton"
        default_font = tkfont.nametofont("TkDefaultFont")
        self.emphasis_font = default_font.copy()
        self.emphasis_font.configure(weight="bold")
        self.style.configure(
            self.emphasis_style,
            foreground="#c0392b",
            font=self.emphasis_font,
        )
        self.style.map(
            self.emphasis_style,
            foreground=[("disabled", "#c0392b")],
        )

        self._refresh_actions()

    def _refresh_actions(self):
        for child in self.actions_frame.winfo_children():
            child.destroy()

        event_name = self.event_var.get()
        event = EVENTS[event_name]
        self.action_buttons = {}

        for label, script, dangerous in event["actions"]:
            btn = ttk.Button(
                self.actions_frame,
                text=label,
                command=lambda s=script, e=event, d=dangerous, n=event_name, l=label: self._run_script(
                    e, s, d, n, l
                ),
            )
            btn.pack(fill=tk.X, pady=4)
            self.action_buttons[script] = btn
        self._update_drive_url()
        self._update_latest_update_label()
        self._update_latest_download_label()
        self._update_event_image()
        self._start_drive_update()
        self._refresh_download_button_state()

    def _load_drive_urls(self):
        urls = {}
        for event_name, event in EVENTS.items():
            drive_id = self._read_drive_id(event["dir"])
            if drive_id:
                urls[event_name] = f"https://drive.google.com/drive/folders/{drive_id}"
            else:
                urls[event_name] = ""
        return urls

    def _read_drive_id(self, event_dir):
        defines_path = os.path.join(os.getcwd(), event_dir, "Defines.py")
        try:
            spec = importlib.util.spec_from_file_location(f"Defines_{event_dir}", defines_path)
            if spec is None or spec.loader is None:
                return ""
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, "drive_id", "")
        except Exception:
            return ""

    def _update_drive_url(self):
        url = self.drive_urls.get(self.event_var.get(), "")
        self.current_drive_url = url
        if url:
            self.drive_url_var.set(url)
            self.drive_url_label.config(fg="#0078D7", cursor="hand2")
            self.copy_drive_url_button.config(state=tk.NORMAL)
        else:
            self.drive_url_var.set("URLが見つかりません")
            self.drive_url_label.config(fg="#666666", cursor="")
            self.copy_drive_url_button.config(state=tk.DISABLED)

    def _update_event_image(self):
        event_name = self.event_var.get()
        event_dir = EVENTS[event_name]["dir"]
        image_path = os.path.join(os.getcwd(), event_dir, "image.png")
        self.event_image_canvas.delete("all")

        if not os.path.exists(image_path):
            self.event_image = None
            self.event_image_canvas.create_text(
                self.image_target_width // 2,
                self.image_target_height // 2,
                text="画像なし",
                fill="#666666",
            )
            return

        try:
            image = tk.PhotoImage(file=image_path)
        except Exception:
            self.event_image = None
            self.event_image_canvas.create_text(
                self.image_target_width // 2,
                self.image_target_height // 2,
                text="画像読込失敗",
                fill="#666666",
            )
            return

        w, h = image.width(), image.height()
        if w == 0 or h == 0:
            return

        scale = min(self.image_target_width / w, self.image_target_height / h)
        if scale >= 1:
            factor = max(1, int(scale))
            image = image.zoom(factor, factor)
        else:
            factor = max(1, int(1 / scale))
            image = image.subsample(factor, factor)

        self.event_image = image
        self.event_image_canvas.create_image(
            self.image_target_width // 2,
            self.image_target_height // 2,
            image=self.event_image,
        )

    def _update_latest_update_label(self):
        event_name = self.event_var.get()
        cache = self.latest_update_cache.get(event_name, {})
        text = cache.get("text", "")
        self.latest_update_var.set(text)

    def _update_latest_download_label(self):
        event_name = self.event_var.get()
        cache = self.latest_download_cache.get(event_name, {})
        text = cache.get("text", "")
        self.latest_download_var.set(text)

    def _schedule_drive_update(self):
        self._start_drive_update()
        self.after(60000, self._schedule_drive_update)

    def _start_drive_update(self):
        event_name = self.event_var.get()
        event_dir = EVENTS[event_name]["dir"]
        self._refresh_download_status(event_name, event_dir)
        if self.latest_update_running:
            return
        self.latest_update_running = True
        thread = threading.Thread(
            target=self._worker_drive_update, args=(event_name, event_dir), daemon=True
        )
        thread.start()

    def _worker_drive_update(self, event_name, event_dir):
        text, latest_dt = self._get_latest_modified_text(event_dir)
        self.after(0, lambda: self._apply_drive_update(event_name, text, latest_dt))

    def _apply_drive_update(self, event_name, text, latest_dt):
        self.latest_update_running = False
        self.latest_update_cache[event_name] = {"text": text, "dt": latest_dt}
        if self.event_var.get() == event_name:
            self.latest_update_var.set(text)
        self._refresh_download_button_state()

    def _get_latest_modified_text(self, event_dir):
        context = self._read_drive_context(event_dir)
        if not context:
            return "取得失敗（Defines読込）", None
        drive_id, service_account_path, filename_header, groupnames = context
        if not drive_id:
            return "取得失敗（drive_idなし）", None
        if not service_account_path or not os.path.exists(service_account_path):
            return "取得失敗（認証ファイルなし）", None
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except Exception:
            return "取得失敗（API未設定）", None

        try:
            creds = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
            drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
            expected_names = {
                f"{filename_header}.{name}" for name in groupnames + ["テンプレート", "記入例"]
            }
            latest_dt = None
            page_token = None
            query = (
                f"'{drive_id}' in parents and "
                "mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
            )
            while True:
                results = drive_service.files().list(
                    q=query,
                    fields="nextPageToken, files(id, name, modifiedTime)",
                    pageToken=page_token,
                ).execute()
                for item in results.get("files", []):
                    if item.get("name") not in expected_names:
                        continue
                    mt = item.get("modifiedTime")
                    if not mt:
                        continue
                    try:
                        dt = datetime.fromisoformat(mt.replace("Z", "+00:00"))
                    except Exception:
                        continue
                    if latest_dt is None or dt > latest_dt:
                        latest_dt = dt
                page_token = results.get("nextPageToken")
                if not page_token:
                    break
            if latest_dt is None:
                return "対象ファイルなし", None
            return latest_dt.astimezone().strftime("%Y/%m/%d %H:%M:%S"), latest_dt
        except Exception as exc:
            return f"取得失敗（{type(exc).__name__}）", None

    def _read_drive_context(self, event_dir):
        defines_path = os.path.join(os.getcwd(), event_dir, "Defines.py")
        try:
            spec = importlib.util.spec_from_file_location(f"Defines_{event_dir}_d", defines_path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            drive_id = getattr(module, "drive_id", "")
            service_account_file = getattr(module, "service_account_file", "")
            filename_header = getattr(module, "filename_header", "")
            groupnames = list(getattr(module, "l_groupname", []))
            if service_account_file and not os.path.isabs(service_account_file):
                service_account_file = os.path.join(os.getcwd(), event_dir, service_account_file)
            return drive_id, service_account_file, filename_header, groupnames
        except Exception:
            return None
    
    def _refresh_download_status(self, event_name, event_dir):
        text, oldest_dt = self._get_latest_download_text(event_dir)
        self.latest_download_cache[event_name] = {"text": text, "dt": oldest_dt}
        if self.event_var.get() == event_name:
            self.latest_download_var.set(text)
        self._refresh_download_button_state()

    def _get_latest_download_text(self, event_dir):
        download_folder = self._read_download_folder(event_dir)
        if not download_folder:
            download_folder = ".\\DownloadTSV"
        folder_path = os.path.join(os.getcwd(), event_dir, download_folder)
        if not os.path.isdir(folder_path):
            return "未取得", None
        oldest_ts = None
        for name in os.listdir(folder_path):
            if not name.lower().endswith(".tsv"):
                continue
            path = os.path.join(folder_path, name)
            try:
                ts = os.path.getmtime(path)
            except Exception:
                continue
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
        if oldest_ts is None:
            return "未取得", None
        dt_utc = datetime.fromtimestamp(oldest_ts, tz=timezone.utc)
        return dt_utc.astimezone().strftime("%Y/%m/%d %H:%M:%S"), dt_utc

    def _read_download_folder(self, event_dir):
        defines_path = os.path.join(os.getcwd(), event_dir, "Defines.py")
        try:
            spec = importlib.util.spec_from_file_location(f"Defines_{event_dir}_dlf", defines_path)
            if spec is None or spec.loader is None:
                return ""
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, "download_folder", "")
        except Exception:
            return ""

    def _refresh_download_button_state(self):
        event_name = self.event_var.get()
        btn = self.action_buttons.get("02_DownloadTSV.py")
        if not btn:
            return
        update_dt = self.latest_update_cache.get(event_name, {}).get("dt")
        download_dt = self.latest_download_cache.get(event_name, {}).get("dt")
        needs_download = bool(update_dt and (not download_dt or update_dt > download_dt))
        if needs_download:
            btn.config(text="※ ダウンロード（更新あり）", style=self.emphasis_style)
        else:
            btn.config(text="ダウンロード", style="TButton")

    def _open_drive_url(self, _event=None):
        if not self.current_drive_url:
            messagebox.showinfo("URLなし", "GoogleドライブのURLが見つかりません。")
            return
        webbrowser.open(self.current_drive_url)

    def _copy_drive_url(self):
        if not self.current_drive_url:
            messagebox.showinfo("URLなし", "GoogleドライブのURLが見つかりません。")
            return
        self.clipboard_clear()
        self.clipboard_append(self.current_drive_url)
        self.update_idletasks()

    def _set_buttons_state(self, state):
        for child in self.actions_frame.winfo_children():
            if isinstance(child, ttk.Button):
                child.config(state=state)
        for rb in self.event_radiobuttons:
            rb.config(state=state)
        if state == tk.DISABLED:
            self.status_label.config(text="実行中…")
            self.status_label.config(fg="#0078D7", cursor="")
            self.status_label.pack(side=tk.LEFT)
        else:
            text = self.last_action_text or ""
            if self.status_path:
                text = f"{text} > {self.status_path}"
                self.status_label.config(fg="#0078D7", cursor="hand2")
            else:
                self.status_label.config(fg="#666666", cursor="")
            self.status_label.config(text=text)
            if self.last_action_text:
                self.status_label.pack(side=tk.LEFT)
            else:
                self.status_label.pack_forget()

    def _run_script(self, event, script, dangerous, event_name, label):
        if self.running:
            return
        if dangerous:
            ok = messagebox.askyesno(
                "確認",
                "入力シートを再作成します。元に戻せません。\n実行してよろしいですか？",
            )
            if not ok:
                return

        stdin_text = None
        if script == "05_CreateTournament.py":
            stdin_text = self._prompt_tournament_seed(event["dir"])
            if stdin_text is None:
                return
            self.pending_output_file_path = self._get_tournament_output_path(event["dir"])
        elif script == "03_SumPlayer.py" and event["dir"] == "春の大会":
            self.pending_output_file_path = self._get_spring_booklet_output_path(event["dir"])
        else:
            self.pending_output_file_path = ""

        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, f"\n$ {event['dir']}\\{script}\n")
        self.output_text.configure(state=tk.DISABLED)
        self.output_text.see(tk.END)

        self.last_action_text = f"{event_name} > {label}"
        self.last_action_script = script
        self.last_action_event_name = event_name
        self.last_action_event_dir = event["dir"]
        self.running = True
        self._set_buttons_state(tk.DISABLED)

        thread = threading.Thread(
            target=self._worker_run, args=(event["dir"], script, stdin_text), daemon=True
        )
        thread.start()

    def _worker_run(self, event_dir, script, stdin_text):
        try:
            cmd = [sys.executable, "-u", script]
            proc = subprocess.Popen(
                cmd,
                cwd=os.path.join(os.getcwd(), event_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin_text is not None else None,
                text=True,
                encoding=locale.getpreferredencoding(False),
                errors="replace",
                bufsize=1,
            )
            if stdin_text is not None:
                proc.stdin.write(stdin_text)
                proc.stdin.close()
            for line in proc.stdout:
                self.output_queue.put(line)
            proc.wait()
            self.output_queue.put(f"\n[終了コード] {proc.returncode}\n")
        except Exception as exc:
            self.output_queue.put(f"\n[エラー] {exc}\n")
        finally:
            self.output_queue.put(None)

    def _poll_output(self):
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item is None:
                    self.running = False
                    self._append_output_file_path()
                    self._set_buttons_state(tk.NORMAL)
                    if self.last_action_script == "02_DownloadTSV.py":
                        self._refresh_download_status(
                            self.last_action_event_name, self.last_action_event_dir
                        )
                    continue
                self.output_text.configure(state=tk.NORMAL)
                self.output_text.insert(tk.END, item)
                self.output_text.configure(state=tk.DISABLED)
                self.output_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self._poll_output)

    def _append_output_file_path(self):
        if not self.pending_output_file_path:
            self.status_path = ""
            return
        self.status_path = self.pending_output_file_path

    def _open_status_path(self, _event=None):
        if not self.status_path:
            return
        folder = os.path.dirname(self.status_path)
        if folder and os.path.exists(folder):
            os.startfile(folder)

    def _copy_output(self):
        text = self.output_text.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_idletasks()

    def _prompt_tournament_seed(self, event_dir):
        last_seed, last_time = self._read_last_tournament_seed(event_dir)

        dialog = tk.Toplevel(self)
        dialog.title("トーナメント作成")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=f"前回の抽選結果番号: {last_seed}").pack(anchor=tk.W)
        ttk.Label(frame, text=f"（発行日時: {last_time}）").pack(anchor=tk.W, pady=(0, 10))

        choice = tk.StringVar(value="previous")
        entry_var = tk.StringVar(value="")

        def on_choice_change():
            state = tk.NORMAL if choice.get() == "manual" else tk.DISABLED
            entry.configure(state=state)

        ttk.Radiobutton(frame, text="前回の抽選結果番号を使用", value="previous", variable=choice, command=on_choice_change).pack(anchor=tk.W)
        ttk.Radiobutton(frame, text="新しい抽選結果番号（ランダム）で作成", value="new", variable=choice, command=on_choice_change).pack(anchor=tk.W, pady=(2, 2))

        manual_row = ttk.Frame(frame)
        manual_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Radiobutton(manual_row, text="抽選結果番号を指定", value="manual", variable=choice, command=on_choice_change).pack(side=tk.LEFT)
        entry = ttk.Entry(manual_row, width=20, textvariable=entry_var, state=tk.DISABLED)
        entry.pack(side=tk.LEFT, padx=(8, 0))

        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, pady=(12, 0))

        result = {"text": None}

        def on_ok():
            if choice.get() == "previous":
                result["text"] = "\n"
            elif choice.get() == "new":
                result["text"] = "new\n"
            else:
                value = entry_var.get().strip()
                if not value.isdigit():
                    messagebox.showerror("入力エラー", "数字を指定する場合は半角数字のみで入力してください。", parent=dialog)
                    return
                result["text"] = value + "\n"
            dialog.destroy()

        def on_cancel():
            result["text"] = None
            dialog.destroy()

        ttk.Button(button_row, text="OK", command=on_ok).pack(side=tk.RIGHT)
        ttk.Button(button_row, text="キャンセル", command=on_cancel).pack(side=tk.RIGHT, padx=(0, 6))

        on_choice_change()
        # ダイアログを親ウィンドウ中央に配置（親が最小化等の時は画面中央）
        dialog.update_idletasks()
        parent_w = self.winfo_width()
        parent_h = self.winfo_height()
        dialog_w = dialog.winfo_width()
        dialog_h = dialog.winfo_height()

        if parent_w <= 1 or parent_h <= 1 or not self.winfo_ismapped():
            screen_w = dialog.winfo_screenwidth()
            screen_h = dialog.winfo_screenheight()
            x = max(0, (screen_w - dialog_w) // 2)
            y = max(0, (screen_h - dialog_h) // 2)
        else:
            parent_x = self.winfo_rootx()
            parent_y = self.winfo_rooty()
            x = parent_x + max(0, (parent_w - dialog_w) // 2)
            y = parent_y + max(0, (parent_h - dialog_h) // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.wait_window()
        return result["text"]

    def _read_last_tournament_seed(self, event_dir):
        logfile = os.path.join(os.getcwd(), event_dir, "tournament_no.log")
        last_seed = "100000000"
        last_time = "2000/01/01 00:00:00"
        try:
            with open(logfile, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",", 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        last_seed, last_time = parts[0], parts[1]
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return last_seed, last_time

    def _get_tournament_output_path(self, event_dir):
        tournament_folder = self._read_tournament_folder(event_dir)
        if not tournament_folder:
            tournament_folder = ".\\TournamentFiles"
        filename = ""
        if event_dir == "春の大会":
            filename = "Tournament_春.xlsx"
        elif event_dir == "秋の大会":
            filename = "Tournament_秋.xlsx"
        if not filename:
            return ""
        return os.path.join(os.getcwd(), event_dir, tournament_folder, filename)

    def _read_tournament_folder(self, event_dir):
        defines_path = os.path.join(os.getcwd(), event_dir, "Defines.py")
        try:
            spec = importlib.util.spec_from_file_location(f"Defines_{event_dir}_t", defines_path)
            if spec is None or spec.loader is None:
                return ""
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, "tournament_folder", "")
        except Exception:
            return ""

    def _get_spring_booklet_output_path(self, event_dir):
        if event_dir != "春の大会":
            return ""
        folder = "BookletFiles"
        filename = "PlayerList_春.xlsx"
        return os.path.join(os.getcwd(), event_dir, folder, filename)


if __name__ == "__main__":
    app = PortalGUI()
    app.mainloop()
