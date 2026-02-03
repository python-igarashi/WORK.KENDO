import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import subprocess
import queue
import locale


EVENTS = {
    "研修会": {
        "dir": "研修会",
        "actions": [
            ("URLを出力", "01_PrintURL.py", False),
            ("ダウンロード", "02_DownloadTSV.py", False),
            ("選手集計", "03_SumPlayer.py", False),
            ("(注意)入力シート再作成", "xx_RemakeFiles.py", True),
        ],
    },
    "春の大会": {
        "dir": "春の大会",
        "actions": [
            ("URLを出力", "01_PrintURL.py", False),
            ("ダウンロード", "02_DownloadTSV.py", False),
            ("選手集計", "03_SumPlayer.py", False),
            ("係員集計", "04_SumStaff.py", False),
            ("トーナメント作成", "05_CreateTournament.py", False),
            ("(注意)入力シート再作成", "xx_RemakeFiles.py", True),
        ],
    },
    "秋の大会": {
        "dir": "秋の大会",
        "actions": [
            ("URLを出力", "01_PrintURL.py", False),
            ("ダウンロード", "02_DownloadTSV.py", False),
            ("選手集計", "03_SumPlayer.py", False),
            ("係員集計", "04_SumStaff.py", False),
            ("トーナメント作成", "05_CreateTournament.py", False),
            ("(注意)入力シート再作成", "xx_RemakeFiles.py", True),
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

        self._build_ui()
        self._poll_output()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="大会を選択:").pack(side=tk.LEFT)
        self.event_var = tk.StringVar(value="研修会")
        for name in EVENTS.keys():
            ttk.Radiobutton(
                top, text=name, value=name, variable=self.event_var, command=self._refresh_actions
            ).pack(side=tk.LEFT, padx=6)

        self.actions_frame = ttk.LabelFrame(self, text="操作", padding=10)
        self.actions_frame.pack(fill=tk.X, padx=10, pady=5)

        output_frame = ttk.LabelFrame(self, text="標準出力", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        copy_row = ttk.Frame(output_frame)
        copy_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(copy_row, text="コピー", command=self._copy_output).pack(side=tk.RIGHT)
        self.status_label = ttk.Label(copy_row, text="実行中…", foreground="#0078D7")
        self.status_label.pack(side=tk.LEFT)
        self.status_label.pack_forget()

        self.output_text = tk.Text(output_frame, wrap=tk.NONE, height=20)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        y_scroll = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.configure(yscrollcommand=y_scroll.set)

        x_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.output_text.xview)
        x_scroll.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.output_text.configure(xscrollcommand=x_scroll.set)

        self._refresh_actions()

    def _refresh_actions(self):
        for child in self.actions_frame.winfo_children():
            child.destroy()

        event_name = self.event_var.get()
        event = EVENTS[event_name]

        for label, script, dangerous in event["actions"]:
            btn = ttk.Button(
                self.actions_frame,
                text=label,
                command=lambda s=script, e=event, d=dangerous, n=event_name, l=label: self._run_script(
                    e, s, d, n, l
                ),
            )
            btn.pack(fill=tk.X, pady=4)

    def _set_buttons_state(self, state):
        for child in self.actions_frame.winfo_children():
            if isinstance(child, ttk.Button):
                child.config(state=state)
        if state == tk.DISABLED:
            self.status_label.config(text="実行中…")
            self.status_label.pack(side=tk.LEFT)
        else:
            self.status_label.config(text=self.last_action_text or "")
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

        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, f"\n$ {event['dir']}\\{script}\n")
        self.output_text.configure(state=tk.DISABLED)
        self.output_text.see(tk.END)

        self.last_action_text = f"{event_name} > {label}"
        self.running = True
        self._set_buttons_state(tk.DISABLED)

        thread = threading.Thread(
            target=self._worker_run, args=(event["dir"], script, stdin_text), daemon=True
        )
        thread.start()

    def _worker_run(self, event_dir, script, stdin_text):
        try:
            cmd = [sys.executable, script]
            proc = subprocess.Popen(
                cmd,
                cwd=os.path.join(os.getcwd(), event_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin_text is not None else None,
                text=True,
                encoding=locale.getpreferredencoding(False),
                errors="replace",
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
                    self._set_buttons_state(tk.NORMAL)
                    continue
                self.output_text.configure(state=tk.NORMAL)
                self.output_text.insert(tk.END, item)
                self.output_text.configure(state=tk.DISABLED)
                self.output_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self._poll_output)

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


if __name__ == "__main__":
    app = PortalGUI()
    app.mainloop()
