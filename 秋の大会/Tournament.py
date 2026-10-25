"""団体ごとに山を二分して配置するトーナメント作成。

同じ入力とseedで同じ抽選を再現する。旧版との抽選結果の一致は求めない。
各団体と不戦勝を、上下、4分割、8分割…の各区間へ均等に分ける。
同人数の団体や同条件の配置先は抽選する。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
import os
import random

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Side


@dataclass(frozen=True)
class Participant:
	name: str
	kana: str | None
	groupname: str | None


@dataclass(frozen=True)
class Ref:
	"""前ラウンドの試合の勝者を参照する。試合番号は0始まり。"""
	match_index: int


Slot = Participant | Ref


@dataclass
class Match:
	left: Slot
	right: Slot
	winner: Participant | None = None


Round = list[Match]
Pair = tuple[Participant, Participant]
BYE = Participant("BYE", "BYE", "-")


def _is_bye(player) -> bool:
	return isinstance(player, Participant) and player.name == "BYE"


def _same_group(left: Participant, right: Participant) -> bool:
	return bool(left.groupname) and left.groupname == right.groupname


def _divide_players(players: list[Participant], rng: random.Random):
	"""各団体を半分ずつ分け、奇数の余りを空きの多い側へ振り分ける。"""
	groups = defaultdict(list)
	for player in players:
		# 所属がない参加者は、それぞれ独立した1人の団体として扱う。
		groups[player.groupname or object()].append(player)
	memberships = list(groups.values())
	rng.shuffle(memberships)
	memberships.sort(key=len, reverse=True)

	left, right = [], []
	for members in memberships:
		rng.shuffle(members)
		half = len(members) // 2
		left.extend(members[:half])
		right.extend(members[half:2 * half])

	for members in memberships:
		if len(members) % 2:
			left_space = len(players) // 2 - len(left)
			right_space = (len(players) + 1) // 2 - len(right)
			if left_space > right_space or (left_space == right_space and rng.randrange(2) == 0):
				left.append(members[-1])
			else:
				right.append(members[-1])
	return left, right


def _four_slot_options(players: list[Participant], rng: random.Random) -> list[list[Participant]]:
	"""2試合分を作る。3人なら、同団体の初戦を避けられる不戦勝者を列挙する。"""
	if len(players) == 2:
		return [[players[0], BYE, players[1], BYE]]
	if len(players) == 3:
		options = []
		for index, player in enumerate(players):
			opponents = players[:index] + players[index + 1:]
			options.append([player, BYE, *opponents])
		without_conflict = [option for option in options if not _same_group(option[2], option[3])]
		return without_conflict or options
	left, right = _divide_players(players, rng)
	return [left + right]


def make_first_round_pairs(participants: list[Participant], seed: int | None = None) -> list[Pair]:
	"""山を二分し続けて配置する。候補全体の作り直しや重み付き評価は行わない。"""
	if len(participants) < 2:
		raise ValueError("トーナメントには2人（2チーム）以上の参加者が必要です。")
	rng = random.Random(seed)

	def bye_conflicts(slots):
		# 3回戦までの同じ山で、不戦勝者同士が同団体に偏る数。
		groups = Counter(
			player.groupname for player, opponent in zip(slots[::2], slots[1::2])
			if _is_bye(opponent) and player.groupname
		)
		return sum(count * (count - 1) // 2 for count in groups.values())

	def place(players, slot_count):
		if slot_count == 2:
			rng.shuffle(players)
			return players + [BYE] * (2 - len(players))
		if slot_count == 4:
			return rng.choice(_four_slot_options(players, rng))
		left, right = _divide_players(players, rng)
		if slot_count == 8:
			# 各半分に3人いる場合でも、選択肢は最大3×3通りだけ。
			# 同条件なら抽選し、不戦勝者が同団体に偏る配置を避ける。
			options = [
				first + second for first, second in product(
					_four_slot_options(left, rng), _four_slot_options(right, rng),
				)
			]
			rng.shuffle(options)
			return min(options, key=bye_conflicts)
		return place(left, slot_count // 2) + place(right, slot_count // 2)

	slot_count = 1 << (len(participants) - 1).bit_length()
	players = list(participants)
	rng.shuffle(players)
	slots = place(players, slot_count)
	return list(zip(slots[::2], slots[1::2]))


def build_full_bracket(participants: list[Participant], seed: int | None = None) -> list[Round]:
	"""1回戦を抽選し、各試合の勝者を次のラウンドへ接続する。"""
	first_round = []
	for left, right in make_first_round_pairs(participants, seed):
		winner = left if _is_bye(right) else right if _is_bye(left) else None
		first_round.append(Match(left, right, winner))
	rounds = [first_round]
	while len(rounds[-1]) > 1:
		previous = rounds[-1]
		next_round = [
			Match(previous[index].winner or Ref(index), previous[index + 1].winner or Ref(index + 1))
			for index in range(0, len(previous), 2)
		]
		rounds.append(next_round)
	return rounds


def print_bracket(rounds: list[Round]) -> None:
	"""1回戦の組み合わせを表示する。"""
	def slot_text(slot):
		if isinstance(slot, Participant):
			return f"{slot.name} ({slot.groupname or '-'})"
		return f"Winner of Match {slot.match_index + 1}"

	if rounds:
		for index, match in enumerate(rounds[0], 1):
			print(f"Match {index}: {slot_text(match.left)}  vs  {slot_text(match.right)}")


def save_bracket_xlsx(
	rounds: list[Round],
	path="bracket.xlsx",
	init_workbook=False,
	hide_groupname=False,
	summary_name="xxxxxxxxの部",
	match_name="xxxxxxxxの部",
	match_date="2099.12.31",
	match_place1="第一試合場",
	match_place2="第二試合場",
	playername_formatter=None,
	groupname_formatter=None,
):
	"""部門情報、参加者、対戦線の順に、旧版と同じセル配置で出力する。"""
	if init_workbook:
		try:
			os.remove(path)
			print(f"[{path}] を削除しました。")
		except FileNotFoundError:
			pass
		workbook = Workbook()
		sheet = workbook.create_sheet(summary_name)
		for name in workbook.sheetnames:
			if name != summary_name:
				workbook.remove(workbook[name])
		workbook.save(path)
		print(f"[{path}] を作成しました。")
	else:
		workbook = load_workbook(path)
		if summary_name in workbook.sheetnames:
			workbook.remove(workbook[summary_name])
		sheet = workbook.create_sheet(summary_name)
		workbook.save(path)

	sheet.title = summary_name
	for column, scale in (("A", 1 / 4), ("B", 1 / 3), ("C", 2), ("D", 1.25)):
		sheet.column_dimensions[column].width *= scale
	if hide_groupname:
		sheet.column_dimensions["D"].hidden = True
	line = Side(style="thin")

	def is_real(slot):
		return isinstance(slot, Participant) and slot.name != "BYE"

	# 部門情報と回戦の見出し。参加者の開始位置は6行目。
	sheet.cell(1, 1, match_name)
	for row, value in enumerate((match_date, match_place1, match_place2), 2):
		sheet.cell(row, 2, value)
	for index in range(len(rounds)):
		label = "決勝" if index == len(rounds) - 1 else (
			"準決勝" if index == len(rounds) - 2 else f"{index + 1}回戦"
		)
		sheet.cell(5, index + 5, label).alignment = Alignment(horizontal="right")

	# 不戦勝を除いて選手番号と行位置を決める。1人につき2行を使う。
	players = [
		player for match in rounds[0] for player in (match.left, match.right)
		if is_real(player)
	]
	player_numbers = {player: number for number, player in enumerate(players, 1)}
	player_rows = {player: 6 + (number - 1) * 2 for player, number in player_numbers.items()}
	for player in players:
		number = player_numbers[player]
		row = player_rows[player]
		for column, horizontal in ((2, "right"), (3, "left"), (4, "left")):
			sheet.merge_cells(start_row=row, start_column=column, end_row=row + 1, end_column=column)
			sheet.cell(row, column).alignment = Alignment(
				horizontal=horizontal, vertical="center", wrap_text=True if column == 3 else None,
			)
		sheet.row_dimensions[row].height = 15 * 1.25
		sheet.row_dimensions[row + 1].height = 15 * 1.25
		name = player.name
		if playername_formatter is not None:
			name = playername_formatter(name, summary_name)
		kana = f"\n({player.kana})" if player.kana is not None and player.kana != "" else ""
		group = player.groupname
		if groupname_formatter is not None and group is not None:
			group = groupname_formatter(group, summary_name)
		sheet.cell(row, 2, number)
		sheet.cell(row, 3, name + kana)
		sheet.cell(row, 4, f"[{group}]")

	# 不戦勝を除く実施試合に、回戦順の通し番号を付ける。
	match_numbers = {}
	for round_index, matches in enumerate(rounds):
		for match_index, match in enumerate(matches):
			if not (_is_bye(match.left) or _is_bye(match.right)):
				match_numbers[round_index, match_index] = len(match_numbers) + 1

	def draw_connector(column, upper_row, lower_row, number):
		"""上下の水平線の間を縦線で結び、その内側に試合番号を置く。"""
		sheet.merge_cells(
			start_row=upper_row + 1, start_column=column,
			end_row=lower_row, end_column=column,
		)
		for row in range(upper_row + 1, lower_row + 1):
			sheet.cell(row, column).border = Border(right=line)
		sheet.cell(upper_row + 1, column).value = number
		sheet.cell(upper_row + 1, column).alignment = Alignment(horizontal="right", vertical="center")

	# 勝者の水平線の直上行を記録し、次の回戦の接続位置として使う。
	winner_rows = {}
	for round_index, matches in enumerate(rounds):
		column = round_index + 5
		for match_index, match in enumerate(matches):
			key = round_index, match_index
			if round_index == 0:
				upper = player_rows.get(match.left) if is_real(match.left) else None
				lower = player_rows.get(match.right) if is_real(match.right) else None
				if upper is not None:
					sheet.cell(upper, column).border = Border(bottom=line)
				if lower is not None:
					sheet.cell(lower + 1, column).border = Border(top=line)
				if upper is not None and lower is not None:
					draw_connector(column, upper, lower, match_numbers.get(key))
					winner_rows[key] = upper + 1
				else:
					winner_rows[key] = upper if upper is not None else lower
			else:
				upper = winner_rows[round_index - 1, match_index * 2]
				lower = winner_rows[round_index - 1, match_index * 2 + 1]
				sheet.cell(upper, column).border = Border(bottom=line)
				sheet.cell(lower + 1, column).border = Border(top=line)
				draw_connector(column, upper, lower, match_numbers.get(key))
				winner_rows[key] = (upper + lower) // 2

	sheet.cell(winner_rows[len(rounds) - 1, 0], len(rounds) + 5).border = Border(bottom=line)
	workbook.save(path)
