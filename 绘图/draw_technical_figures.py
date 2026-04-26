# -*- coding: utf-8 -*-
"""Generate technical scheme figures for the Dunhuang music score project."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "输出"
OUT_DIR.mkdir(exist_ok=True)


def set_chinese_font():
	candidates = [
		"Microsoft YaHei",
		"SimHei",
		"Noto Sans CJK SC",
		"Source Han Sans SC",
		"PingFang SC",
		"Arial Unicode MS",
	]
	available = {f.name for f in font_manager.fontManager.ttflist}
	for name in candidates:
		if name in available:
			plt.rcParams["font.sans-serif"] = [name]
			break
	plt.rcParams["axes.unicode_minus"] = False


def add_box(ax, xy, w, h, text, fc="#f7f9fc", ec="#3b4a5a", lw=1.2, fontsize=10):
	box = FancyBboxPatch(
		xy,
		w,
		h,
		boxstyle="round,pad=0.02,rounding_size=0.035",
		linewidth=lw,
		edgecolor=ec,
		facecolor=fc,
	)
	ax.add_patch(box)
	ax.text(
		xy[0] + w /2,
		xy[1] + h /2,
		text,
		ha="center",
		va="center",
		fontsize=fontsize,
		color="#1f2933",
		linespacing=1.35,
	)
	return box


def add_arrow(ax, start, end, color="#5b6770", lw=1.2, rad=0.0):
	arrow = FancyArrowPatch(
		start,
		end,
		arrowstyle="-|>",
		mutation_scale=12,
		linewidth=lw,
		color=color,
		connectionstyle=f"arc3,rad={rad}",
	)
	ax.add_patch(arrow)
	return arrow


def draw_overall_route():
	fig, ax = plt.subplots(figsize=(13.5,4.2))
	ax.set_xlim(0,13.5)
	ax.set_ylim(0,4.2)
	ax.axis("off")

	boxes = [
		(0.35,2.15,1.65,1.0, "敦煌曲谱\n原始图像"),
		(2.45,2.15,1.75,1.0, "图像预处理\n与版面分析"),
		(4.65,2.15,1.75,1.0, "符号识别\n与结构化表达"),
		(6.85,2.15,1.75,1.0, "缺损修复\n与内容补全"),
		(9.05,2.15,1.75,1.0, "音乐合成\n与可听化表达"),
		(11.25,2.15,1.9,1.0, "AIGC辅助创作\n与传播展示"),
	]
	colors = ["#f4f1ea", "#eef5fb", "#eef5fb", "#f5f3ff", "#f0f7f1", "#fff7ed"]

	for i, (x, y, w, h, text) in enumerate(boxes):
		add_box(ax, (x, y), w, h, text, fc=colors[i], fontsize=10.5)
		if i < len(boxes) -1:
			add_arrow(ax, (x + w +0.08, y + h /2), (boxes[i +1][0] -0.08, y + h /2))

	# support = FancyBboxPatch(
	# 	(0.55,0.65),
	# 	12.25,
	# 	0.65,
	# 	boxstyle="round,pad=0.02,rounding_size=0.03",
	# 	linewidth=1.0,
	# 	edgecolor="#7a8793",
	# 	facecolor="#f8fafc",
	# 	linestyle="--",
	# )
	# ax.add_patch(support)
	# ax.text(
	# 	6.675,
	# 	0.975,
	# 	"人机协同校验 / 专家审校 / 来源标注 /生成边界控制",
	# 	ha="center",
	# 	va="center",
	# 	fontsize=10.5,
	# 	color="#344054",
	# )
	# add_arrow(ax, (6.675,1.35), (6.675,2.05), color="#7a8793", lw=1.0)

	fig.savefig(OUT_DIR / "图1_总体技术路线.svg", bbox_inches="tight")
	fig.savefig(OUT_DIR / "图1_总体技术路线.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


def draw_core_framework():
	fig, ax = plt.subplots(figsize=(10.8,6.6))
	ax.set_xlim(0,10.8)
	ax.set_ylim(0,6.6)
	ax.axis("off")

	add_box(ax, (0.45,4.65),2.15,0.95, "原始曲谱图像 I\n残损掩膜 M\n人工标注 A", fc="#f4f1ea", fontsize=10)
	add_box(ax, (3.1,5.15),2.25,0.68, "图像特征提取", fc="#eef5fb", fontsize=10)
	add_box(ax, (3.1,4.25),2.25,0.68, "版面结构分析", fc="#eef5fb", fontsize=10)
	add_box(ax, (3.1,3.35),2.25,0.68, "谱字/符号检测", fc="#eef5fb", fontsize=10)
	add_box(ax, (6.35,4.25),2.15,0.95, "结构化曲谱表示 S", fc="#f0f7f1", fontsize=10.5)

	add_box(ax, (0.65,1.1),1.95,0.82, "符号识别结果\nŜ", fc="#f8fafc", fontsize=10)
	add_box(ax, (3.0,1.1),1.95,0.82, "修复图像\nÎ", fc="#f8fafc", fontsize=10)
	add_box(ax, (5.35,1.1),1.95,0.82, "补全曲谱符号\nS'", fc="#f8fafc", fontsize=10)
	add_box(ax, (7.7,1.1),1.95,0.82, "音乐事件序列\nE", fc="#f8fafc", fontsize=10)
	add_box(ax, (6.525,0.15),2.3,0.66, "合成音频/生成展示 G", fc="#fff7ed", fontsize=10)

	add_arrow(ax, (2.68,5.12), (3.02,5.48))
	add_arrow(ax, (2.68,5.12), (3.02,4.59))
	add_arrow(ax, (2.68,5.12), (3.02,3.69))
	add_arrow(ax, (5.42,5.49), (6.27,4.97))
	add_arrow(ax, (5.42,4.59), (6.27,4.73))
	add_arrow(ax, (5.42,3.69), (6.27,4.45))

	bus_y =2.62
	bus_x0, bus_x1 =1.62,8.68
	add_arrow(ax, (7.42,4.18), (7.42,bus_y +0.08), color="#5b6770", lw=1.15)
	ax.plot([bus_x0,bus_x1], [bus_y,bus_y], color="#5b6770", linewidth=1.15)
	for x in [1.625,3.975,6.325,8.675]:
		add_arrow(ax, (x,bus_y -0.02), (x,2.0), color="#5b6770", lw=1.15)
	add_arrow(ax, (6.325,1.06), (7.675,0.86), color="#5b6770", lw=1.15)
	add_arrow(ax, (8.675,1.06), (7.825,0.86), color="#5b6770", lw=1.15, rad=-0.03)

	add_box(
		ax,
		(1.05,5.98),
		8.7,
		0.38,
		"识别准确性、修复一致性、音乐结构合理性与来源可追溯性协同约束",
		fc="#ffffff",
		ec="#b8c2cc",
		lw=0.9,
		fontsize=9.5,
	)

	fig.savefig(OUT_DIR / "图2_核心方法框架.svg", bbox_inches="tight")
	fig.savefig(OUT_DIR / "图2_核心方法框架.png", dpi=300, bbox_inches="tight")
	plt.close(fig)


if __name__ == "__main__":
	set_chinese_font()
	draw_overall_route()
	draw_core_framework()
	print(f"Figures saved to: {OUT_DIR}")
