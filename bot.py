"""Transformice Bot — vision + optional memory hybrid navigation."""
import sys
import os
import time
import ctypes
import cv2
import numpy as np
import psutil
import win32gui
import win32process
from pynput.keyboard import Key, Controller as KBC, Listener

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision import (Vision, save_png, LABEL_TRUST_SCORE, LABEL_TRUST_MARGIN)
from mem_reader import Mem
from _ring import Ring
# 起别名！主循环里 `tgt` 是个局部变量（原来的 `tgt = None`），模块级的 `import tgt`
# 会被它整个遮住，写成 `tgt.pick_target` 就变成对 None/坐标元组取属性。
import tgt as tgt_mod

# 所有日志行加 [HH:MM:SS] 前缀。日志是这个项目唯一的诊断手段，而之前完全没有时间戳：
# 事后查「这次锁定活了多久、标定多久试一次」只能靠数 [PERF] 行的密度倒推，很容易推错。
# 改 builtins 而不是加个 _log() 帮手，是为了连 vision.py 的 [LOCK]/[TRACK] 也一起带上，
# 不必改几十处调用点。
import builtins
_raw_print = builtins.print


def _ts_print(*a, **kw):
    # 控制台是 GBK 代码页，日志里混进一个 GBK 编不出的符号（实测 `⇔`）会直接抛
    # UnicodeEncodeError 把主循环打死——而且是打完成功日志才死，最坑的位置。日志
    # 掉了远不如把字符降级成问号：诊断信息比符号好看重要得多。
    try:
        _raw_print(time.strftime("[%H:%M:%S]"), *a, **kw)
    except UnicodeEncodeError:
        _raw_print(time.strftime("[%H:%M:%S]"),
                   *(str(x).encode("gbk", "replace").decode("gbk") for x in a), **kw)


builtins.print = _ts_print

# ===== 可调参数（集中在此，改这里即可）=====
MANUAL_TIMEOUT = 45          # 按方向键后手动接管时长（秒）
MANUAL_KEY_GAP = 0.3         # 判定「人工按键」的最小间隔（秒），超过才认为不是 bot 发的键
TAP_MS = 30                  # 普通按键默认按下时长（毫秒）
JUMP_TAP_MS = 40             # 跳跃按键按下时长（毫秒）
STUCK_TURN_MS = 100          # 卡住时转向按键时长（毫秒）
STUCK_STREAK_RECHECK = 3     # 连续 N 次脱困都没挪窝 → 复核锁定是不是已经锁在不动的东西上了
STUCK_SAME_DIR = 4           # 脱困时同一个方向连着用 N 次再换向（见下面的说明）

# —— 输入标定：按方向键找出「跟着动的那块像素」= 本地玩家 ——
IDENTITY_HOLD = 0.75         # 标定探针按住方向键的时长（秒）。2026-10-26 从 0.40 加到
                             # 0.75：官方房 40+ 只老鼠挤在一堆，0.4s 常常只挤出去十几像素
                             # 甚至不动，标定就报「没有跟手位移」；按住久一点能挤出人群。
                             # 代价是每次标定多 1s 左右，换的是「标定能成」。
PROBE_SEARCH = 110           # 探针匹配的搜索窗半径。0.40s 实测走 ~40px，但换地图/上坡会更快，
                             # 留足余量。默认 80 的失效方式离线实测过（_off_peak.py，1.0s 位移
                             # ~105px 的帧对）：位移一旦超过窗，真峰要么被推到窗边被 `abs(dx)
                             # >= search-3` 丢掉，要么分数塌到 0.7 以下当「无匹配」静默丢弃
                             # （实测同一块 score 0.466 vs search=110 时 0.726 @ dx=+105）。
                             # 代价：搜索窗变大，可采样区收窄（顶部 y<search 那一带探不到）
IDENTITY_MIN_DX = 12         # 判定「跟着按键移动了」的最小水平位移（像素）
IDENTITY_MIN_SCORE = 0.80    # 模板匹配最低得分，低于此值不算可信匹配
PATCH_SIZE = 48              # 锁定用的模板边长（须与 vision.match_translation 的 patch 一致）
# _find_body_at 是从差分图里量「哪块动过」的，房间里挤一堆老鼠时它会把**整堆**量成一只
# 身体：2026-10-07 23:22:21 实测量出 101x80，随后被当成身体、锁定边长直接顶到 80
# （`size = min(80, max(48, bh+6))`）。80x80 的模板盖住了一大片几乎静止的地形，匹配分
# 一直是 1.00、锁撑了 92 秒却什么都没干（诊断里那条「豁免1/45」）。真精灵量出来是
# 48~54（当晚 5 次好锁：48/52/53/53/54），所以卡在上限 58：超了就说明差分被整堆人占满。
BODY_MAX_W = 72              # 身体框（差分量出来的那个）宽/高上限，超了走「人群里分不出」那条路
BODY_MAX_H = 58
SPRITE_MAX_PX = 60           # 锁定模板边长上限（48~54 是好锁实测值，留一点余量）
REV_SEARCH = 64              # 反向验证时在左移帧里搜多宽（比 IDENTITY_HOLD 的位移大就行）
REV_MIN_SCORE = 0.45         # 反向验证单块最低匹配分：老鼠会动画+镜像，比 IDENTITY_MIN_SCORE 低得多
REV_STRONG_SCORE = 0.80      # 「一块顶两票」的高分门槛：小簇靠单块高分反向过关时用
REV_VOTE_MIN = 2             # 反向验证至少这么多块一致左移才算通过
# ===== 深度学习自定位（_dl/dl_loc.py，2026-10-27 起）=====
# 一帧画面直接回归「老鼠在哪」，**不按任何键**、十几毫秒出结果。
# 存在的理由：老标定要按方向键做位移相关分析，一次 4~5s，这段时间角色站着不动，
# 而用户实测「开局不动的小鼠会死」，挂机死的连萨满的复活技能都救不回来。
# 模型在 corpus 10500 帧上训练，验证集正中位误差 4~20px（800x600 舞台像素）。
DL_LOC = os.environ.get("DL_LOC", "1").strip().lower() not in ("0", "false", "no", "off")   # 无深度学习版：DL_LOC=0
DL_MIN_CONF = 0.30    # 峰值置信度门槛。语料内正确帧峰值中位 0.49~0.58，但**活画面**（万圣节
                      # 夜图、新地图）实测只有 0.32 左右——按 0.55 卡会整局锁不上（10-27 上午
                      # 就是这样）。0.30 配上 dl_loc 里的「时域稳定门」（连续 ≥3 帧落在同一点
                      # ±35px 才认）来兜误锁：单帧 0.3 是噪声，连着几帧同点就不是。
DL_TRY_S = 0.2        # 盲锁期最多多久跑一次（别每帧都跑）
CALIB_TRIES = 3              # 启动时连续尝试标定的次数
CALIB_RETRY_S = 8.0          # 未锁定时每隔多久重试一次（也用于判断「不可操作」）
CALIB_MAX_RETRY_S = 12.0     # 反复「刚锁上就被丢」时退避的上限（见 LOCK_MIN_USEFUL_S）
                             # 2026-10-26 从 60s 降到 12s：一局只有 ~2 分钟，退避 60s
                             # 等于整局报废（20:49 那场标定成功后 4.3s 跟丢，然后一直
                             # NO-LOCK 到回合结束）。现在还有个不按键的「名字标签重挂」
                             # 在退避之前先试，按键标定失败的代价小多了。
LABEL_TRY_S = 0.3            # 没锁定时多久试一次「名字标签重挂」（纯视觉，不按键）
LOCK_MIN_USEFUL_S = 20.0     # 一次锁定至少撑这么久才算「真的用上了」。实测：能玩的回合里
                             # 锁定从 08:15:10 一直跟到回合结束（好几分钟），而不可玩的局里
                             # 每次都只撑 5~8s 就被丢（08:39:09→08:39:14、08:40:41→08:40:49）

CALIB_MAX_CANDIDATES = 8     # 反向验证最多试几个候选簇（背景视差/重复纹理会造出一堆假簇）
LABEL_MIN_SCORE = 0.60       # 自己的金色名字标签逐帧匹配的最低 F1。实测好帧恒为 1.00，
                             # 被别的玩家标签压住时掉到 0.72（_k2_2_holdL.png），0.60 留足余量
LABEL_SEARCH_Y = 44          # 标签匹配的纵向搜索半径：标签只会小幅上下浮动（跳跃/爬坡），
                             # 纵向搜太宽反而会跟下面那行金色称号串味
LABEL_TAG_AGREE = 6          # 把多个命中并成整块标签时，位移跟最佳命中差这么多 px 以内才算
                             # 「同一块文字」。真名字行和称号行的 (右移, 左移) 几乎相同
LABEL_TAG_MAX_DX = 80        # 同组命中的水平间距上限。两行文字是上下叠着的、x 只差几个像素
LABEL_TAG_MAX_DY = 40        # 同组命中的**竖直**间距上限。2026-10-07 加：原来只看水平距离，
                             # 于是别人那只老鼠的名字（竖直差 119~133px）会被并进来 → 并出的
                             # 块高 133/147 > LABEL_TAG_MAX_H 被整块弃用，一次标定就白费。
                             # 实测自己的名字行↔称号行竖直只差 13px（18:58:31 那次的
                             # (211,342)+(215,355) → 36x22 成功），而 11 次失败全是差 119~133px。
LABEL_TAG_MAX_W = 160        # 并出来的标签块尺寸上限。实测好标签 33~98 宽、9~28 高
LABEL_TAG_MAX_H = 16         # 2026-10-26 从 34 降到 16：**只收单行**。34 允许把
                             # 「名字 + 称号 + 旁边的聊天字」并成 44x31 那种三行块，而
                             # 那种块里通常有一块不是跟着按键走的 —— 实测标定因此反复
                             # 锁到别人身上（不对称位移 +52/-110 就是这么来的）。单行
                             # 名字块（36x9、43x9）本来就够当锚点，CCOEFF 匹配实测
                             # 主峰 1.00 / 次峰 0.31，比并成多行还干净。
                             # ——见 _identify_by_label 里为什么必须卡这两条（伪标签 736x36）
LABEL_BODY_GAP = 9           # 金色文字块底边 → 老鼠身体顶边的距离（量自 _k2_0_before.png：
LABEL_BODY_H = 52            # 称号底 295 → 精灵顶 304、脚底 356）。人群里差分分不出哪只
                             # 是自己时用这组常数把身体框估出来：跟踪真正的锚点是名字标签
                             # （静态文字，比精灵稳），这组数只影响上报的脚底 y。
DBG_DUMP_S = 2.0             # 每隔几秒把带标注的一帧存成 _dbg_frame.png（翻图核对误检）

# —— 帧环形缓冲 / 离线语料（见 _ring.py）——
# 默认**关**：DBG_RING 不设 = 0，平时的跑法一行都不多干；采集时置 1。
DBG_SAMPLE_S = float(os.environ.get("DBG_SAMPLE_S", "30"))   # 每 N 秒另落一张单帧
                             # 「背景样本」（不进前后文缓冲）。攒语料时调小（如 15）。
CB_RING_EDGE = 4             # cb_count 升到这个值 → 记一次事件（假阳性簇的起点）
CB_RING_FALL = 2             # cb_count 跌回这个值 → 记一次事件（簇的终点）
NEAR_TARGET_PX = 20          # 人到目标曼哈顿距离 ≤ 此值 → 记一次「贴上去」事件。
# 2026-10-06 加这个触发是为了抓 `has_cheese_on_back` 的**真阳性样本**：
# 要标定它必须看「机器人自己挨着芝士」的帧，而以前根本抓不到 ——
# 三条老日志的「人到目标距离」地板是 33/58/76px，**一次也没到过 20px 以内**。
# 8b 那处（self-pad 18→6 + 已锁定目标豁免）修完之后，实测最小已经到 4px、
# 20px 以内有 30 个采样，所以现在抓得到。等攒够样本再动那个函数。
                             # 为什么用「沿」不用「跃变」：cb 是 15 帧滑窗求和，每帧最多
                             # 变 ±1，单帧跃变 ≥6 永远不会发生（2026-10-05 设计时差点写错）。

# —— 目标选择迟滞（改动一，核心逻辑在 tgt.py）——
# 改前这里是 `tgt = min(reach, key=最近距离)`：**无记忆的纯函数**，每帧对候选集
# 重算最近。两块芝士距离接近时必然逐帧翻转 → 净位移趋近 0 → 永远走不到任何一块
# 跟前。2026-10-05 基线实测：相邻采样「目标跳变」中位 104px，而「玩家自身位移」
# 中位只有 46px —— 目标跳得比人走得还远 2.3 倍。
# TGT_HYST 三态：
#   "1"（默认）= 开迟滞；"0" = 一键回退改前的裸 min（tgt.pick_target(on=False)，
#   有逐帧一致性断言）；"auto" = 同场每 TGT_AB_PERIOD_S 秒自动翻转，一遍跑完两种
#   配置，把房间/人数/地图的方差直接消掉（历史日志跨场波动极大，不同会话目标跳变
#   中位在 14~167px 之间，跨场对比根本不可信）。
TGT_HYST = os.environ.get("TGT_HYST", "1")
TGT_AB_PERIOD_S = float(os.environ.get("TGT_AB_PERIOD_S", "300"))   # A/B 自动翻转周期

# —— 内存（可选增强）——
USE_MEM_TARGETS = False      # 用内存里的芝士/洞世界坐标当目标。**默认关**：实测这条折算
                             # 出的屏幕目标会跑到画面外（`tgt=(-178,521)`，也有正好等于
                             # 自己位置的 `tgt=(874,259)`），bot 照着它一路走出地图摔掉，
                             # 连带把刚标定好的锁也丢了（2026-09-19 实跑连锁三次）。它依赖
                             # 的方向前提 `_mem_dy_sign` 也不可靠——同一次标定量到
                             # `跳跃 world Δy=+0.0`。视觉本来就能检出芝士和洞，先让 bot 用
                             # 视觉真正玩起来。置 True 时才会 attach（读内存会触发杀毒误报）。
MEM_RESCAN_S = 120.0         # 全堆扫描最短间隔。实测一趟 **8.8~18s**（不是 2~3s），
                             # 6s 的间隔等于每次标定都先花十几秒扫堆，一次 4 分钟的实跑
                             # 有 90 秒（38%）耗在这里，而扫出来的玩家锚点每轮都失效
                             # （见 [[transformice-bot-state]]：内存坐标本来就不可信）。
                             # 扫描只是「白捡」，不该挡在标定前面，间隔放大到 2 分钟。
MEM_BIND_MIN_DX = 3.0        # 绑定本地玩家时，world x 至少要动这么多才算「跟着按键动」
MEM_BIND_MAX_DX = 300.0      # 上限：单次按键（~0.3s）里没人能移动这么多；超了就是绑到了垃圾结构体
MEM_PLAYER_TYPES = (1, 2, 26)  # 玩家结构体的类型 ID；绑定时再核对一次，防止扫到的地址已被回收/复用
MEM_MAX_OFFSET = 1200        # 内存目标折算到屏幕后超过这个距离就丢弃（防坐标串了瞎跑）

# —— 状态机：捡芝士(0) → 找洞(1) ——
CB_WINDOW = 15               # 背芝士滑动窗口帧数
CB_ENTER_HOLE = 8            # 窗口内 ≥N 帧有背芝士 → 切到找洞
CB_EXIT_HOLE = 2             # 窗口内 ≤N 帧有背芝士 → 切回找芝士

# —— 反掉落 ——
NO_GROUND_FALL_FRAMES = 15   # 连续 N 帧脚下无地面 → 判定掉落
FALL_GIVEUP_S = 3.0          # 判定掉落后连续这么久还没落地 → 认定定位失效，丢锁定重标定
FALL_STATIC_DY = 8           # 这么久里屏幕 y 都没往下挪这么多 → 不是掉落，是地面检测失灵

# —— 悬崖保护（别把角色带下平台摔死）——
# 2026-09-19 实测：清静房（老鼠：2）里每次锁定后约 5s，聊天栏就出现「你没有奶酪吃过了！^_^」，
# 角色死了——死掉后精灵消失、模板分数塌到 0.53、锁被判废、退避涨到 60s，一局就这么废掉。
# 同一秒的标定探针量到 `right dx=+22 dy=+16`：一边右移一边往下掉，就是照着**够不到的**
# 那个芝士（在别的平台上）一路走出平台。下面两条一起拦：优先选脚下这条平台上的目标 +
# 要去的方向就是平台边缘时不再往前走。
EDGE_GUARD_PX = 26           # 脚下平台边缘离脚底不到这么多像素时，不再往那个方向按
EDGE_DOWN_SLACK = 20         # 目标比脚底低这么多 = 本来就该往下走，放行走出边缘
SAME_PLATFORM_DY = 60        # 判定「目标和我站在同一层」的高度容差

# —— 反卡住 ——
STUCK_MOVE_TOL = 12          # 相对锚点漂移 ≥N 像素才算「真的在走」（走路 1 秒远不止 12px，动画抖动够不到）
STUCK_FRAMES = 60            # 连续静止 N 帧 → 判定卡住

# 2026-10-27：**够不着的目标要「记仇」。**
# tgt.py 里已经有「守够帧数还没靠近就换一块」，但它只在**还有别的候选**时才换
# （`alt is not None`）。实测 00:17 那一段：整屏只有 1 个候选，而且是个假目标
# （树上的绿叶当成芝士、在头顶 53px 够不着），于是机器人一路按「上」原地跳，
# [STUCK] 连报十几次、STAT 里 tgt 一直钉在 (869,396) 不动 —— 几分钟一格没挪。
# 这里在 STUCK 这一层兜底：同一个位置反复卡死就拉黑一段时间，候选表里剔掉，
# 宁可无目标去探索，也别站着跳给谁看。
TGT_BAD_CELL = 24            # 记仇按 24px 的格子归并（同一个目标抖动几像素算同一个）
TGT_BAD_HITS = 3             # 同一格子卡住这么多次 → 拉黑
TGT_BAD_FORGET_S = 20.0      # 距上次卡住超过这么久，旧账作废（地图可能换了一张）
TGT_BAD_KEEP_S = 45.0        # 拉黑时长
TGT_BAD_R = 30               # 拉黑半径（像素，候选落在这个方框里就剔掉）
STUCK_EXPLORE_JUMP = 40      # 探索（无目标）时连续静止 N 帧就跳一下
EXPLORE_TURN_S = 9.0         # 探索时顶住墙超过这么久就换方向（见探索分支那段说明）
EXPLORE_EDGE_COOL_S = 0.8    # 探索时「前面是平台边缘就掉头」两次换向的最小间隔（防窄平台上每帧抖）
# 盲锁期（还没认到自己）左右踱步/跳的节奏。**这一段不许站桩**，理由见 _anti_idle。
IDLE_WALK_S = 0.45           # 单向走多久换边（小幅度左右摆，不会一路走出平台）
IDLE_JUMP_EVERY = 1.8        # 隔多久跳一下
IDLE_JUMP_S = 0.30           # 每次跳按多久

# —— 运动（基准 600px，运行时按窗口缩放：scale = min(w,h)/600）——
SCALE_BASE = 600             # 缩放基准（窗口最小边长像素）
DX_DEAD = 25                 # 水平死区：目标水平距离小于此值不动（仅在目标与己同层时用）
DX_DEAD_CHEESE = 8           # 找芝士时的水平死区，必须比拾取半径小（见 1700 行那段说明）
ONLEVEL_DEAD_PX = 6          # 目标在**另一层**时的水平死区（要小得多，见 1690 行那段说明）
OFFLEVEL_SCAN_S = 1.5        # 目标在另一层、水平又对齐时，朝一个方向走这么多秒再换向（别站着等）
# 2026-10-26：目标在**另一层**时「算已对齐」的横带。原来的死区只有 6px，于是机器人
# 卡在目标那一列左右微调、像钟摆一样原地晃：实测 19:59 那 45 秒（tage=2732）位置在
# x=812~838 之间来回，目标 x=817 —— keys 一直有，位移却是零，**一步也没往平台边缘走**。
# 它要做的不是「对准芝士」，而是「走到平台边上掉下去」；对准到 ±45px 就够交棒给
# 下面的找边界逻辑了。
OFFLEVEL_ALIGN_BAND = 45     # 目标在另一层时，|dx| 小于这么宽就当作「成列」→ 去平台边缘
HOLE_DELIVER_PX = 34         # 芝士离开背时，人离洞口中心多近才算「送进洞了」（见 1536 行那段说明）
# 2026-10-27：**只认真地上一动不动的芝士。** 别人背上的绿芝士和地上的长得一模一样，
# 机器人追过去人家一跑就白跑（详见 cheeses = keep 后面那段说明）。
CHEESE_STILL_FRAMES = 4      # 连续这么多帧待在同一小块地方 → 认定是「地上的芝士」
CHEESE_STILL_R = 12          # 认定「同一块」的半径（按 scale 缩放；一帧走 12px 以上肯定是活的）

# 洞口同样要「静止」。实测 2026-10-27 00:2x（墓园那张图，_probe_holes.png）：
# 灰度法吐出的 5~13 个「洞」全在天上/墓碑/老鼠身上——活动期老鼠穿深褐麻袋装，
# 又暗又圆又站在地上，地面闸门拦不住；而 STAT 里 hole=5~13 会让驮着芝士的机器人
# 奔向一个墓碑。洞是**地图的一部分，一动不动**；老鼠在跑。所以按时间维过滤。
HOLE_STILL_WIN = 8           # 回看最近这么多帧
HOLE_STILL_MIN = 5           # 其中至少这么多帧在附近出现过 → 认定是洞
HOLE_STILL_R = 10            # 「同一个」的半径（按 scale 缩放）
# 2026-10-26：背芝士找洞口时的水平死区。**洞的中心天然比脚底低 20~50px**（洞是个
# 开在地表里的坑），所以「目标略低 + 水平对齐」这个组合几乎每一帧都成立，
# 而按原来的逻辑它属于「同一层、已在死区内」→ want=0 → 站着不动，直到 STUCK 来救。
# 实测 10-26 全天：背芝士且有目标的 870 次采样里，「|dx|≤30 且 15≤dy≤60」共 129 次，
# **其中 126 次原地不动（98%）**；对比 dy>60 的那档只有 1% 不动（那是另一层的扫描分支在动）。
# 这就是用户说的「在洞口徘徊」：人已经站在洞口上了，但最后那一步永远不迈。
# 收到 7px 后它会继续朝洞心走，走进洞里 cb 自然掉 0 → 判送洞。
HOLE_ENTER_DEAD_PX = 7
# 2026-10-26 二修：上面「收到 7px 就会掉进去」**实测不成立**。改成 7px 后重启测了
# 12 分钟（_bot_run_1026_2003.log）：背芝士且 |dx|<=8 的采样 26 帧，**26 帧 keys=[]、
# 送进洞 0**。人能站在洞心正上方的唯一解释是——那不是脚下这层的洞（下一层的洞，
# 或者节日地图上的假洞）。所以加一条：水平对上、洞心在脚下方 15~60px、**站了这么久
# 还没掉下去**，就改判「另一层」，交给下面「走到平台边缘」的逻辑。站住不动必错，
# 往下走至少还有机会。
HOLE_LEVEL_HOLD_S = 1.2      # 在洞心水平对齐但仍没掉下去，站这么久就改判「洞在另一层」
CB_EXIT_HOLD_S = 1.2         # 「背芝士」要持续这么久没背才算离开背（防一帧抖动就切回找芝士）
CHEESE_SELF_PAD = 6          # 判「这个芝士其实长在我自己身上」时，身体框往外放的余量。
CHEESE_SELF_R = 40           # 再兜一道：离自己中心 (px,py) 曼哈顿距离 ≤ 此值的黄块一律不算芝士。
# 两道判据都要，因为**身体框经常对不准自己**：实测框中心和 px,py 差 27~56px，有时整个框
# 锁在棕榈树/果子上（corpus/1006_175302/181340_near_target_109_+0.png），只按框裁会漏掉
# 贴在自己身上的那批。px,py 是跟得住的，所以再按「到自己的距离」兜一道。
# 2026-10-06 第二次修正 —— 前两版各对了一半，写在这儿免得再绕回去：
#   · 09-30 「按身体框剔自身」是对的，症状就是「bot 站在自己脑袋上捡芝士、cb 永远 0」；
#   · 10-06 把它降到 6 又**无条件豁免当前锁**是错的：豁免把自身假芝士原样放了回来。
#     证据 corpus/1006_175302/180309_near_target_062_+0.png：detect_cheeses 五个候选里
#     最近的是 (619,336) d=17，正落在身体框内（老鼠自己那身土黄）；而**真芝士在
#     (333,338)、d=301，一直好好在候选表里**，被「最近优先」每帧跳过。跑完 1h24m
#     仍是 0 次送洞、距离地板纹丝不动卡在 17px（=自身那块黄到自己的距离）。
#   · 正确口径 = 剔自身 + **只豁免「从远处锁上的」那个目标**，见下面 `_far_lock`。
# 半径值不敏感：语料 1631 帧，34/38/42/46 四个取值过滤后「无候选」帧数都是 29（1.8%）。
DY_JUMP = 30                 # 垂直跳跃死区：目标高于此值才跳
FAR_GAP_DX = 100             # 目标水平距离超过此值视为「远处」，用于跨缝隙判断
PLATFORM_MIN_WIDTH = 150     # 脚下平台宽度小于此值且目标远 → 起跳跨缝隙

_kb = KBC()
# 前台门闸：只往**当前前台**就是游戏窗口时才发键。pynput 的按键进的是前台窗口，
# 2026-10-18 用户实测：机器人还在跑的时候把焦点点到 PowerShell，方向键就打进了终端
# （翻出以前的命令/移动光标）；重登输密码时也是同一个原因——机器人把键盘抢走了，
# 密码根本没输进去。**这里不抢焦点**：用户在别的窗口干活时，机器人安静暂停，别去抢回来。
_gate_hwnd = None          # Cap 选到/重绑窗口后同步过来；None = 不校验
_gate_skipped = 0          # 因焦点不在游戏上而跳过的按键次数（日志限流用）
_manual_until = 0
_last_bot_key = 0
# 机器人**自己**注入输入的墙钟时刻。用来把「我在按键」和「用户在打字」分开：
# GetLastInputInfo 把两者算成同一种「输入」，于是机器人一按方向键就刷新了空闲计时，
# 自己的 90 秒让位闸门永远打不开（2026-10-26 22:1x 实测：机器人一直 NO-LOCK，
# 日志里每一轮都是「你在用电脑 → 这轮不抢前台」，可电脑前根本没人）。
_last_bot_input = 0.0
_quit = False
_held = set()          # 监听线程自己维护的「当前按下的方向键」，用来区分新按下和系统自动重复
_pending_bot = {}      # key -> 注入时刻：bot 注入的键登记在这，监听线程回调时认领，不当人工输入
_BOT_KEY_TTL = 5.0     # 登记最多留这么久等认领，超时说明这次注入的回调根本没来


def _on_press(key):
    global _manual_until, _quit
    if key in (Key.esc, Key.f1):
        _quit = True
        return
    if key in (Key.left, Key.right, Key.up, Key.down):
        t_inj = _pending_bot.pop(key, None)
        if t_inj is not None and time.time() - t_inj < _BOT_KEY_TTL:
            # 这是 bot 自己注入的键。**不能靠「距上次注入的时间差」判断**：标定期间主线程
            # 在跑 cv2/内存扫描，钩子回调会排队几十~几百 ms 才回来，gap 于是变成 0.4s 以上，
            # 被当成人工接管——实测每轮标定的几个探针按键全部触发，一回合白扔 45s。
            # 注入时登记、回调时认领，回调迟到多久都能对上。
            _held.add(key)
            return
        repeat = key in _held
        _held.add(key)
        gap = time.time() - _last_bot_key
        if gap > MANUAL_KEY_GAP:
            # Windows 长按会自动重复发 keydown，pynput 照样回调 on_press，也不能算人工。
            if repeat:
                print(f"[KEY] 忽略自动重复 {key}（距上次 bot 按键 {gap:.2f}s）——不是人工接管")
                return
            _manual_until = time.time() + MANUAL_TIMEOUT
            print(f"[MANUAL] 判定人工接管：{key}（新按下），距上次 bot 按键 {gap:.2f}s")


def _on_release(key):
    _held.discard(key)


_listener = Listener(on_press=_on_press, on_release=_on_release)
_listener.daemon = True
_listener.start()

# ====== Keyboard Helpers ======
VK_MAP = {Key.left: "left", Key.right: "right", Key.up: "up"}


def _key_down(d):
    global _gate_skipped
    # 焦点不在游戏窗口上就**一个键都不发**（见 _gate_hwnd 上面那段）。
    # 只挡「按下」，不挡松开/release_all —— 否则真按着的键会卡在按下状态。
    if _gate_hwnd:
        try:
            fg = win32gui.GetForegroundWindow()
        except Exception:
            fg = None
        if fg != _gate_hwnd:
            _gate_skipped += 1
            if _gate_skipped == 1 or _gate_skipped % 300 == 0:
                print(f"[Bot] 焦点不在游戏窗口上 → 暂停发键（累计 {_gate_skipped} 次）；"
                      f"把游戏窗口点回前台就自动继续")
            return False
    if isinstance(d, str):
        d = {"left": Key.left, "right": Key.right, "up": Key.up}[d]
    _pending_bot[d] = time.time()
    globals()["_last_bot_input"] = time.time()
    _kb.press(d)
    return True


def _key_up(d):
    global _last_bot_input
    if isinstance(d, str):
        d = {"left": Key.left, "right": Key.right, "up": Key.up}[d]
    _last_bot_input = time.time()
    _kb.release(d)


def _key_tap(d, ms=TAP_MS):
    _key_down(d)
    time.sleep(ms / 1000)
    _key_up(d)


# ====== Screen Capture ======
class Cap:
    def __init__(self):
        self.hwnd = None
        self.cam = None
        self.mon_w = 0
        self.mon_h = 0
        self._last = None
        self._why = ""

    def _pick_window(self, verbose=False):
        """枚举标题为 Transformice 的可见窗口，优先 C 盘进程；返回 hwnd，找不到返回 None。"""
        windows = []

        def cb(h, _):
            if win32gui.IsWindowVisible(h) and win32gui.GetWindowText(h) == "Transformice":
                rect = win32gui.GetWindowRect(h)
                _, pid = win32process.GetWindowThreadProcessId(h)
                windows.append((rect, pid, h))

        win32gui.EnumWindows(cb, None)

        if not windows:
            return None

        if verbose:
            for rect, pid, h in windows:
                try:
                    exe = psutil.Process(pid).exe().lower()
                except Exception:
                    exe = "?"
                print(f"[Bot] Found window pid={pid} exe={exe}")

        # Prefer C-drive process
        for rect, pid, h in windows:
            try:
                exe = psutil.Process(pid).exe().lower()
                if exe.startswith("c:\\"):
                    if verbose:
                        print(f"[Bot] Selected C-drive game (pid={pid})")
                    return h
            except Exception:
                pass

        if verbose:
            print(f"[Bot] Fallback to first window (pid={windows[0][1]})")
        return windows[0][2]

    def rebind(self):
        """游戏重启（或窗口句柄失效）后重新找一个窗口。找到返回 True。

        句柄失效时 GetWindowRect 会抛 1400，快照一直失败；这里重扫一遍窗口就能接上。"""
        h = self._pick_window()
        if h is None:
            return False
        global _gate_hwnd
        self.hwnd = h
        _gate_hwnd = h
        self._last = None
        self._why = ""
        return True

    def start(self):
        global _gate_hwnd
        self.hwnd = self._pick_window(verbose=True)
        _gate_hwnd = self.hwnd
        if self.hwnd is None:
            return False
        try:
            import dxcam
            self.cam = dxcam.create(output_idx=0, output_color="BGR")
            self.mon_w, self.mon_h = self.cam.width, self.cam.height
        except Exception as e:
            print(f"[Bot] dxcam init failed: {e}")
            return False
        return True

    # 抢前台/置顶这类 USER32 调用是**同步**的：内核把消息投给目标窗口的线程并等它返回。
    # 目标那一刻不处理消息（换地图、加载中、或者它自己已经卡死）就会一直等下去。
    # 2026-10-10 实测就是这么冻住的：机器人主线程卡死在 BringWindowToTop 里出不来，
    # 紧接着两个游戏客户端也一起僵死（Windows 事件日志里那串 AppHang 是同一类事）。
    # 三道保险，顺序不能换：
    #   ① 动手之前先问窗口「你还回消息吗」（IsHungAppWindow + SendMessageTimeout(WM_NULL)）：
    #      不回就**一个前台动作都别做** —— 对僵死窗口提层级/抢前台必卡，而且卡的是内核级等待；
    #   ② 整段抢前台动作扔进子线程，主线程只等 _FOCUS_TIMEOUT_S，卡住最多损失一个线程；
    #   ③ 卡过一次就把这窗口拉黑 _FOCUS_BAN_S，别反复去撞。
    # ⚠ 别把 AttachThreadInput 那招删了：实测游戏不在前台时**只有它能抢到**（见 focus 文档字符串），
    #   删掉的表现是「永远抢不到前台 → 永远 NO-LOCK」。它确实是把两个进程的输入队列接在一起的
    #   死锁配方，所以只靠 ①②③ 兜住风险（先确认对面活着 + 超时 + 拉黑）。
    # 代价：抢不到前台 = 按键进不去游戏 = 标定不过（NO-LOCK 空转），但**不会把游戏搞死**。
    _FOCUS_TIMEOUT_S = 5.0
    _FOCUS_BAN_S = 120.0
    _focus_ban_until = 0.0
    _focus_ban_logged = False

    def _window_alive(self, h, ms=400):
        """窗口还在处理消息吗？僵死的窗口一个前台动作都别做（见上 ①）。"""
        try:
            u = ctypes.windll.user32
            if u.IsHungAppWindow(h):
                return False
            SMTO_ABORTIFHUNG = 0x0002
            r = ctypes.c_size_t(0)          # PDWORD_PTR：x64 下是 8 字节，别用 c_ulong（会写出界）
            rc = u.SendMessageTimeoutW(h, 0, 0, 0, SMTO_ABORTIFHUNG, int(ms), ctypes.byref(r))
            return bool(rc)
        except Exception:
            return True                     # 查不了就当活着（保持原来的行为，别把能用的路堵死）

    def _raise_win(self, h):
        """原始的「置顶 + 抢前台」那一套（在子线程里跑，见上面三道保险）。"""
        try:
            win32gui.ShowWindow(h, 9)      # SW_RESTORE
            win32gui.BringWindowToTop(h)
        except Exception:
            pass
        u = ctypes.windll.user32
        u.SetForegroundWindow(h)
        if win32gui.GetForegroundWindow() != h:
            SWP_NOSIZE, SWP_NOMOVE = 0x0001, 0x0002
            u.SetWindowPos(h, -1, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE)  # TOPMOST
            u.SetWindowPos(h, -2, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE)  # NOTOPMOST
            u.SetForegroundWindow(h)
        if win32gui.GetForegroundWindow() != h:
            # 前台锁：调用方自己不是前台进程时 SetForegroundWindow 会被系统直接忽略。
            # 把当前前台窗口的线程接到本线程上，两者成了「同一个输入队列」，这时再抢就会被接受
            # （实测：游戏不在前台时前两招都没抢到，只有这招成）。
            # GetCurrentThreadId 在 kernel32，不在 user32。finally 一定要解绑。
            fg = win32gui.GetForegroundWindow()
            if fg:
                ft = u.GetWindowThreadProcessId(fg, None)
                ct = ctypes.windll.kernel32.GetCurrentThreadId()
                u.AttachThreadInput(ct, ft, True)
                try:
                    u.SetForegroundWindow(h)
                    u.BringWindowToTop(h)
                finally:
                    u.AttachThreadInput(ct, ft, False)
        return win32gui.GetForegroundWindow() == h

    # 2026-10-22：人在用电脑（打字/点鼠标）时**不抢前台**。以前机器人一标定就把
    # 游戏窗口顶到最前，用户正在输入的中文当场被顶掉、输入法连上屏的机会都没有。
    # 抢前台只在「人已经离开键盘 90 秒」时做：这个阈值要足够长，用户看完一段回复
    # 再接着打字也不会被抢；只要还在敲字/滚鼠标，就一直让位。
    _USER_IDLE_HOLD_S = 90.0
    _focus_user_hold_logged = False

    def _user_active(self, hold=None):
        """系统最近有没有**真实输入**（键、鼠都算）。

        用 GetLastInputInfo 一次拿全系统最近一次输入时间。

        ⚠ 2026-10-26 22:1x 修：这个 API **分不清是谁输入的**，机器人自己注入的方向键
        也算一次「输入」。于是「空转 → 标定 → 按探针键 → 空闲计时被自己刷掉 →
        下一轮又判『你在用电脑』」形成死循环：一晚上每一轮都是「这轮不抢前台」，
        永远回不到游戏里（日志 _bot_run_1026_2214 就是整段这样空转到天亮级别）。
        现在拿 `_last_bot_input` 对一下：最近这一次输入如果**就是我自己 0.25s 内注入的**，
        那它不可能是用户在打字，直接判「用户不在」。
        """
        hold = self._USER_IDLE_HOLD_S if hold is None else hold
        try:
            class _LII(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
            li = _LII()
            li.cbSize = ctypes.sizeof(li)
            if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(li)):
                return False
            now = ctypes.windll.kernel32.GetTickCount()
            idle = ((now - li.dwTime) & 0xFFFFFFFF) / 1000.0
            last_in_wall = time.time() - idle
            if last_in_wall - _last_bot_input <= 0.25:
                return False            # 这条「输入」是机器人自己刚发的
            return idle < hold
        except Exception:
            return False

    def focus(self):
        """把游戏窗口切到前台：pynput 的按键只会进当前前台窗口，焦点不在游戏上
        时标定一定测到 dx=0（而且方向键会被打进终端）。

        ⚠ 抢之前先看人在不在用电脑（见 _USER_IDLE_HOLD_S）：在打字就原样让位，
        返回 False，调用方放弃本次标定——宁可机器人空转一轮，也别把用户的中文顶掉。

        注意**不要**用「空按一下 ALT」那招绕过 SetForegroundWindow 的限制：单独按
        ALT 会激活窗口的系统菜单（最小化/最大化/关闭），菜单一开，方向键全被它吃掉，
        游戏根本收不到键——实测就是这么来的标定必然失败。改用临时置顶抢前台。

        整段抢前台动作带超时 + 先查窗口活没活（见上面三道保险）：抢不到就返回 False，
        由调用方决定放弃本次标定，绝不让主线程停在这里。"""
        import threading
        try:
            already_fg = win32gui.GetForegroundWindow() == self.hwnd
        except Exception:
            already_fg = False
        if not already_fg and self._user_active():
            if not self._focus_user_hold_logged:
                self._focus_user_hold_logged = True
                print(f"[Bot] 你在用电脑（{self._USER_IDLE_HOLD_S:.0f}s 内有键鼠输入）→ "
                      f"这轮不抢游戏前台，机器人先等着；你停手 {self._USER_IDLE_HOLD_S:.0f}s 后自动接管")
            return False
        if time.time() < self._focus_ban_until:
            return False
        if not self.hwnd or not win32gui.IsWindow(self.hwnd):
            if not self.rebind():
                return False
        if not self._window_alive(self.hwnd):
            self._focus_ban_until = time.time() + self._FOCUS_BAN_S
            if not self._focus_ban_logged:
                self._focus_ban_logged = True
                print(f"[Bot] ⚠ 游戏窗口不响应窗口消息（IsHungAppWindow/WM_NULL 超时）——"
                      f"客户端多半已经僵死。不再对它做任何抢前台动作（对僵死窗口提层级必卡），"
                      f"{self._FOCUS_BAN_S:.0f}s 内不再试；这期间只能空转（NO-LOCK），"
                      f"要把客户端重启（卡死守护 _wedge_guard.py 会自动干这事）。")
            return False
        box = {}

        def _job():
            try:
                box["ok"] = self._raise_win(self.hwnd)
            except Exception as e:           # 句柄失效等：当作抢失败
                box["err"] = e

        th = threading.Thread(target=_job, name="focus-raise", daemon=True)
        th.start()
        th.join(self._FOCUS_TIMEOUT_S)
        if th.is_alive():
            # 卡在 USER32 里了 —— 窗口在抢的过程中僵死了。拉黑，别再制造第二个卡住的线程。
            self._focus_ban_until = time.time() + self._FOCUS_BAN_S
            if not self._focus_ban_logged:
                self._focus_ban_logged = True
                print(f"[Bot] ⚠ 抢前台卡住（{self._FOCUS_TIMEOUT_S:.0f}s 未返回）——"
                      f"游戏窗口在这中间停止处理窗口消息了，通常=客户端卡死。"
                      f"放弃抢前台，{self._FOCUS_BAN_S:.0f}s 内不再试；"
                      f"这期间机器人只能空转（NO-LOCK），需要把客户端重启。")
            return False
        if box.get("err"):
            print(f"[Bot] focus 失败: {box['err']}")
            return False
        ok = bool(box.get("ok"))
        time.sleep(0.2)
        ok = ok and win32gui.GetForegroundWindow() == self.hwnd
        if ok:
            self._focus_ban_logged = False
            self._focus_user_hold_logged = False
            self.click_canvas()
        return ok

    def click_canvas(self):
        """在游玩区点一下，把**键盘焦点**交给 Flash 画布。

        只 SetForegroundWindow 是不够的：窗口明明在前台（GetForegroundWindow 就是
        它），Flash 画布没拿到焦点时方向键照样不生效——实测按住 → 1.5 秒，416 个
        可信块里 415 个纹丝不动、内存里也没一个结构体坐标变化；点一下画布再按，
        立刻有几十块跟着动。点 35% 高度、水平居中，避开顶栏和底部的背包/聊天面板。
        游戏里左键点空地没有副作用（不像底部物品栏）。"""
        try:
            x1, y1, x2, y2 = win32gui.GetWindowRect(self.hwnd)
        except Exception:
            return
        if x2 - x1 < 50 or y2 - y1 < 50 or x1 <= -30000:
            return
        mx, my = (x1 + x2) // 2, y1 + int((y2 - y1) * 0.35)
        global _last_bot_input
        try:
            from pynput.mouse import Button, Controller as MC
            MC().position = (mx, my)
            time.sleep(0.05)
            MC().click(Button.left)
            _last_bot_input = time.time()
            time.sleep(0.05)
        except Exception:
            pass

    def snap(self):
        try:
            x1, y1, x2, y2 = win32gui.GetWindowRect(self.hwnd)
            # 最小化时 GetWindowRect 返回 -32000 附近的哨兵值
            if x1 <= -30000 or y1 <= -30000:
                self._why = "游戏窗口被最小化"
                return None
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.mon_w, x2), min(self.mon_h, y2)
            if x2 - x1 < 50 or y2 - y1 < 50:
                self._why = f"游戏窗口太小 ({x2 - x1}x{y2 - y1})"
                return None
            frame = self.cam.grab(region=(x1, y1, x2, y2))
            if frame is None:
                # 画面没变化时 dxcam 无新帧，沿用上一帧
                return self._last
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._last = (frame, gray)
            self._why = ""
            return self._last
        except Exception as e:
            self._why = f"截图异常 {type(e).__name__}: {e}"
            return None

    def why(self):
        """最近一次 snap() 失败的原因，供主循环打印一次（否则失败时静默空转）。"""
        return getattr(self, "_why", "") or "未知"


# ====== Main Bot ======
class Bot:
    def __init__(self):
        self.cap = Cap()
        self.vision = Vision()
        self.fn = 0
        self._no_player_frames = 0
        self.playable = None          # None=未知, True=能操作, False=按键无响应
        # 盲锁期的「别站桩」状态（见 _anti_idle）
        self._idle_dir = 1
        self._idle_turn_next = 0.0
        self._idle_jump_next = 0.0
        self._idle_jump_until = 0.0
        # 盲锁期是不是已经接管了方向键。**不能每帧 release_all() 再按下**，否则 60fps
        # 下每秒发 60 次抬起/按下，角色会在「动一下停一下」之间抖，等于还是没在好好走。
        self._anti_idle_on = False
        # 深度学习自定位（见 _dl_predict）。None=还没加载，False=不可用
        self._dl = None
        self._dl_tried = False
        self._next_dl_try = 0.0
        self._lock_src = None         # 这把锁是「模板标定 / 名字重挂 / 深度学习」哪条路来的
        self._next_calib = 0.0
        self._next_label_try = 0.0    # 下次试「名字标签重挂」的时刻（纯视觉，不按键）
        self._tpl_reload_next = 0.0   # 下次「从磁盘重读名字字形」的时刻（模板自愈用）
        self._had_lock = False        # 上一帧有没有锁定（用来在「刚丢锁」那一刻记账退避）
        self._lock_since = 0.0        # 本次锁定是什么时候拿到的
        self._drop_streak = 0         # 连续几次锁定没撑到 LOCK_MIN_USEFUL_S
        self.state = 0  # 0 = seek cheese, 1 = seek hole
        self._keys = set()
        self._dbg = "-"   # 供 [STAT] 周期打印的一行状态摘要
        self._cap_dead = False   # 截图是否处于失败状态（只打印一次原因）
        self._cap_fail = 0       # 连续截图失败次数，用来决定多久重扫一次窗口
        self._last_dump = 0.0    # 上次存 _dbg_frame.png 的时间

        # Anti-stuck state
        self.stuck_frames = 0
        self.stuck_x = 0
        self.stuck_y = 0
        self._stuck_count = 0
        self._stuck_streak = 0     # 连续几次脱困都没挪窝（一动就清零）；用来触发锁定复核
        self._stuck_tgt = {}       # {(格子x,格子y): (次数, 上次时刻)} —— 见 TGT_BAD_HITS
        self._bad_tgts = []        # [(x, y, 到期时刻)] 这段时间内不再追的目标
        self._scan_dir = 1         # 目标在另一层时的「沿平台找路」扫描方向（+1 右 / -1 左）
        self._scan_since = 0.0     # 上次换扫描方向的时刻（见 OFFLEVEL_SCAN_S）
        self._anchor_gray = None   # 卡住判定的锚点帧；None = 下一帧重新锚定
        self._bg_shift = 0         # 本帧背景相对锚点帧滚了多少（相机是否跟随的证据）
        self._bg_hits = 0          # 背景位移**连续**成立了多少帧（见 anti-stuck 里的说明）
        self._explore_dir = 1      # 没有目标时的探索方向（+1 右 / -1 左），撞住了才换边
        self._explore_t = 0.0      # 上次换探索方向的时刻（防「一辈子朝右边」）
        self._bg_n = 0             # 上面这个估计采到几块可信样本（太少就当「未知」）

        # Anti-fall state
        self.no_ground_frames = 0
        self._fall_since = 0.0     # 本段「判定掉落」从什么时候开始（0 = 没在掉落）
        self._fall_py0 = 0         # 开始判掉落那一刻的屏幕 y（判「真掉还是检测失灵」用）

        # 帧环形缓冲（离线语料，见 _ring.py）。默认关，DBG_RING=1 才落盘。
        self._ring = Ring()
        self._cb_prev = 0        # 上一帧的 cb_count，用来抓 cb 的上升/下降沿
        self._state_prev = 0     # 上一帧的 state，用来抓状态机跳变
        self._near_prev = False  # 上一帧是否已经贴到目标跟前（抓「贴上去」的边沿）
        self._bg_sample_t = 0.0  # 上次落背景样本的时间
        self._ring_note = "-"    # 本帧的备注（事件触发时写进 csv）

        # 目标迟滞的跨帧状态（改动一，逻辑在 tgt.py，这里只持有）
        self._tgt_state = tgt_mod.new_state()
        # 「当前这个目标曾经在自身半径之外被我看见过」的锁存位（2026-10-06）。
        # 不能在过滤那一步直接用 tgt.py 的 start_d：TGT_HYST=0（裸 min 兜底模式）
        # 下 tgt.py 每帧都重锁一次，start_d 等于**当前**距离²，机器人一走近就归零，
        # 豁免会中途失效、目标被删、机器人掉头 —— 自己跟自己振荡。
        # 锁存位只在「换了一块」时才清，两种模式下都对。
        self._tgt_far = False
        self._tgt_far_xy = None
        # SAME_PLATFORM_DY 在两边各有一份（这里 146 行、tgt.DEFAULTS 里也有一份，
        # 那边给 _same_platform 用）。今天都是 60，但放着不管迟早会改一边忘一边，
        # 那就成了「同平台过滤按哪个值走」的静默分歧 —— 在这里显式喂进去，单一真值。
        self._tgt_C = dict(tgt_mod.DEFAULTS)
        self._tgt_C["SAME_PLATFORM_DY"] = SAME_PLATFORM_DY
        self._tgt_ev = 0            # 本场目标事件数（换锁/放弃/清空）
        self._tgt_switches = 0      # 其中「抢锁换目标」的次数
        self._tgt_on = self._tgt_hyst_on()
        self._tgt_on_prev = self._tgt_on

        # Cheese-on-back sliding window (last 15 frames, True=cheese detected)
        self._cb_history = []
        self._cb_low_since = 0.0     # 「背芝士」塌下去是从什么时候开始的（见 CB_EXIT_HOLD_S）
        # 「洞在另一层」的判定：0=没在计时；否则=开始水平对齐站住的时刻。
        # 一旦判成另一层就**锁死**（不清零），直到换了目标或状态机回 0 —— 否则一离开
        # 洞心 dx 就变大、计时被清零，机器人又走回来，变成在洞和平台边之间来回摆。
        self._hole_lvl_since = 0.0
        self._hole_lvl_tgt = None    # 当时锁的那个洞口坐标，换了目标就作废
        # 送进洞 / 掉芝士 的计数（见 GOT_HOLE -> CHEESE 那个分支）
        self._delivered = 0
        self._dropped = 0
        self._selfdrop = 0    # 被剔掉的「长在自己身上」的假芝士个数（见 detect_cheeses 之后那段）
        self._cheese_hist = []   # 芝士候选的连续帧历史 [(x, y, 连续命中帧数, 原始元组)]
        self._cheese_still_n = 0  # 上一帧「判定为地上不动芝士」的个数（进 STAT）
        self._hole_hist = []     # 最近若干帧的洞候选，用来判「是不是一直待在那儿」
        self._hole_still_n = 0   # 上一帧过了静止闸的洞个数

        # Memory reader (optional)
        self.mem = Mem()
        self.use_mem = False
        self._mem_player_wx = 0
        self._mem_player_wy = 0
        # 地址缓存：全堆扫一趟要 2~3s，芝士/洞各 1.1s，绝不能每帧扫。改成一趟拿
        # 地址列表，之后每帧只重读这些地址（几十个地址 <1ms）。
        self._mem_players = []      # 可能是玩家的结构体地址（type 1/2/26）
        self._mem_cheese = []       # 芝士对象地址
        self._mem_hole = []         # 洞口对象地址
        self._mem_scan_t = 0.0      # 上次全扫的时间
        self._mem_player_addr = None   # 绑定好的本地玩家结构体地址
        self._mem_key_sign = 1      # 按 right 时 world x 的变化方向（+1/-1，实测才定）
        self._mem_dy_sign = None    # 跳跃时 world y 的方向：None=未标定

    def _guard(self, fn, *a):
        global _last_bot_key
        _last_bot_key = time.time()
        return fn(*a)

    def _kdown(self, k):
        self._guard(_key_down, k)

    def _kup(self, k):
        self._guard(_key_up, k)

    def _ktap(self, k, ms=TAP_MS):
        self._guard(_key_tap, k, ms)

    def _key(self, k, down):
        self._guard(self._key2, k, down)

    def _key2(self, k, down):
        if down and k not in self._keys:
            self._keys.add(k)
            _key_down(k)
        elif not down and k in self._keys:
            self._keys.discard(k)
            _key_up(k)

    def release_all(self):
        self._guard(self._release_all_impl)

    def _release_all_impl(self):
        for k in list(self._keys):
            self._key(k, False)

    def _probe_move(self, key, hold_s=IDENTITY_HOLD, search=PROBE_SEARCH):
        """按一下方向键，返回 (按下前 snap, 松开后 snap, 位移列表)；snap 为 (bgr, gray)。"""
        r0 = self.cap.snap()
        if r0 is None:
            return None, None, None
        self._kdown(key)
        t = time.time()
        while time.time() - t < hold_s:
            self._guard(lambda: None)      # 刷新 _last_bot_key，避免被判成人工接管
            time.sleep(0.02)
        self._kup(key)
        time.sleep(0.08)                   # 等画面稳定
        r1 = self.cap.snap()
        # dxcam 在桌面没有新帧时返回 None，snap() 只好沿用上一帧。如果沿用的正好是
        # r0，后面就成了拿一帧跟自己比——必然全是 dx=0、score=1.000，标定白白失败
        # 一次（日志里「dx 分布 top: [(0, 746)]」+「score=1.000」就是这个）。等到
        # 真的换了帧再走；一直等不到就是画面真没动，交给上层判「不可操作」。
        for _ in range(15):
            if r1 is not None and r1 is not r0:
                break
            time.sleep(0.02)
            r1 = self.cap.snap()
        if r1 is None or r1 is r0:
            return r0, None, None
        return r0, r1, self.vision.match_translation(r0[1], r1[1], search=search)

    @staticmethod
    def _frozen(a, b, tol=2):
        """按键前后两帧是不是一模一样（灰度图最大差 <= tol）。

        真的在跑的游戏永远有东西在动（倒计时、云、水面、别的老鼠），所以「整屏零
        变化」是个很干净的「不在游戏里」判据。b 为 None 也算——探针等了 0.3s 都
        没等来新帧，说明桌面根本没刷新。

        **拿不到帧、或两帧尺寸不同，一律返回 True**：absdiff 尺寸不符会抛 cv2.error
        把主循环打死（换地图时窗口改尺寸，实测崩过一次），而且这两种情况本来也判不了。"""
        if a is None or b is None or a[1] is None or b[1] is None:
            return True
        if a[1].shape != b[1].shape:
            return True
        return int(cv2.absdiff(a[1], b[1]).max()) <= tol

    @staticmethod
    def _clusters(moved):
        """把位移块按空间邻近聚簇，按「分数和」从大到小返回。

        同一只老鼠的「名字标签 + 身体」会连成一片一起动（网格步长 18px、块 48px，
        所以相邻块中心只差 18px，容差给到 45px）；同时要求位移量接近，免得把
        紧挨着的移动平台/别的玩家并进同一簇。
        """
        n = len(moved)
        used = [False] * n
        groups = []
        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            stack, grp = [i], []
            while stack:
                k = stack.pop()
                grp.append(moved[k])
                for j in range(n):
                    if used[j]:
                        continue
                    if (abs(moved[k][0] - moved[j][0]) <= 45
                            and abs(moved[k][1] - moved[j][1]) <= 45
                            and abs(moved[k][2] - moved[j][2]) <= 8):
                        used[j] = True
                        stack.append(j)
            groups.append(grp)
        groups.sort(key=lambda g: -sum(p[4] for p in g))
        return groups

    @staticmethod
    def _world_dx(matches, min_score=IDENTITY_MIN_SCORE):
        """估计「背景整体」的水平位移：短图相机静止 → 0；长图相机跟随 → 按 → 时
        世界整片左移（实测约 -26px）。取所有可信匹配块 dx 的中位数——背景块数量
        占绝对多数，玩家/别的老鼠只是少数派，带不偏中位数。中位数绝对值 < 8 就
        当作静止（位移量化/相机抖动的余量）。"""
        ds = sorted(p[2] for p in (matches or []) if p[4] >= min_score)
        if not ds:
            return 0
        m = ds[len(ds) // 2]
        return 0 if abs(m) < 8 else m

    def _track_bg_shift(self, gray):
        """本帧背景相对锚点帧滚了多少像素，存进 self._bg_shift / self._bg_n。

        为什么要拿锚点帧直接比、而不是逐帧累加：逐帧位移在 70fps 下只有 1~2px，
        取整后噪声和真值同量级，累加几十帧会系统性偏小。直接和锚点帧比就精确得多
        （锚点每移动 12px 就重设一次，所以相对位移始终小，search=30 够用）。

        **只在「屏幕坐标判据不成立」时才调用**（见主循环 anti-stuck 那段的注释）：
        world_shift 是全帧相位相关，实测中位 21.3ms，是锁定路径上最大的一笔没被
        [PERF] 计时的开销。角色在屏幕上正常移动时这条证据改变不了结论，白付这 21ms。
        """
        if self._anchor_gray is None or self._anchor_gray.shape != gray.shape:
            self._anchor_gray = gray.copy()
            self._bg_shift, self._bg_n = 0, 0
            return
        dx, dy, n = self.vision.world_shift(self._anchor_gray, gray)
        self._bg_shift, self._bg_n = max(abs(dx), abs(dy)), n

    # ====== 内存增强（可选；attach 失败时下面全是空操作）======
    def _mem_rescan(self, force=False):
        """重扫堆，刷新「像玩家 / 芝士 / 洞口」的地址列表。

        全堆扫一趟实测 8.8~18 秒（早期注释写的 2~3s 是错的），所以按 MEM_RESCAN_S 限流。
        换地图或进出洞后旧地址会失效
        （对象被回收重建），等下一轮重扫即可；需要立刻可用时传 force。
        """
        if not self.use_mem:
            return
        if not force and time.time() - self._mem_scan_t < MEM_RESCAN_S:
            return
        t = time.time()
        self._mem_players = self.mem.scan_player_addrs()
        # 换地图/进洞后对象会被重建，之前绑定的地址可能已经失效。它不在新一批候选里
        # 就丢掉绑定，退回视觉（下次标定会重新 bind），别拿失效地址硬算坐标。
        if self._mem_player_addr and self._mem_player_addr not in self._mem_players:
            print(f"[MEM] 玩家结构体 {hex(self._mem_player_addr)} 已失效，解除绑定")
            self._mem_player_addr = None
        objs = self.mem.scan_object_addrs([30, 3, 94, 68, 7])
        # 芝士优先 type 30、洞口优先 94，取不到退到备选 ID：类型 ID 会随版本/地图变，
        # 而旧那条「按 ID 挨个试到有结果」的探测没法用在一次合并扫描里。
        self._mem_cheese = objs.get(30) or objs.get(3) or []
        self._mem_hole = objs.get(94) or objs.get(68) or objs.get(7) or []
        self._mem_scan_t = time.time()
        print(f"[MEM] 扫描完成：玩家候选 {len(self._mem_players)}、芝士 {len(self._mem_cheese)}、"
              f"洞 {len(self._mem_hole)}（{self._mem_scan_t - t:.1f}s）")

    def _mem_positions(self, addrs):
        """重读一批已知地址，返回 [(x, y), ...]；读不到/越界的自动丢弃。"""
        return [(x, y) for (x, y, _t) in self.mem.read_many(addrs).values()]

    def _bind_mem_player(self, before, after):
        """用「按键前后谁动了」绑定本地玩家结构体，并定出 world x 的方向。

        为什么必须这么做：locate_player() 取的是候选列表的第一个，而实测那个结构体
        坐标永远不动（按 right/left/up，world Δ 全是 +0.00）——拿它当锚点，换算出来
        的目标位置全是错的。这里改成让数据说话：按 right 时 world x 必须有变化。

        返回 True 表示绑定成功；同时把「按 right 时 world x 变大还是变小」记进
        _mem_key_sign（实测不一定是 +1，不能假定）。
        """
        if not before or not after:
            # 静默失败过一次，日志里只剩「没有绑定成功」这一条线索，白查半天。
            print(f"[MEM] 绑定失败：结构体读数拿不到（按键前 {len(before)} 个、"
                  f"按键后 {len(after)} 个）")
            return False
        cands = []
        best = 0.0
        for a, (x1, y1, t1) in after.items():
            if a not in before:
                continue
            if t1 not in MEM_PLAYER_TYPES:
                # 实测绑到过 type=10240 的地址：扫描时它还是玩家结构体，按键前后这段时间
                # 里被回收/复用成了别的东西（x 恰好也在动）。类型对不上的直接排除。
                continue
            dx = x1 - before[a][0]
            if abs(dx) > MEM_BIND_MAX_DX:
                # 实测绑到过 Δx=+2280 的垃圾结构体：那种「x」只是恰好被刷新的任意内存，
                # 拿它当玩家锚点会把目标算到地图外。单次按键内老鼠不可能移动这么多。
                continue
            best = max(best, abs(dx))
            if abs(dx) >= MEM_BIND_MIN_DX:
                cands.append((abs(dx), a, dx, t1))
        if not cands:
            print(f"[MEM] 绑定失败：{len(after)} 个结构体没有一个跟着按键动"
                  f"（最大 Δx={best:.2f}，需在 {MEM_BIND_MIN_DX}~{MEM_BIND_MAX_DX} 之间）"
                  f"——角色没动，大概率不在可玩状态")
            return False
        cands.sort(reverse=True)
        _, addr, dx, tid = cands[0]
        self._mem_player_addr = addr
        self._mem_key_sign = 1 if dx > 0 else -1
        print(f"[MEM] 绑定玩家结构体 @ {hex(addr)} type={tid}（按 right 时 Δx={dx:+.1f}，"
              f"故 world x 按 right {'变大' if dx > 0 else '变小'}）")
        return True

    def _calib_mem_sign(self):
        """跳一下，实测 world y 的正方向。

        文档里没有约定，只能试：Transformice 的 world y 可能「向上为正」，也可能像
        屏幕那样「向下为正」。记进 _mem_dy_sign 供目标换算用（None = 没标定过）。
        """
        if not self.use_mem or not self._mem_player_addr:
            return
        b = self.mem.read_many([self._mem_player_addr]).get(self._mem_player_addr)
        self._ktap("up", JUMP_TAP_MS)
        time.sleep(0.12)
        a = self.mem.read_many([self._mem_player_addr]).get(self._mem_player_addr)
        if not b or not a:
            return
        dy = a[1] - b[1]
        # 跳跃在屏幕上是「向上」，所以换算系数要让它得到负的 screen dy
        self._mem_dy_sign = 1 if dy < 0 else -1
        print(f"[MEM] 跳跃 world Δy={dy:+.1f} 对应 world y "
              f"{'向下为正（屏幕式）' if self._mem_dy_sign > 0 else '向上为正'}")

    def _reverse_vote(self, cluster, b_right, b_left, wdx_r, wdx_l):
        """按 ← 之后，这一簇里有多少块**相对背景**往左走了 —— 簇内多块投票。

        为什么不能沿用「同一块像素内容必须原样匹配」：实测本地玩家会被它误杀。
        Transformice 的老鼠**换行走方向时精灵水平镜像翻转**，加上逐帧动画，同一块
        像素内容 0.25 秒后本来就变了——正确候选反向只有 score=0.292，改用镜像模板
        重试也只到 0.726，都过不了 IDENTITY_MIN_SCORE=0.80 的门槛（对动画小精灵
        这个判据根本不成立）。所以改成：单块允许低分、允许镜像，但**要多块一致
        左移**才算数；背景整体滚动是相对背景 == 0（被减掉），独立移动的杂物/别的
        玩家凑不出这么多票。

        返回 (票数, 有效块数, 明细[(ax, ay, score, rel), ...])。
        """
        g = b_left[1]
        h, w = g.shape
        half = PATCH_SIZE // 2
        votes, tried, detail = 0, 0, []
        for p in cluster:
            ax, ay, dx, dy, _sc = p
            # (ax,ay) 是 A 帧坐标，位移 dx/dy 之后就是它在 B 帧里的位置
            ccx, ccy = ax + dx + half, ay + dy + half
            if not (half <= ccx <= w - half and half <= ccy <= h - half):
                continue
            tmpl = b_right[1][ccy - half:ccy + half, ccx - half:ccx + half]
            x0, y0 = max(0, ccx - REV_SEARCH), max(0, ccy - REV_SEARCH)
            x1, y1 = min(w, ccx + REV_SEARCH), min(h, ccy + REV_SEARCH)
            if x1 - x0 < PATCH_SIZE or y1 - y0 < PATCH_SIZE:
                continue
            win = g[y0:y1, x0:x1]
            best_mx, best_rel = -1.0, 0
            for cand in (tmpl, cv2.flip(tmpl, 1)):     # 换方向后精灵会镜像
                r = cv2.matchTemplate(win, cand, cv2.TM_CCOEFF_NORMED)
                _, mx, _, ml = cv2.minMaxLoc(r)
                if mx > best_mx:
                    best_mx = mx
                    best_rel = (x0 + ml[0]) - (ccx - half) - wdx_l
            tried += 1
            if best_mx >= REV_MIN_SCORE and best_rel <= -IDENTITY_MIN_DX:
                votes += 1
            detail.append((ax, ay, float(best_mx), int(best_rel)))
        return votes, tried, detail

    def _identify_player(self):
        """输入相关性定位本地玩家：按 → 谁**相对背景**往右跑就是谁，再按 ← 交叉验证。

        **两条路**：先试 `_identify_by_label`（自己的金色名字标签，只在相机静止的短图
        上成立），不成立再走下面这套 48x48 像素块匹配（长图/相机跟随时靠它）。短图正是
        老路子彻底失明的情形——玩家精灵只有 ~40x20，切成 48x48 的块之后成分几乎全是静止
        背景，dx≈0 挤不进候选；真正动的那点又会被树叶/木板的周期纹理假峰顶掉。

        颜色法在自定义皮肤下失效（掩膜命中的是草地/UI/树花），所以这里完全不看
        颜色，只做像素块位移匹配。

        关键：**相机在短图里静止、在长图里跟随玩家**（用户实测：短图背景不动，
        长图背景整片跟着滚）。所以「位移 dx≥+12 的块」在短图里是玩家、在长图里
        却是往左滚的世界本身——直接按绝对位移找会找错甚至找不到。改用**相对背景
        的位移**：先估出背景整体怎么动的（`_world_dx`，短图≈0、长图≈-26），再找
        相对背景往右跑的那批块，两种图都成立。

        但多人房间里别的玩家/机关也在动，「相对背景往右」不能证明是我们的键，
        所以还要反向验证：按 ← 后同一块内容必须相对背景往左走（我们按的键只对
        自己生效）。按候选簇逐个试，第一个通过的才是本地玩家。
        返回 True 表示已锁定；False 表示没有一个候选响应按键。
        """
        # 马上要按 → 探针了，顺便用这段位移把内存里的玩家结构体绑定好（同一份
        # 「谁跟着按键动」的判据，白捡）。扫描很贵，_mem_rescan 自己会限流。
        self._mem_rescan()
        # 每次标定都重新抢一次前台并点画布。只靠 start() 那次是不够的：换地图/回合切换
        # 时 Flash 会丢键盘焦点，而下面那个「moved 为空才重试」的分支永远轮不到——树叶、
        # 植被这种周期纹理会造出一堆高分假位移，moved 从来不为空，于是焦点丢了也没人管，
        # 表现就是「按键进了系统输入队列（GetAsyncKeyState 看得到）但角色纹丝不动」。
        # 放在 _mem_rescan 之后：扫描要 4s 多，焦点得在按键前一刻拿。
        # ⚠ 抢不到就**别再往下按探针**：这些方向键会整批灌进当前前台窗口/系统输入队列，
        # 而角色纹丝不动（2026-10-10 那次卡死之后就是这个状态）。而且 focus 返回 False
        # 最常见的两种原因（抢前台卡住 = 客户端已经死了；被别的窗口占住）都不是多按几次能好转的。
        if not self.cap.focus():
            print("[CALIB] 抢不到游戏前台（或抢前台卡住）→ 方向键进不去游戏，本次标定放弃")
            return False
        mb = self.mem.read_many(self._mem_players) if self.use_mem else {}
        a_right, b_right, m_right = self._probe_move("right")
        if self._frozen(a_right, b_right):
            # 整屏一帧没变 = 根本没在渲染活的游戏。后面还有 1 次重试探针 + 1 次左探针
            # + 最多 4 次验证探针，全都会白白灌进这个界面——实测 C 盘账号被封那一个
            # 半小时里就是这么灌了 66287 帧方向键进去的。直接结束本次尝试。
            print("[CALIB] 按键前后整屏零变化 → 不是在游戏里（弹窗/登录页/窗口卡死/已最小化），"
                  "本次放弃，不再继续灌按键")
            return False
        wdx_r = self._world_dx(m_right)
        moved = [p for p in (m_right or [])
                 if p[4] >= IDENTITY_MIN_SCORE and p[2] - wdx_r >= IDENTITY_MIN_DX]
        if not moved:
            # 没有块跟着 → 可能只是焦点被别的窗口抢走了（按键进了终端而不是游戏）。
            # 重切一次前台再试一遍；若还是空，才是真的不在可玩状态
            # （回合间隙/已进洞/幽灵，见 project_playable_states）。
            # 抢不到前台就跳过这次重探针：再按也是灌进别的窗口。下面的「金色名字标签」
            # 那条路照走（它用的是已有的三帧，不额外按键）。
            if self.cap.focus():
                mb = self.mem.read_many(self._mem_players) if self.use_mem else {}
                a_right, b_right, m_right = self._probe_move("right")
                wdx_r = self._world_dx(m_right)
                moved = [p for p in (m_right or [])
                         if p[4] >= IDENTITY_MIN_SCORE and p[2] - wdx_r >= IDENTITY_MIN_DX]
        self._calib_debug(a_right, b_right, m_right, moved, wdx_r)
        if self.use_mem and self._mem_players and not self._mem_player_addr:
            self._bind_mem_player(mb, self.mem.read_many(self._mem_players))
        if b_right is None:
            return False

        _, b_left, m_left = self._probe_move("left")
        if b_left is None:
            return False
        # 短图先走「金色名字标签」这条路（老路子在短图上锁不住，见 _identify_by_label）。
        # 复用上面两次探针的帧，不额外按键：A=按→之前、b_right=按完→、b_left=按完←。
        # **必须排在 moved 为空的那个 return 前面**：短图上老路子失效的表现恰恰就是
        # moved 为空（玩家那块 patch 几乎全是静止背景），先返回就永远轮不到这条路。
        if self._identify_by_label(a_right, b_right, b_left, wdx_r):
            return True
        if not moved:
            return False
        wdx_l = self._world_dx(m_left)
        g = b_left[1]
        h, w = g.shape
        half = PATCH_SIZE // 2

        cands = self._clusters(moved)
        # 屏幕固定的 HUD（顶栏计时/比分/地图名横幅）也满足「相对背景右移」，但它不是
        # 玩家。不管短图长图都把它排到后面，优先试画面里的，全失败才轮到它
        # （避免真玩家站在高处时被一刀切掉）。背景视差/重复纹理会造出一堆假簇，
        # 所以这里不再是「只对长图生效」。
        top_lim = int(h * 0.13)
        cands.sort(key=lambda grp: min(p[1] for p in grp) < top_lim)
        cands = cands[:CALIB_MAX_CANDIDATES]
        for ci, cluster in enumerate(cands):
            anchor = max(cluster, key=lambda p: p[4])   # 簇里内容最可信的一块当锚点
            votes, tried, detail = self._reverse_vote(cluster, b_right, b_left, wdx_r, wdx_l)
            need = max(REV_VOTE_MIN, (tried + 2) // 3) if tried else REV_VOTE_MIN
            top = sorted(detail, key=lambda d: -d[2])[:3]
            print(f"[CALIB]   候选#{ci} 锚点@({anchor[0]},{anchor[1]}) 相对右移="
                  f"{anchor[2] - wdx_r:+d} 簇{len(cluster)}块 → 反向投票 {votes}/{tried}"
                  f"（需 {need}）top {[(round(d[2], 2), d[3]) for d in top]}")
            # 簇小时凑不出票数门槛：1 块最多投 1 票，need 却恒 ≥2，于是「单块高分反向」这种
            # 很强的证据永远过不了（实测 run17：score=0.91/rel=-22、0.79/rel=-40 都被卡掉）。
            # 小簇（≤2 块）允许一块高分反向顶票数；大簇仍按票数走——移动平台那种「多块一致、
            # 偶有一块反向」的簇不能靠单块开门。
            strong = any(d[2] >= REV_STRONG_SCORE and d[3] <= -IDENTITY_MIN_DX for d in detail)
            if votes < need and not (tried <= 2 and strong):
                continue
            # 反向验证过了，用 A 帧里的簇心当锁定位置（(ax,ay) 就是 A 帧坐标）。
            # 按 → 再按 ← 之后角色基本回到原位，所以拿 A 帧坐标去 B_left 上取模板是对的
            # ——原来的代码也是这么用的（_find_body 返回的同样是 A 帧坐标）。
            lx = int(np.mean([p[0] for p in cluster])) + half
            ly = int(np.mean([p[1] for p in cluster])) + half
            lx = max(half, min(lx, w - half))
            ly = max(half, min(ly, h - half))
            size = PATCH_SIZE

            # 上面锁定的是「名字标签」那一块——它在老鼠正上方，位置好用但不等于老鼠。
            # 在 A/B 差异图上，标签和身体之间有竖直空档，所以标签下方那一坨变化像素
            # 就是老鼠身体：按它定位才能让模板底边=脚底（找地面/防掉落都靠这个）。
            found = self._find_body(a_right, b_right, cluster, anchor[2] - wdx_r, wdx_r)
            body, tag = found if found is not None else (None, None)
            if body is not None:
                bx, by, bw, bh = body
                lx, ly = bx + bw // 2, by + bh // 2
                size = int(min(SPRITE_MAX_PX, max(PATCH_SIZE, bh + 6)))
                print(f"[CALIB]   标签下方找到身体 ({bx},{by}) {bw}x{bh} → 锁定中心 ({lx},{ly}) 边长 {size}")
                if tag is not None:
                    print(f"[CALIB]   名字标签 {tag[2]}x{tag[3]} @({tag[0]},{tag[1]})"
                          f" 一并当锚点（标签是静态 UI，不吃精灵动画）")
            else:
                # 找不到标签下方的身体，说明这一簇根本不是「名字标签 + 身体」。
                # 实测 run9 锁定 (740,566)：那是**计分板上的文字**（截图核对过，红框落在
                # 玩家名单上），它之所以被选中是因为探针按键让相机滚了一段，而计分板屏幕
                # 固定、文后的世界动了，于是它「相对背景位移」达标。原来这里会退回标签位置
                # 下移一个身位硬锁，结果 pos 钉死 60 帧、[STUCK] 连锁 72 次。
                # 宁可这一轮标定失败（等下一个回合重标），也不要锁在一个不动的东西上。
                print(f"[CALIB]   候选#{ci} 标签下方找不到身体（不是玩家，可能是计分板/UI 文字）→ 弃用")
                continue

            if not self.vision.set_player_lock(g, lx, ly, size, bgr=b_left[0], tag=tag):
                print(f"[CALIB]   候选#{ci} 锁定处是平坦区域（模板会退化，实测天空块"
                      f"60px 外也拿 1.0000 分）→ 换下一个候选")
                continue
            if not self._verify_lock_moves():
                print(f"[CALIB]   候选#{ci} 锁上之后按键挪不动它（锁到了静止的东西）→ 换下一个候选")
                self.vision.clear_player_lock()
                continue
            self._calib_lock_shot(b_left[0], cluster, (lx, ly), size)
            print(f"[Bot] 输入标定成功：本地玩家 @ ({lx}, {ly})"
                  f"（候选#{ci} 相对右移 {anchor[2] - wdx_r:+d}px、"
                  f"反向投票 {votes}/{tried} 块一致左移，验证通过）")
            # 玩不了的时候（回间隙/已进洞）标定会失败，所以 y 方向标定放在这里：
            # 能走到这儿说明角色现在真的能操作，跳一下量出来的才是有效数据。
            if self.use_mem and self._mem_player_addr and self._mem_dy_sign is None:
                self._calib_mem_sign()
            return True

        print(f"[CALIB] 拒绝：{len(cands)} 个候选簇都没有跟着 ← 走（单次按键内没捕捉到反向）")
        return False

    @staticmethod
    def _mask_shift(mA, box, mB, rx=PROBE_SEARCH, ry=LABEL_SEARCH_Y):
        """在 B 的文字掩膜里找 box 处的字跑到哪去了，返回 (dx, dy, F1)；找不到返回 None。

        F1 = 2*命中 / (2*命中 + 多出来的 + 少掉的)。只在**文字像素掩膜**上比，所以半
        透明名字气泡底下压着的地形/天空在变也无所谓——这是它逐帧都能拿 1.00 的原因
        （拿原图比像素的版本只有 0.32~0.75，见 _track.py 的失败记录）。
        """
        x, y, w, h = box
        if not (0 <= x and 0 <= y and x + w <= mA.shape[1] and y + h <= mA.shape[0]):
            return None
        tpl = mA[y:y + h, x:x + w]
        ts = float(tpl.sum())
        if ts < 40:                       # 金像素太少，模板没信息，匹配分没有意义
            return None
        bx0, by0 = max(0, x - rx), max(0, y - ry)
        bx1, by1 = min(mB.shape[1], x + w + rx), min(mB.shape[0], y + h + ry)
        win = mB[by0:by1, bx0:bx1]
        if win.shape[0] < h or win.shape[1] < w:
            return None
        tp = cv2.matchTemplate(win, tpl, cv2.TM_CCORR)
        ker = np.ones((h, w), np.float32)
        tot = cv2.filter2D(win, -1, ker, anchor=(0, 0))[:tp.shape[0], :tp.shape[1]]
        f1 = 2 * tp / np.maximum(2 * tp + (tot - tp) + (ts - tp), 1e-6)
        _, best, _, loc = cv2.minMaxLoc(f1)
        return (bx0 + loc[0] - x, by0 + loc[1] - y, float(best))

    @staticmethod
    def _find_body_at(a_bgr, b_bgr, x0, x1, y_start, dx, limit=200):
        """在 A/B 差异图里找 (x0..x1, 从 y_start 往下) 的第一坨变化像素 = 老鼠身体。

        和 _find_body 的区别是「标签框已知」：不用从簇里猜标签在哪，直接把 y_start
        传成金色文字块的底边——名字和称号两行都是自己的金字，都会被探针匹上，从整块
        的底边往下扫才不会把称号那一行当成身体。

        A/B 差异会把 A 位置和 B 位置并成一条（横向拉长 |dx|），所以左边缘是 A 的左边
        缘、真实宽度要减掉 |dx|——和 _find_body 同一个道理。
        """
        ga = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
        if ga.shape != gb.shape:
            return None
        h, w = ga.shape
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        if x1 - x0 < 8:
            return None
        mask = cv2.absdiff(ga, gb)[:, x0:x1] > 25
        rows = mask.sum(axis=1)
        end = min(h, int(y_start) + limit)
        y = max(0, int(y_start))
        while y < end:
            if rows[y] > 3:
                ys = y
                while y < end and rows[y] > 3:
                    y += 1
                bh = y - ys
                cols = np.nonzero(mask[ys:y].sum(axis=0))[0]
                if len(cols):
                    inner = max(8, int(cols[-1] - cols[0] + 1) - abs(dx))
                    if 14 <= inner <= 110 and 14 <= bh <= 110:
                        return (int(x0 + cols[0]), ys, inner, bh)
                    print(f"[CALIB]   标签下方 {inner}x{bh} 不像老鼠身体（需 14~110）→ 继续往下找")
            y += 1
        return None

    def _calib_fail_dump(self, a, b1, b2, rows):
        """把标定失败时用的这三帧原样存盘，并把候选框和量到的 dx/F1 画上去。

        目的只有一个：**能看**。之前失败时只留几行统计值，只能靠推理猜是哪种失败
        （被挡住？不是我们的键？标签没检出？），而推理错过好几次。存成固定文件名
        覆盖，不攒垃圾。
        """
        d = os.path.dirname(os.path.abspath(__file__))
        try:
            for nm, r in (("a", a), ("b1", b1), ("b2", b2)):
                if r is not None:
                    save_png(os.path.join(d, f"_calib_fail_{nm}.png"), r[0])
            img = a[0].copy()
            for box, _why, r1, r2 in rows:
                cv2.rectangle(img, (box[0], box[1]),
                              (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 1)
                lab = f"r={r1[0]:+d}/{r1[2]:.2f}" if r1 is not None else "r=None"
                if r2 is not None:
                    lab += f" l={r2[0]:+d}/{r2[2]:.2f}"
                cv2.putText(img, lab, (max(0, box[0] - 20), max(10, box[1] - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            save_png(os.path.join(d, "_calib_fail_annot.png"), img)
            save_png(os.path.join(d, "_calib_fail_gold.png"),
                        self.vision.gold_mask(a[0]).astype(np.uint8) * 255)
            print(f"[CALIB]   失败帧已存盘 _calib_fail_a/b1/b2.png + _calib_fail_annot.png"
                  f"（{len(rows)} 个候选框已标注，看一眼就知道是哪种失败）")
        except Exception as e:
            print(f"[CALIB]   失败帧存盘出错：{e}")

    def _lock_by_stored_template(self, a):
        """**先问存下来的名字字形**：它在这一帧上高分且独一无二，就地锁定，一个键都不用按。

        这才是「身份」的正确用法。原来那段「字形在整屏上独一无二 → 跳过按键验证」是拿
        **候选**字形去量唯一性——别人的名字在画面上同样独一无二，于是 21:42 那次把一块
        54x9 的灰字（别人的名字）认成了自己存了盘，之后每次开机都锁到别人身上。
        存下来的字形是**我们自己的笔画**，它对上了才算数，对不上就老老实实按键标定。
        """
        if self.vision._label_body_off is None:
            return False
        _gc = self.vision.label_confidence(a[0])
        if _gc is None or _gc[0] < LABEL_TRUST_SCORE \
                or _gc[0] - _gc[1] < LABEL_TRUST_MARGIN:
            return False
        got = self.vision.reacquire_by_label(a[0])
        if got is None:
            return False
        cx, cy, size, tag = got
        if not self.vision.set_player_lock(a[1], cx, cy, size, bgr=a[0], tag=tag):
            print("[CALIB]   存下来的标签位置是平坦区域 → 不采用")
            return False
        self._calib_lock_shot(a[0], [], (cx, cy), size)
        print(f"[Bot] 输入标定成功（沿用存下来的名字字形）：本地玩家 @ ({cx}, {cy})，"
              f"标签 {tag[2]}x{tag[3]} @({tag[0]},{tag[1]})"
              f"（峰 {_gc[0]:.2f} / 次峰 {_gc[1]:.2f}，没按键）")
        return True

    def _identify_by_label(self, a, b1, b2, wdx_r):
        """用「自己名字标签」定位本地玩家——**短图（相机静止）上唯一靠谱的锚点**。

        为什么需要它：老路子把画面切成 48x48 的块去找「按 → 谁往右跑」，可短图里相机
        不动、玩家精灵只有 ~40x20，它那块 patch 的成分几乎全是静止背景，测出来 dx≈0
        根本进不了候选池；真正在动的那少数派又被树叶/木板的周期纹理假峰顶掉。实测
        （_hunt.py，短图）老路子一个都锁不住，而**名字标签是静态文字**，逐帧匹配分
        1.00、次高只有 0.35，一找一个准。

        判据是双向的：同一个金色文字行，从 A 到「按住 → 之后」要右移 ≥IDENTITY_MIN_DX，
        从「按住 → 之后」再到「按住 ← 之后」要左移 ≥IDENTITY_MIN_DX（回到原处不算，
        必须真的反向走）。全屏只有一样东西听我们的键，所以这条件同时排掉了静止的暖色
        美术（狐狸尾巴/沙地/花）和别的玩家。

        只在相机静止（wdx_r == 0）时用：相机跟着滚的时候标签在屏幕上几乎不动，这套
        屏幕坐标判据不成立——而那种图老路子本来就能锁住，交给它就好（秒退，不浪费探针）。
        """
        # 存档的字形先在整幅舞台上量一把：对得上就一步到位（长短图都适用，不按键）。
        if self._lock_by_stored_template(a):
            return True
        if wdx_r != 0:
            return False
        cands = self.vision.detect_own_labels(a[0])
        if not cands:
            print("[CALIB]   金色名字标签：一个候选都没有")
            return False
        mA = self.vision.gold_mask(a[0]).astype(np.float32)
        m1 = self.vision.gold_mask(b1[0]).astype(np.float32)
        m2 = self.vision.gold_mask(b2[0]).astype(np.float32)
        hits = []
        rows = []
        for box in cands:
            r1 = self._mask_shift(mA, box, m1)
            why = ''
            r2 = None
            if r1 is None:
                why = '掩膜太稀/越界'
            else:
                # 第二段用「按完 → 之后」的那帧当参照（不是 A）：A 只证明它往右走过，
                # 再看按 ← 之后它往哪走，才排除掉「被相机/别人的动作带着走」这类假象。
                box1 = (box[0] + r1[0], box[1] + r1[1], box[2], box[3])
                r2 = self._mask_shift(m1, box1, m2)
                if r2 is None:
                    why = '第二段掩膜太稀/越界'
            if not why:
                dxr, dxl = r1[0], r2[0]
                okr, okl = r1[2] >= LABEL_MIN_SCORE, r2[2] >= LABEL_MIN_SCORE
                # **任一方向跟手就算数**：角色被墙/平台挡住时，按 → 只能挪几像素
                # （实测 dx=+6、F1=1.00）而按 ← 走得动。原来要求「右移 ≥12 且左移
                # ≤-12」两条都得满足，这种好标签会被整个扔掉。2026-09-19 实跑里
                # 6 次标定全倒在 dx=+6 上（日志 `_live_run.log` 07:53:55）。
                fwd_r = okr and dxr >= IDENTITY_MIN_DX
                fwd_l = okl and dxl <= -IDENTITY_MIN_DX
                # 反着走才真可疑：按 → 却往左、按 ← 却往右 = 它不听我们的键（别的玩家、
                # 相机滚动的世界）。这种无论另一个方向多漂亮都必须否掉——它同时也
                # 挡住了「哪段 F1 高就用哪段」的误判。
                back_r = okr and dxr <= -IDENTITY_MIN_DX
                back_l = okl and dxl >= IDENTITY_MIN_DX
                if not (fwd_r or fwd_l):
                    why = (f'两段都没有 ≥{IDENTITY_MIN_DX}px 跟手位移'
                           f'（右 {dxr:+d}/F1={r1[2]:.2f} 左 {dxl:+d}/F1={r2[2]:.2f}）')
                elif back_r or back_l:
                    why = f'反向矛盾（右 {dxr:+d} 左 {dxl:+d}）不听我们的键'
            rows.append((box, why, r1, r2))
            if why:
                print(f"[CALIB]     {box} → {why}")
                continue
            print(f"[CALIB]     {box} → 右 {r1[0]:+d}(F1={r1[2]:.2f}) "
                  f"左 {r2[0]:+d}(F1={r2[2]:.2f})  "
                  f"{'右向作准' if fwd_r else '左向作准（右边被挡住）'}  命中")
            hits.append((box, r1, r2))
        print(f"[CALIB]   金色名字标签：{len(cands)} 个候选，跟着按键动的 {len(hits)} 个"
              f"（{[(h[0][:2], h[1][0], h[2][0]) for h in hits[:4]]}）")
        if not hits:
            self._calib_fail_dump(a, b1, b2, rows)
            return False
        # 名字一行 + 称号一行都会入选（两行都是自己的金字、都跟着走）。取它们并起来
        # 的整块当锚点：身体一定在这整块的下方。
        #
        # **但只能并「同一块文字」的那些**。把凡是跟手的金块全并进来，会拼出一个横跨
        # 半屏的巨框——2026-09-28/29 实跑就是这么拼出 370x61、618x2、365x9 这些伪标签，
        # 其中一次进的模板库条目是 736 宽（[LOCK] 建库 偏移 (339,42,36,736)），搜索窗
        # 随即被撑到整幅（[LOCK] 搜索窗异常大），灰度模板在哪儿都能凑出高分、掩膜也总能
        # 找到金像素，锁立刻被拽走 → 秒丢。那几场「刚锁上就被丢」的存活中位只有 4~7.5s。
        # 判据用「位移一致 + 水平挨着」：真身那两行的 (右移, 左移) 几乎逐像素相同，
        # x 也只差几个像素，而远处那块假金字的位移是另一套数。
        best = max(hits, key=lambda h: h[1][2])
        dx_r, dx_l = best[1][0], best[2][0]
        # 左右位移必须**近似对称**：同一个物体被按右推 +39px、按左推 -32px，比值
        # 0.7~1.0；而「两块不相干的文字被误并成一块」实测出过 +52 / -110（比值 2.1）。
        # 不对称说明这块根本不是跟着按键走的那个东西，拿它当锚点整局都会锁在别处。
        # 2026-10-26 23:5x：下界从 0.55 放到 0.15。实测正身被墙/台阶挡住时，
        # 「按 → 走 +102、按 ← 只回来 -34」（比值 0.33）—— 这**正是真人**，
        # 却被上界那套「近似对称」骗过去扔掉了：23:00 那次标定唯一的命中
        # (583,153,49,9) 就是这么被否的，之后只能退到更差的重挂。
        # 上界 1.8 留着：那才是「两块不相干的文字被误并」的症状（实测 +52/-110）。
        if not (8 <= dx_r <= 160 and -160 <= dx_l <= -8
                and 0.15 <= (-dx_l) / float(dx_r) <= 1.8):
            print(f"[CALIB]   最佳标签块 {best[0]} 左右位移不对称"
                  f"（右 {dx_r:+d} / 左 {dx_l:+d}）→ 不是跟着按键走的文字，弃用")
            self._calib_fail_dump(a, b1, b2, rows)
            return False
        grp = [h for h in hits
               if abs(h[1][0] - dx_r) <= LABEL_TAG_AGREE
               and abs(h[2][0] - dx_l) <= LABEL_TAG_AGREE
               and abs(h[0][0] - best[0][0]) <= LABEL_TAG_MAX_DX
               and abs(h[0][1] - best[0][1]) <= LABEL_TAG_MAX_DY]
        if len(grp) < len(hits):
            print(f"[CALIB]   并标签时剔掉 {len(hits) - len(grp)} 个位移/位置跟最佳命中不一致的"
                  f"（保留 {[h[0][:2] for h in grp]}）")

        def _bbox(gs):
            x0 = min(h[0][0] for h in gs)
            y0 = min(h[0][1] for h in gs)
            x1 = max(h[0][0] + h[0][2] for h in gs)
            y1 = max(h[0][1] + h[0][3] for h in gs)
            return (x0, y0, x1 - x0, y1 - y0)

        tag = _bbox(grp)
        # 兜底一：并出来的框还是超限（真名字旁边另有一块位移恰好一致的假金字，竖直也没超），
        # 缩到「最佳那一块自己」。单行标签本来就能当锚点——实测成功的例子里 36x9、71x13
        # 都是单块；没必要因为旁边有块跟手的假金字就把整次标定作废。
        if (tag[2] > LABEL_TAG_MAX_W or tag[3] > LABEL_TAG_MAX_H) and len(grp) > 1:
            print(f"[CALIB]   并出来的标签块 {tag} 超过上限 "
                  f"{LABEL_TAG_MAX_W}x{LABEL_TAG_MAX_H} → 缩到最佳那一块 {best[0]}")
            grp = [best]
            tag = _bbox(grp)
        # 兜底二：单块自己就超限（伪标签 736x36 那种），不能拿它当锚点——进库就是
        # 「搜索窗撑爆 → 模板在哪儿都能凑高分 → 秒丢」那条链。
        if tag[2] > LABEL_TAG_MAX_W or tag[3] > LABEL_TAG_MAX_H:
            print(f"[CALIB]   并出来的标签块 {tag} 尺寸超过上限 "
                  f"{LABEL_TAG_MAX_W}x{LABEL_TAG_MAX_H} → 弃用，不拿它当锚点")
            self._calib_fail_dump(a, b1, b2, rows)
            return False
        gx0, gy0 = tag[0], tag[1]
        gx1, gy1 = tag[0] + tag[2], tag[1] + tag[3]
        cx = (gx0 + gx1) // 2
        pad = 60 + abs(dx_r)
        body = self._find_body_at(a[0], b1[0], cx - pad, cx + pad, gy1 + 2, dx_r)
        if body is not None and (body[2] > BODY_MAX_W or body[3] > BODY_MAX_H):
            print(f"[CALIB]   差分量出的身体框 {body[2]}x{body[3]} 比一只精灵大得多"
                  f"（上限 {BODY_MAX_W}x{BODY_MAX_H}）→ 差分被整堆老鼠占满，按标签下方经验值估")
            body = None
        if body is None:
            # 房间里挤了一堆老鼠时（官方房常见），差分图里「动过的」是整堆人，量不出
            # 我们这一只。这不影响跟踪——模板库里真正管用的是名字标签那块静态文字，
            # 身体框只决定上报的脚底 y，用标签下方的经验值估一个就够（会偏，但不致命）。
            bw = max(28, min(60, tag[2] + 8))
            bh = LABEL_BODY_H
            body = (cx - bw // 2, gy1 + LABEL_BODY_GAP, bw, bh)
            print(f"[CALIB]   人群里分不出身体（差分被整堆老鼠占满）→ 按标签下方经验值估 "
                  f"({body[0]},{body[1]}) {bw}x{bh}")
        bx, by, bw, bh = body
        lx, ly = bx + bw // 2, by + bh // 2
        size = int(min(SPRITE_MAX_PX, max(PATCH_SIZE, bh + 6)))
        print(f"[CALIB]   金色标签块 {tag} 双向跟手（右 {dx_r:+d} / 左 {dx_l:+d}），"
              f"下方身体 ({bx},{by}) {bw}x{bh} → 锁定中心 ({lx},{ly}) 边长 {size}")
        # 用 A 帧取模板：_find_body_at 量出来的就是 A 帧坐标（这时角色是静止站姿）。
        if not self.vision.set_player_lock(a[1], lx, ly, size, bgr=a[0], tag=tag):
            print("[CALIB]   锁定处是平坦区域（模板会退化）→ 弃用")
            return False
        # 走到这里说明**存档字形没对上**（或者压根没有），那就只认按键验证。
        # ⚠ 不要再用「候选字形在整屏上独一无二」当通行证：别人的名字同样独一无二，
        # 2026-10-26 21:42 就是这么把一块 54x9 的灰字（别人的名字）认成自己、存了盘，
        # 之后每次开机都锁到别人身上。唯一性证明不了「是不是我」。
        if not self._verify_lock_moves():
            print("[CALIB]   金色标签锁上之后按键挪不动它 → 弃用")
            self.vision.clear_player_lock()
            return False
        self._calib_lock_shot(a[0], [], (lx, ly), size)
        # 只有**按键标定 + 反向验证都过了**的这一刻才把名字字形存盘：这份模板会在下次
        # 开机时用来「不按键直接重挂」，存错一次就会一直错。重挂那条路不存（见 vision）。
        self.vision.save_label_tpl(a[0])
        print(f"[Bot] 输入标定成功（金色名字标签法）：本地玩家 @ ({lx}, {ly})，"
              f"标签 {tag[2]}x{tag[3]} @({tag[0]},{tag[1]})")
        if self.use_mem and self._mem_player_addr and self._mem_dy_sign is None:
            self._calib_mem_sign()
        return True

    def _verify_lock_moves(self, min_move=10, hold_s=0.30, tries=4, min_score=0.70):
        """锁定后按方向键探几下，确认模板在**屏幕坐标**上真的跟着按键走。

        判据三条，都是从实测数据里划的线（run10/11/12 的好锁 vs 坏锁）：

        1. **相机必须静止**（`_world_dx == 0`）。相机一跟着滚，屏幕坐标就分不开「角色被
           相机跟住」和「一块屏幕固定的 UI」——实测 run9 锁定 (740,566)（计分板文字），
           按 right 后模板 Δ=0、背景 Δ=-21，减完背景 rx=+21 照样过。
        2. **匹配分要够高**（≥ min_score）。好锁实测 0.89~0.94，坏锁 0.43~0.60 —— 低分时
           「位移」只是模板匹配在别处瞎猜一个位置，不是角色在动。
        3. **位移必须是水平的、且朝按键方向**，|dy| 要小。好锁实测 dy 恒为 +0；run12
           锁定 (486,339) 那次按 right 却报 `dx=-16 dy=-92`（模板往上跳了 92px），
           原来那句 `or abs(dy) >= min_move` 把它当成「验证通过」，锁的其实是个死物
           （随后 60 帧 [STUCK]）。

        相机在滚的探针判**无效**：既不算通过也不算失败，换方向再试——角色在死区一侧时，
        往反向按总能把它送回「相机不动」的区间。同理「分太低」的探针也判**无效**（见
        下面 score < min_score 那段的说明）：它量不出结论，不能算作锁是坏的证据。

        **返回值是三态**：True=验证跟手；False=量得到、但就是不跟手（确定的坏锁）；
        None=四次探针全都量不了（相机一直在滚 / 一直没等到新帧 / 分数一直太低）→ **不做判断**。
        长图里相机跟着角色滚，「相机静止」这条永远不成立，返回 None 才不会把好锁冤杀。
        两个调用点用的都是 `if not ...`，None 和 False 一样被当成「没过」，行为不变。
        调用方要区分「确定坏」和「判不了」时才需要看 None——见主循环里 STUCK 引发的那次复核。

        按住时长 0.10s → **0.30s**：0.10s 从静止起步只走 ~7px，卡在 min_move=10 的线
        下面，好锁也会被判「位移不足」扔掉（实测 _run2.log 07:43:41：金色标签法标定
        成功、锁在 (577,433)，4 次验证探针量到 +7/+1/+2/−3 全不及格 → 白白弃锁）。
        """
        p0 = self.vision._last_player
        if p0 is None:
            return False
        # 先把所有键松开再探：两个调用点（标定后、STUCK 复核）都可能在**按着键**的状态下
        # 进来，而探针只按 left/right。特别是主循环现在会把 up **按住**（连续跳，见
        # Movement 那段），up 一按着老鼠就在跳，探针量出来的 dy 全是十几像素，判据里
        # `abs(dy) <= 8` 直接不过 → 好锁被自己的验证环节冤杀。
        self.release_all()
        tried = []
        judgeable = False     # 有没有哪一次真的量到了「相机静止 + 高分 + 模板匹配上」，见返回值说明
        for i in range(tries):
            key = "right" if i % 2 == 0 else "left"
            _, r1, m1 = self._probe_move(key, hold_s=hold_s)
            if r1 is None:
                return True          # 等不到新帧，判不了，别否决
            # 搜索窗放大到 120：验证时按得短、但角色速度快，±64 可能已经跑出去了
            # 掩膜条目（金色名字标签）必须喂当前帧的金色掩膜，否则库里最强的那块锚点
            # 会被跳过，只剩灰度身体模板——实测那会量到 0.56（< min_score 0.70），
            # 好锁被自己的验证环节否掉。
            p1 = self.vision.track_locked(r1[1], search=120, bgr=r1[0],
                                          gold=self.vision.text_mask(r1[0]).astype(np.float32))
            score = self.vision._last_score
            carry = self.vision._lock_lost     # >0 表示这帧其实是「沿用上一帧」，没真匹配上
            if p1 is None or carry:
                tried.append(f"{key}:没匹配上({score:.2f})")
                continue
            wdx = self._world_dx(m1)
            if wdx != 0:
                tried.append(f"{key}:相机在滚({wdx:+d})")
                continue
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            ok_x = dx >= min_move if key == "right" else dx <= -min_move
            if score < min_score:
                # 分太低 = **这一次量不出结论**，不是「锁是坏的」。判据 2 的原话就是
                # 「低分时位移只是模板在别处瞎猜的位置」——既然不可信，就不能拿它当
                # 「不跟手」的证据。好锁在角色换姿势/起跳时实测也会掉到 0.68~0.70，
                # 正好压在这条线附近。2026-09-19 run10 实测：锁定框明明正套在自己老鼠
                # 身上（框内灰度 std=55.8、框上方 40px 有 166 个金像素——自己的金色名字
                # 标签就在头顶），却因为 4 次探针全是「分太低(0.68~0.70)」被判
                # 「锁在不响应的东西上」丢掉，接着退避一路涨到 60s。
                tried.append(f"{key}:分太低({score:.2f} dx={dx:+d})")
                continue
            # 走到这儿才是「相机静止 + 高分 + 模板匹配上」→ 位移可信，这次探针有资格下结论
            judgeable = True
            if ok_x and abs(dy) <= 8:
                print(f"[CALIB]   锁定验证通过：相机静止，按 {key} 模板 ({p0[0]},{p0[1]})"
                      f"→({p1[0]},{p1[1]}) dx={dx:+d} dy={dy:+d} score={score:.2f}")
                return True
            tried.append(f"{key}:位移不对(dx={dx:+d} dy={dy:+d} score={score:.2f})")
        verdict = ("锁在不响应的东西上" if judgeable
                   else "判不了（相机在滚 / 没等到新帧 / 分数一直太低）")
        print(f"[CALIB]   锁定验证{'失败' if judgeable else '给不出结论'}："
              f"{tries} 次探针都没拿到「相机静止 + 高分 + 水平跟手」"
              f"（{'、'.join(tried)}）→ {verdict}")
        return False if judgeable else None

    @staticmethod
    def _find_body(a, b, cluster, dx, world_dx=0):
        """在 A/B 差异图上找「标签下方那一坨变化像素」= 老鼠身体，返回 A 帧坐标的
        (x, y, w, h)，找不到返回 None。

        标签和身体之间有一道竖直空档（字底到头顶），所以按行统计变化像素数、
        找标签之后的下一个连续非空行区间即可。x 上差异图会把 A 位置和 B 位置并成
        一条（横向拉长 |dx|），所以左边缘就是 A 的左边缘、真实宽度要减掉 |dx|。

        dx 是**相对背景**的位移；长图里背景整片滚了 world_dx，直接差分满屏都是
        变化、找不到身体，所以先把 B 帧按 -world_dx 平移回去对齐背景，剩下的差异
        才是玩家自己动的部分。
        """
        if a is None or b is None:
            return None
        ga = cv2.cvtColor(a[0], cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b[0], cv2.COLOR_BGR2GRAY)
        if world_dx:
            gb = np.roll(gb, -world_dx, axis=1)
        h, w = ga.shape
        ax0 = min(p[0] for p in cluster)
        ay0 = min(p[1] for p in cluster)
        ax1 = max(p[0] for p in cluster) + PATCH_SIZE
        ay1 = max(p[1] for p in cluster) + PATCH_SIZE
        cx = (ax0 + ax1) // 2
        x0, x1 = max(0, cx - 90), min(w, cx + 90)
        mask = cv2.absdiff(ga, gb)[:, x0:x1] > 25
        rows = mask.sum(axis=1)
        limit = min(h, ay1 + 170)
        runs = []
        y = ay0
        while y < limit:
            if rows[y] > 3:
                ys = y
                while y < limit and rows[y] > 3:
                    y += 1
                runs.append((ys, y))
            y += 1
        # runs[0] 是标签；标签下方第一个够高的区间就是身体
        for ys, ye in runs[1:]:
            if ye - ys < 8:
                continue
            cols = np.nonzero(mask[ys:ye].sum(axis=0))[0]
            if len(cols) == 0:
                continue
            span = int(cols[-1] - cols[0] + 1)
            inner = max(8, span - abs(dx))
            bh = ye - ys
            # 尺寸合理性：老鼠精灵在这个分辨率下大约 40~60px。实测这一坨变化像素会
            # 被拉成 146x139（整块平台/视差跟着变），照它去锁模板，48~80px 的方块里
            # 老鼠只占一小角、背景占大头，匹配自然漂到背景上（实测按 → 却向下跳 74px）。
            # 超过上限就当作「没找到身体」，让上层用标签位置 + 身体偏移兜底。
            if not (14 <= inner <= 110 and 14 <= bh <= 110):
                print(f"[CALIB]   标签下方的变化区 {inner}x{bh} 不像老鼠身体（超出 14~110）→ 弃用")
                return None
            bx = int(x0 + cols[0])
            return (bx, ys, inner, bh), Bot._tag_box(mask, runs[0], x0, abs(dx))
        return None

    @staticmethod
    def _tag_box(mask, run, x0, dx):
        """把 runs[0]（名字标签那一行区间）变成 A 帧坐标的 (x,y,w,h)。

        标签是**不动的 UI 文字**（不逐帧动画、不随朝向镜像），所以它比老鼠精灵更适合
        当跟踪锚点：离线拿被动截的 80 帧量过，身体模板 0.93~1.00 来回抖（走路动画），
        标签模板**全程 1.00**。代价是它会被别的老鼠标签压住，所以身体模板照留不误，
        两边都放进模板库取最大值。"""
        ys, ye = run
        rows = mask[ys:ye]
        cols = np.nonzero(rows.sum(axis=0))[0]
        if len(cols) == 0:
            return None
        # 差异图把 A 位置和 B 位置并成一条（横向拉长 |dx|），左边缘才是 A 的左边缘，
        # 真实宽度要减掉 |dx|——和 _find_body 里算身体宽度是同一个道理。
        w = max(8, int(cols[-1] - cols[0] + 1) - dx)
        h = ye - ys
        if not (20 <= w <= 220 and 8 <= h <= 60):
            return None
        return (int(x0 + cols[0]), ys, w, h)

    def _calib_lock_shot(self, bgr, cluster, lock, size=PATCH_SIZE):
        """把最终锁定位置画出来存盘，供人工翻图核对是否真的套在老鼠身上。"""
        d = os.path.dirname(os.path.abspath(__file__))
        marked = bgr.copy()
        for p in sorted(cluster, key=lambda p: (p[1], p[0])):
            ax, ay, dx, dy, sc = p
            print(f"[CALIB]   簇块 at ({ax:4d},{ay:4d}) dx={dx:+3d} dy={dy:+3d} score={sc:.3f}"
                  f" → B({ax + dx:4d},{ay + dy:4d})")
            cv2.rectangle(marked, (ax, ay),
                          (ax + PATCH_SIZE, ay + PATCH_SIZE), (255, 0, 255), 1)
        half = size // 2
        lx, ly = lock
        cv2.rectangle(marked, (lx - half, ly - half), (lx + half, ly + half), (0, 0, 255), 2)
        save_png(os.path.join(d, "_calib_lock.png"), marked)

    def _calib_debug(self, a, b, matches, moved, world_dx=0):
        """把一次标定探针的原始证据落盘：命中块数、位移分布、标注图。

        用来区分「真的锁定到本地玩家」和「相机滚动/大物体」等假阳性——只看统计值
        会误判，必须能翻截图核对（见 feedback_investigate_not_guess）。"""
        d = os.path.dirname(os.path.abspath(__file__))
        # 位移直方图：大量 patch 共享同一个 dx = 整屏平移。短图里这是 0（相机静止），
        # 长图里这是背景滚动量（相机跟随）——两种都正常，所以要显式打出来。
        hist = {}
        for p in matches or []:
            hist[p[2]] = hist.get(p[2], 0) + 1
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:6]
        cam = "短图/相机静止" if world_dx == 0 else f"长图/相机跟随（背景滚 {world_dx:+d}px）"
        print(f"[CALIB] 命中 {len(matches or [])} 块，相对背景右移(≥{IDENTITY_MIN_DX}px) {len(moved)} 块；"
              f"{cam}；dx 分布 top: {top}")
        for i, p in enumerate(sorted((moved or matches or []), key=lambda p: -(p[4]))[:5]):
            print(f"[CALIB]   #{i} at ({p[0]},{p[1]}) dx={p[2]:+d} dy={p[3]:+d} score={p[4]:.3f}"
                  f"（相对 {p[2] - world_dx:+d}）")
        if a is None or b is None:
            return
        save_png(os.path.join(d, "_calib_a.png"), a[0])
        save_png(os.path.join(d, "_calib_b.png"), b[0])
        # 标注图：把「移动了的块」画在原帧上，直接看命中的是什么物体
        marked = a[0].copy()
        for p in matches or []:
            ax, ay, dx, dy, sc = p
            col = (0, 0, 255) if abs(dx) >= IDENTITY_MIN_DX else (0, 255, 0)
            cv2.rectangle(marked, (ax, ay), (ax + PATCH_SIZE, ay + PATCH_SIZE), col, 1)
        for p in moved or []:
            ax, ay, dx, dy, sc = p
            cv2.rectangle(marked, (ax, ay), (ax + PATCH_SIZE, ay + PATCH_SIZE), (255, 0, 255), 2)
            cv2.arrowedLine(marked, (ax + PATCH_SIZE // 2, ay + PATCH_SIZE // 2),
                            (ax + PATCH_SIZE // 2 + dx, ay + PATCH_SIZE // 2 + dy),
                            (255, 0, 255), 2, tipLength=0.4)
        save_png(os.path.join(d, "_calib_marked.png"), marked)

    def _dump_frame(self, bgr, player, cheeses, holes, tgt):
        """把检测结果画在帧上存盘：红框=锁定的玩家，黄圈=芝士，橙圈=洞，
        青线=当前目标。用来翻图核对误检（黄圈画在椰子上就说明 detect_cheeses
        把装饰当芝士了），比看计数靠谱。"""
        img = bgr.copy()
        for c in cheeses:
            cv2.circle(img, (int(c[0]), int(c[1])), 12, (0, 255, 255), 2)
        for hp in holes:
            cv2.circle(img, (int(hp[0]), int(hp[1])), 14, (0, 128, 255), 2)
        if player:
            px, py, pfeet, bbox = player
            bx, by, bw, bh = bbox
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            cv2.circle(img, (int(px), int(py)), 3, (255, 0, 255), -1)
            cv2.line(img, (int(px), int(pfeet)), (int(px), int(pfeet) + 6), (255, 0, 255), 2)
        if tgt:
            cv2.arrowedLine(img, (int(px), int(py)), (int(tgt[0]), int(tgt[1])),
                            (255, 255, 0), 2, tipLength=0.05)
        cv2.putText(img, f"state={self.state} chz={len(cheeses)} hole={len(holes)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        save_png(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dbg_frame.png"), img)

    def _ring_meta(self, px, py, cb, falling, stuck, bbox):
        """给 _ring 的一帧附带信息，原样写进 corpus 的 labels.csv。

        keys 用 list 传（_ring 内部会 join 成 `a|b`）——**不要自己拼成逗号分隔**，
        那是 csv 的分隔符，会把这行拆错列。bbox 同理用分号拼。
        """
        return {"fn": self.fn, "state": self.state, "cb": cb, "px": px, "py": py,
                "bbox": "%d;%d;%d;%d" % tuple(bbox),
                "score": "%.2f" % getattr(self.vision, "_last_score", 0.0),
                "keys": sorted(self._keys), "fall": int(falling), "stuck": int(stuck),
                "bg": "%d/%d" % (self._bg_shift, self._bg_n)}

    def _ring_push_early(self, bgr):
        """在「会 continue 掉主循环末尾」的分支里补一次入队。

        为什么必须补：主循环末尾那次 `_ring.push` 前面有几条 continue（NO-LOCK 分支、
        掉落判定失效、脱困探针判定锁废了），而**这几种情况恰恰是最想留帧的**。
        2026-10-07 实测：一次 hole_deliver 事件的「前 15 帧」fn 从 39286 直接跳到 41293
        （差 2007 帧）—— 当时正处在 NO-LOCK 长段里（那一场只有 28.8% 的帧有锁），
        环形缓冲整个冻结在最后一次有锁的旧帧上，「事件前的画面」根本不是事件前的画面。

        这里用当前已知的状态补一帧（px/py/bbox 留空——没锁定本来就没有这些值）；
        不走 continue 的帧仍然走末尾那次，那里的 meta 是完整的。两条路互斥，不会重复入队。
        """
        if not self._ring.on or bgr is None:
            return
        self._ring.push(bgr, {
            "fn": self.fn, "state": self.state, "cb": "", "px": "", "py": "",
            "bbox": "", "score": "%.2f" % getattr(self.vision, "_last_score", 0.0),
            "keys": sorted(self._keys), "fall": "", "stuck": "",
            "bg": "%d/%d" % (self._bg_shift, self._bg_n)})

    def _tgt_hyst_on(self):
        """本帧用不用目标迟滞。见 TGT_HYST 的三态说明。"""
        if TGT_HYST == "auto":
            return int(time.time() / TGT_AB_PERIOD_S) % 2 == 0
        return str(TGT_HYST).strip().lower() not in ("0", "off", "false", "no", "")

    def _cb_low_ok(self):
        """「背芝士」塌下去后是否已经稳定够久（见 CB_EXIT_HOLD_S）。

        2026-10-26：原来 cb_count<=2 一帧就切回找芝士。实测全天 GOT_CHEESE/GOT_HOLE
        在 1 秒内来回翻的次数极多——背着芝士经过洞口上方时，芝士被地形挡掉一帧，
        窗口就塌下去，状态切走、判一次「掉芝士」，下一帧又背起来。加一个观察期，
        期间继续往洞口走，真掉了才切。
        """
        now = time.time()
        if self._cb_low_since == 0.0:
            self._cb_low_since = now
        return (now - self._cb_low_since) >= CB_EXIT_HOLD_S

    def _tgt_log(self, action, why, x, y):
        """tgt.pick_target 的回调：只记**改变决定**的那几次。

        绝不每帧打——12fps × 25min ≈ 1.8 万行会把日志淹掉。`why` 有五种来源：
        kind（芝士↔洞，状态机自己已经打了 GOT_CHEESE/GOT_HOLE）、nohyst（回退模式
        每帧都调，纯噪音）、switch（抢锁换了目标）、unreachable（守够帧数走不到）、
        lost/cooldown（跟丢重挑）、no_cands（候选全没了）。后四种才是要看的事件。
        """
        if why in ("nohyst", "kind"):
            return
        self._tgt_ev += 1
        if why == "switch":
            self._tgt_switches += 1
        print("[TGT] #%-4d %-20s -> (%d,%d)" % (self._tgt_ev, why, x, y))

    def _note_lock_ended(self):
        """刚丢掉一次锁定：按「这次锁定撑了多久」决定下次标定退避多久。

        撑够 `LOCK_MIN_USEFUL_S` 就算真用上了，退避清零；否则是「刚锁上就被丢」——
        2026-09-19 实测这种一串一串地来：_live_run6.log 里一轮不可玩的局，标定成功 5 次
        全在 5~8s 内被丢（看门狗 2 次、掉落判据 2 次），而每丢一次就在 8s 后按一次方向键
        探针，永远在「标定→丢锁→再标定」里打转，一次真正可玩的游玩都没进去。这时该做的是
        少按几次键、多等一会儿（回合间隙/已进洞/幽灵期间按键本来就没人响应）。
        倍退避封顶 `CALIB_MAX_RETRY_S`：撑够时长的锁定会立刻把退避清回 8s，所以正常游玩
        不受影响，最坏也只是换个回合格外等 60s。
        """
        lived = time.time() - self._lock_since
        if lived >= LOCK_MIN_USEFUL_S:
            self._drop_streak = 0
            return
        self._drop_streak += 1
        back = min(CALIB_RETRY_S * (2 ** (self._drop_streak - 1)), CALIB_MAX_RETRY_S)
        self._next_calib = time.time() + back
        print(f"[Bot] 第 {self._drop_streak} 次「刚锁上就被丢」（这次只撑了 {lived:.1f}s，"
              f"不足 {LOCK_MIN_USEFUL_S:.0f}s）→ 退避 {back:.0f}s 再标定"
              f"（大概这一局还不可玩）")

    def _anti_idle(self):
        """还没认到自己（盲锁期）时**绝不能让角色站着不动**。

        2026-10-27 06:1x 用户口径，两条一起才说得通：
          ① 聊天栏那句「你没有奶酪吃了！^_^」就是**小鼠死了**；
          ② **开局不动的小鼠会死**，而且这种「开局挂机」的死了以后，萨满的复活技能
             （灵性指引·救护车，能复活 1~5 只）**也救不回来**。

        原来的盲锁分支只有 `release_all()` 然后干等标定：一局才 ~2 分钟，它可能有大半
        时间在原地杵着不动 → 每回合开场就挂 → 整局报废，接着标定又失败（幽灵不带动镜头），
        于是下一局照旧。这就是「一局下来一块奶酪都没碰过」的总病灶，比目标选择靠前得多。

        所以这里就做一件最笨的事：**左右小幅度踱步 + 隔一阵跳一下**。
        幅度小是为了别一头走出平台；动起来是为了别被系统判成挂机。
        代价：没有锁就没有地面信息，理论上仍可能走出边缘——但「站桩必死」是已知的，
        「走动可能摔」只是可能，这笔账划得来。
        """
        now = time.time()
        if now >= self._idle_turn_next:
            self._idle_dir = -self._idle_dir
            self._idle_turn_next = now + IDLE_WALK_S
        if now >= self._idle_jump_next:
            self._idle_jump_next = now + IDLE_JUMP_EVERY
            self._idle_jump_until = now + IDLE_JUMP_S
        self._key("left", self._idle_dir < 0)
        self._key("right", self._idle_dir > 0)
        self._key("up", now < self._idle_jump_until)

    def _dl_predict(self, bgr):
        """深度学习自定位：返回 (cx, cy, conf) 或 None。**不按键**。"""
        if self._dl is False:          # 试过了、没有可用模型 —— 别再往下走，否则 False.predict 刷屏
            return None
        if self._dl is None:
            if self._dl_tried:
                return None
            self._dl_tried = True
            try:
                d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dl")
                if d not in sys.path:
                    sys.path.insert(0, d)
                import dl_loc
                self._dl = dl_loc if dl_loc.available() else False
            except Exception as e:
                print(f"[DL] 自定位模块不可用：{e}", flush=True)
                self._dl = False
            if self._dl is False:
                return None
        try:
            r = self._dl.predict(bgr, self.vision.stage_rect(bgr), DL_MIN_CONF)
        except Exception as e:
            print(f"[DL] 推理异常：{e}（之后不再用模型）", flush=True)
            self._dl = False
            return None
        if r is None:
            return None
        cx, cy, conf, _ = r
        h, w = bgr.shape[:2]
        if conf < DL_MIN_CONF or not (0 <= cx < w and 0 <= cy < h):
            return None
        return cx, cy, conf

    def _maybe_calibrate(self, bgr, gray):
        """没锁定时周期性尝试标定；返回是否已锁定。"""
        if self.vision.has_lock():
            return True
        now = time.time()
        # ① 先试**不按键**的重挂：名字字形还留在画面上就几毫秒挂回去。
        #    这条路不碰方向键，所以不受「退避」约束——退避本来只是为了别老按键打扰
        #    游戏；而名字字形是纯视觉，代价只有几毫秒。2026-10-26 20:49 那场就是死在
        #    「标定成功 4.3s 后跟丢 → 退避 60s」上：一局才 ~2 分钟，退避一次整局报废。
        if now >= self._next_label_try:
            self._next_label_try = now + LABEL_TRY_S
            got = self.vision.reacquire_by_label(bgr)
            if got is not None:
                cx, cy, size, tag = got
                if self.vision.set_player_lock(gray, cx, cy, size, bgr=bgr, tag=tag):
                    self.playable = True
                    self._next_calib = 0.0
                    self._drop_streak = 0
                    print(f"[Bot] 名字标签重挂成功 @({cx},{cy})（没按键）"
                          f"标签 {tag[2]}x{tag[3]} @({tag[0]},{tag[1]})")
                    self._lock_src = "label"
                    return True
        # ② 深度学习自定位（同样**不按键**）。放在老标定前面：老标定按完 →← 再反向投票
        #    要 4~5s，这 4~5s 角色是站着的（开局站着的会死）。模型十几毫秒就能给出位置，
        #    先锁上让 _anti_idle 之后的正常寻路接管。判错也有兜底：锁住的模板一旦跟不上
        #    实地移动，主循环的丢锁/脱困逻辑照样会把它清掉。
        if DL_LOC and now >= self._next_dl_try:
            self._next_dl_try = now + DL_TRY_S
            got = self._dl_predict(bgr)
            if got is not None:
                cx, cy, conf = got
                size = int(getattr(self.vision, "_label_body_size", 0) or 0) or 48
                size = int(min(SPRITE_MAX_PX, max(PATCH_SIZE, size)))
                if self.vision.set_player_lock(gray, cx, cy, size, bgr=bgr, tag=None):
                    self.playable = True
                    self._next_calib = 0.0
                    self._drop_streak = 0
                    self._lock_src = "dl"
                    print(f"[DL] 自定位锁定 @({cx:.0f},{cy:.0f}) 置信 {conf:.2f} "
                          f"边长 {size}（没按键）", flush=True)
                    return True
        # 2026-10-27 00:2x：**模板自愈。** 今晚 23:58~00:04 出现过一次「连续 5000+ 帧
        # 没有锁」：同一个存档字形，我在新进程里离线跑 `reacquire_by_label` 一次就挂上
        # （峰 0.948），可活着的那个 bot 进程怎么都挂不上——说明**进程内存里那份字形
        # 已经坏了**（运行时被某次重挂/标定顶成了别人的名字），而磁盘上那份还是好的。
        # 重挂连败这么久就别再信内存，隔 20s 从磁盘重新读一次（读的是上次验证过的）。
        if self._no_player_frames > 600 and now >= self._tpl_reload_next:
            self._tpl_reload_next = now + 20.0
            self.vision._load_label_tpl()
            print(f"[Bot] 已连续 {self._no_player_frames} 帧没锁上 → "
                  f"从磁盘重新读一次名字字形（模板自愈）")
        if now < self._next_calib:
            return False
        ok = self._identify_player()
        if ok:
            self.playable = True
            self._next_calib = 0.0
        else:
            if self.playable is not False:
                # 别在这硬说成「按键没让画面动」：反向投票不过、锁到移动平台、真的静止
                # 都会走到这里，具体原因 _identify_player 已经打了 [CALIB] 行。
                print(f"[Bot] 标定未通过（具体原因见上面的 [CALIB] 行）——"
                      f"每 {CALIB_RETRY_S:.0f}s 重试一次")
            self.playable = False
            self._next_calib = time.time() + CALIB_RETRY_S
        return ok

    def _scale_of(self, bgr):
        h, w = bgr.shape[:2]
        return max(1, min(w, h) / float(SCALE_BASE))

    def start(self):
        if not self.cap.start():
            print("[Bot] Cannot find Transformice window")
            return False

        print("[Bot] 把游戏窗口切到前台…")
        if not self.cap.focus():
            print("[Bot] 警告：焦点没拿到（可手动点一下游戏窗口）——按键会进不去游戏")
        time.sleep(0.6)

        r = self.cap.snap()
        if r is None:
            print("[Bot] Cannot capture window")
            return False
        bgr, gray = r
        self.vision.store_frame(gray)

        # 内存增强要在标定之前接上：标定那一下按 → 的位移正好用来绑定内存里的玩家
        # 结构体（_identify_player 里会顺手 bind）。attach 失败就纯视觉跑，不影响。
        self.use_mem = USE_MEM_TARGETS and self.mem.attach()
        if self.use_mem:
            print("[Bot] Memory reader attached")

        # 定位本地玩家：只用输入相关性（颜色法在自定义皮肤下失效）。
        # 失败不代表出错——大概率只是现在不在可玩状态，交给主循环周期重试。
        # 先试**不按键**的两条路（名字字形重挂 / 深度学习自定位），都不行才按方向键标定。
        # 这三条路都在 _maybe_calibrate 里按代价从低到高排好了。
        print("[Bot] 定位本地玩家（先试不按键的：名字标签 / 深度学习）…")
        self.release_all()
        locked = False
        for _ in range(CALIB_TRIES):
            r2 = self.cap.snap()
            if r2 is not None:
                bgr, gray = r2
                if self._maybe_calibrate(bgr, gray):
                    locked = True
                    break
            time.sleep(0.5)
        self.playable = locked
        if locked:
            self._next_calib = 0.0
        else:
            save_png(os.path.join(os.path.dirname(__file__), "debug_snap.png"), bgr)
            print("[Bot] 标定失败：按键没有让画面动——大概率不在可玩状态"
                  "（回合间隙/已进洞/幽灵），进入主循环后每 %.0fs 重试" % CALIB_RETRY_S)
            self._next_calib = time.time() + CALIB_RETRY_S

        return True

    def run(self):
        print("[Bot] Running (pure vision mode)")

        # "iter" 是**整轮**的墙钟时间（含下面那些没单独计时的部分），"player" 是
        # detect_player（原先完全没被计时）。为什么要这两个：锁定中 [PERF] 的 total
        # 只有 17ms 却只跑出 11fps，说明 ~70ms/帧花在了计时桶外面——不把整轮量出来
        # 就只能猜（2026-09-19 就猜错过一次，把 world_shift 当成了主因）。
        pf = {"snap": 0.0, "player": 0.0, "detect": 0.0, "mem": 0.0, "calib": 0.0,
              "iter": 0.0}
        pf_n = 0
        pf_fn0 = self.fn
        pf_t0 = time.time()
        pf_iter_t0 = 0.0

        while True:
            # 上一轮开头到这一轮开头 = 正好一整轮。放在最前面是为了让所有
            # `continue` 分支（标定/接管/掉落放弃…）都照样被计入。
            if pf_iter_t0:
                pf["iter"] += time.perf_counter() - pf_iter_t0
            pf_iter_t0 = time.perf_counter()

            if _quit:
                self.release_all()
                break

            # 每轮都记一次，包含走 continue 的标定/接管分支（否则均值会被稀释）
            pf_n += 1
            if time.time() - pf_t0 >= 2.0:
                wall = time.time() - pf_t0
                fps = (self.fn - pf_fn0) / wall
                ms = {k: v / pf_n * 1000 for k, v in pf.items()}
                acct = ms["snap"] + ms["player"] + ms["detect"] + ms["mem"] + ms["calib"]
                print(f"[PERF] {fps:5.1f} fps | snap {ms['snap']:5.1f} detect {ms['detect']:5.1f} "
                      f"mem {ms['mem']:5.1f} calib {ms['calib']:5.1f} "
                      f"| total {acct:5.1f} ms")
                print(f"[PERF2] 整轮 {ms['iter']:5.1f} ms = 计时 {acct:5.1f} "
                      f"+ player {ms['player']:5.1f} + 未计时 {ms['iter'] - acct - ms['player']:5.1f}")
                print(f"[STAT] {self._dbg}")
                pf = {k: 0.0 for k in pf}
                pf_n = 0
                pf_fn0 = self.fn
                pf_t0 = time.time()

            _t = time.perf_counter()
            r = self.cap.snap()
            pf["snap"] += time.perf_counter() - _t
            if r is None:
                if not self._cap_dead:      # 只在刚坏掉时打印一次，别每帧刷屏
                    self._cap_dead = True
                    print(f"[Bot] 截图失败：{self.cap.why()}——恢复后会自动继续")
                self._cap_fail += 1
                # 大约每 2 秒重扫一次窗口：游戏重启换了句柄时能自动接上
                if self._cap_fail % 40 == 0 and self.cap.rebind():
                    print("[Bot] 重新找到游戏窗口（句柄变了），继续")
                time.sleep(0.05)
                continue
            if self._cap_dead:
                self._cap_dead = False
                self._cap_fail = 0
                print("[Bot] 截图恢复")
            bgr, gray = r
            self.fn += 1
            h, w = gray.shape
            scale = max(1, min(w, h) / float(SCALE_BASE))

            # ====== Manual takeover ======
            if time.time() < _manual_until:
                self.release_all()
                # 刷新 _dbg，否则 [STAT] 会一直重复接管前那一行，看起来像「计数器卡死」
                # （实测就是靠这个误判过一次）。接管期间 fn 照涨，但状态一行没更新。
                self._dbg = f"MANUAL 接管中（剩 {_manual_until - time.time():.0f}s）"
                time.sleep(0.01)     # 别空转：这个分支一帧活都不干，不睡会跑满一个核
                continue

            # ====== Detection ======
            _t = time.perf_counter()
            player = self.vision.detect_player(bgr, gray)
            pf["player"] += time.perf_counter() - _t
            if not self.vision.has_lock():
                # 没有可信的锁定：不要拿颜色/运动兜底的结果去操纵角色
                # （实测颜色法会锁到 UI 图标/草地），改成尝试输入标定。
                # 只在**刚进入**盲锁的那一帧全松开（清掉上一把锁残留的按键）；
                # 之后方向键交给 _anti_idle 一直按着，别每帧抬一次（见 _anti_idle_on 说明）。
                if not self._anti_idle_on:
                    self.release_all()
                    self._anti_idle_on = True
                self.vision.store_frame(gray)
                self._anchor_gray = None       # 标定期间画面会被按键探针搅动，重锁后重新锚定
                self._no_player_frames += 1
                if self._had_lock:
                    self._had_lock = False
                    self._note_lock_ended()    # 刚丢锁，按「撑了多久」决定要不要退避
                _t = time.perf_counter()
                self._maybe_calibrate(bgr, gray)
                pf["calib"] += time.perf_counter() - _t
                # 标定完了还是没锁上 → **别站桩**（见 _anti_idle 的说明）。
                # 放在标定之后：标定自己要按键做探针，先让它把活儿干完。
                if not self.vision.has_lock():
                    self._anti_idle()
                self._dbg = (f"NO-LOCK 可操作={self.playable} 连续={self._no_player_frames}")
                self._ring_push_early(bgr)      # 这一支 continue 掉末尾那次 push（见该函数）
                if self._no_player_frames == 240:
                    print(f"[Bot] 已连续 {self._no_player_frames} 帧没有玩家锁定，"
                          f"可操作={self.playable}")
                time.sleep(0.01)     # 同上：没锁定时这一支几乎不干活
                continue
            self._no_player_frames = 0
            self.playable = True
            if self._anti_idle_on:
                # 重新认到人了：把踱步的按键交还给正常控制逻辑（它会逐帧自己算该按什么）
                self._anti_idle_on = False
                self.release_all()
            if not self._had_lock:
                self._had_lock = True
                self._lock_since = time.time()
                # 重新锁定 = 相机可能已经滚过、玩家也换了个位置，旧目标坐标作废。
                # 不清的话要等 TGT_MISS_CLEAR(10) 帧才自然重挑，白白多站 ~0.8s。
                self._tgt_state = tgt_mod.new_state()
                self._tgt_far, self._tgt_far_xy = False, None   # 见 __init__ 那处说明
                # 2026-10-06：cb 历史同样作废。跟丢锁期间这一支根本不执行，
                # `_cb_history` 既不 append 也不清，于是**上一段锁的 cb 原样存活**，
                # 重锁后立刻参与 CB_ENTER_HOLE 判定 —— 这就是记忆里反复出现的
                # 「刚标定成功 2 秒就报送洞」假阳性。
                # 实证：corpus/1005_231519 的 cb_up 事件，fn 从 20982 跳到 21965
                # （隔 983 帧 / 29 秒没锁），这 29 秒里 cb 一直冻在 4 不动，
                # 重锁后才开始 3→2→1 衰减 —— 说明那 4 个 True 一路带过了空档。
                self._cb_history.clear()
                # ⚠ 2026-10-08 半夜补：**光清窗口还不够，状态机也得一起作废。**
                # 清了窗口 → cb_count 当帧就变 0，而下面 1773 行那句是
                #     elif self.state == 1 and cb_count <= CB_EXIT_HOLE(2): → state=0，
                #     顺手判「人贴不贴着洞口」，贴着就 +1「送进洞」。
                # 于是**「清窗口」与「芝士真的离开背了」在这一帧上完全同形** ——
                # 上一段锁停在 state=1 的话，重锁那一帧无条件命中；只要人恰好站在
                # 某个洞口 34px 内，就白记一次送洞。
                # 实证：_bot_run_1007_2356_stallscan.log
                #   [23:56:45] [LOCK] 建库 → [23:56:46] 标定成功
                #   [23:56:46] [HOLE] 送进洞 #1 洞口(177,406) 人(156,382)
                # 人(156,382) 到洞 (177,406) 距 31.9px < 34*1.1667=39.7px —— 位置判据
                # 本身没错，错的是「这一帧的 state 1→0」根本不是一次真的离开背。
                # 它上一行 STAT 还是 NO-LOCK 连续=3609：state=1 是 ~6 分钟前那段锁留下的，
                # 跨了一整个没锁的空档。**状态机跨锁一律不作数。**
                # 作废不会漏记真送洞：没锁时 bot 不操纵角色、人也不会自己走进洞，真送洞
                # 必然发生在有锁那一段里，走的是 1773 行那条正常路径（cb 按 CB_WINDOW 衰减）。
                if self.state == 1:
                    print("[Bot] 重锁：上一段锁的「背芝士」状态作废（不判送洞/掉芝士）")
                self.state = 0
                self._cb_low_since = 0.0    # 观察计时同理作废（见 _cb_low_ok）
                self._hole_lvl_since = 0.0  # 「洞在另一层」的判定同理作废
                self._hole_lvl_tgt = None
                # 同一个道理：贴没贴到目标跟前，也是「上一段锁」的事实，重锁后作废。
                # 不清的话，上一段锁结束在「贴着」的状态、这段锁一开始就贴着，
                # near_target 那个边沿事件就永远不会触发，白丢一批样本。
                self._near_prev = False

            px, py, pfeet, bbox = player

            _t = time.perf_counter()
            cheeses = self.vision.detect_cheeses(bgr)
            # 2026-09-30 修：把落在**自己身上**的「芝士」剔掉。
            # detect_cheeses 是按颜色找的，玩家自己的黄色部件全中招——萨满的黄色帽子、
            # 射出去的黄色箭/星星特效，都被当成芝士，而且离自己只有 0~20px。
            # 目标按「最近优先」挑，于是**每次挑中的都是自己身上的假芝士**，bot 站在
            # 自己脑袋上「捡芝士」，cb 永远是 0、一次也捡不起来。
            # 证据：_dbg_frame.png 里两个黄圈正画在玩家自己身上（黄帽 + 箭上黄星），
            # 而画面右侧那块**真芝士**整场都没被选中过；该场 145 次采样里 21 次
            # 离目标 <=30px（最近只有 5px）却 cb 全是 0。同一张图证明检测器本身没错——
            # 真芝士那块圈是画对了的，错的只是没排除自己。
            bx, by, bw, bh = bbox
            _pad = int(CHEESE_SELF_PAD * scale)
            _self_r = int(CHEESE_SELF_R * scale)
            # 2026-10-06 第二次修正：豁免只给**从远处锁上的**那个目标。
            # 第一版豁免（无条件豁免当前锁）把 09-30 修好的老毛病又放了回来，
            # 见 CHEESE_SELF_PAD 上面那段里 180309 帧的实测。
            # 判据用下面 `self._tgt_far` 那个锁存位：
            #   从远处锁上 → 是我看见、并主动走过去的一块，靠近了当然要继续豁免，
            #                否则走到 40px 内就被删掉、机器人当场停步（这才是旧毛病）；
            #   一锁上就贴在身上 → 只可能是自己的假芝士（真芝士不会凭空生在我身上），
            #                不豁免，让它按 tgt.py 的 miss 计数自然衰减掉。
            _st = self._tgt_state
            _has_lock = _st["kind"] == 0 and _st["age"] > 0
            if not _has_lock:
                self._tgt_far, self._tgt_far_xy = False, None
                _lx = _ly = -9999
            else:
                _lx, _ly = _st["x"], _st["y"]
                # 换了一块就重新判（位置跳过一个身体宽 = 不是同一块了）
                if (self._tgt_far_xy is None
                        or abs(_lx - self._tgt_far_xy[0]) + abs(_ly - self._tgt_far_xy[1]) > 60):
                    self._tgt_far = False
                if abs(_lx - px) + abs(_ly - py) > _self_r:
                    self._tgt_far = True         # 这一锁期间，确实在自身半径之外见过它
                self._tgt_far_xy = (_lx, _ly)
            _far_lock = bool(self._tgt_far)

            def _keep(c, _lx=_lx, _ly=_ly, _far_lock=_far_lock):
                # 豁免**不能**再加「不许落在身体框里」那一条：身体框就有 58px 宽，
                # 机器人走到真芝士跟前时（实测 24~30px）对方**必然**落在框内 ——
                # 加了这一条等于「快到跟前就把目标删掉」，正是 10-06 那版要修的老毛病
                # （_test_selfdrop.py 的 [2] 就是照这个写的，别删）。
                # 会不会因此把自身假芝士放回来？不会：锁存位只在「这一锁曾出现在
                # 自身半径之外」时才置起，而锁上自身块那一刻它就在 17px 处，置不起来。
                if (_far_lock and (abs(c[0] - _lx) + abs(c[1] - _ly)) <= _self_r):
                    return True                      # 就是我现在锁着、正要走过去的那块
                if abs(c[0] - px) + abs(c[1] - py) <= _self_r:
                    return False                     # 就在我身上/脚边，是我自己的部件
                return not (bx - _pad <= c[0] <= bx + bw + _pad
                            and by - _pad <= c[1] <= by + bh + _pad)

            keep = [c for c in cheeses if _keep(c)]
            if len(keep) != len(cheeses):
                self._selfdrop += len(cheeses) - len(keep)
            cheeses = keep
            # 2026-10-27 00:2x：**别人背上的芝士不算芝士。**
            # detect_cheeses 只按颜色找，别的老鼠背着的绿芝士一模一样会中招——
            # `_z_rt5x.png` 里那个红圈圈中的就是 Drkliy **背上**那块：机器人一路走过去，
            # 人家一跑它就跟丢，于是 STAT 里 cheese 在 0/1 之间跳、[TGT] 大段
            # `no_cands`、人在原地反复 [STUCK]，整场一块芝士都拿不到。
            # 判据用「动没动」：相机是静止的（phaseCorrelate 实测 dx≈±0.1px），所以
            # **地上的芝士一动不动，背上的芝士跟着人走**。只有连续 CHEESE_STILL_FRAMES
            # 帧都待在同一小块地方的，才进候选名单。
            _now_pos = []
            for c in cheeses:
                _now_pos.append((float(c[0]), float(c[1]), c))
            _prev = self._cheese_hist
            _newhist = []
            for _x, _y, _c in _now_pos:
                _hit = 1
                for _px2, _py2, _h, _c2 in _prev:
                    if abs(_x - _px2) <= CHEESE_STILL_R * scale and abs(_y - _py2) <= CHEESE_STILL_R * scale:
                        _hit = _h + 1
                        break
                _newhist.append((_x, _y, _hit, _c))
            self._cheese_hist = _newhist
            _still = [t[3] for t in _newhist if t[2] >= CHEESE_STILL_FRAMES]
            self._cheese_still_n = len(_still)
            if _still != cheeses:
                cheeses = _still
            holes = self.vision.detect_holes(bgr, gray)
            # 洞口静止闸（见 HOLE_STILL_WIN 那段说明）：候选要**最近 8 帧里至少 5 帧
            # 都待在同一小块地方**才算洞。跑动的老鼠、飘动的蝙蝠一帧就出局。
            self._hole_hist.append(list(holes))
            if len(self._hole_hist) > HOLE_STILL_WIN:
                self._hole_hist = self._hole_hist[-HOLE_STILL_WIN:]
            if self._hole_hist and holes:
                _hr = HOLE_STILL_R * scale
                _st = []
                for _h in holes:
                    _cnt = 0
                    for _fr in self._hole_hist:
                        for _q in _fr:
                            if abs(_h[0] - _q[0]) <= _hr and abs(_h[1] - _q[1]) <= _hr:
                                _cnt += 1
                                break
                    if _cnt >= HOLE_STILL_MIN:
                        _st.append(_h)
                self._hole_still_n = len(_st)
                holes = _st
            else:
                self._hole_still_n = 0
            ground_info = self.vision.find_ground_below(bgr, gray, px, pfeet)
            has_cb = self.vision.has_cheese_on_back(bgr, bbox)
            pf["detect"] += time.perf_counter() - _t

            # ====== State: cheese-on-back detection ======
            # 这条路径只在**已经跟住玩家**时走到，所以这里故意**不做全堆扫描**：
            # 一次扫描阻塞 2.5~3.4s，期间跟踪完全停摆，而方向键还按着、老鼠一直在跑，
            # 扫描一结束早就出了 track_locked 的 ±64px 搜索窗，必然 [LOCK] 跟丢。
            # 实测时间线：每次 LOCKED 之后 ~2s 必 LOST，且 LOST 那一刻正好有一条
            # [MEM] 扫描完成——锁就活不过一次扫描。全堆扫描改到 _identify_player 里
            # 做（那时本来就没锁，扫了不亏）。这里只重读绑定好的那一个地址（<1ms）。
            if self.use_mem:
                _t = time.perf_counter()
                if self._mem_player_addr:
                    got = self.mem.read_many([self._mem_player_addr]).get(self._mem_player_addr)
                    if got:
                        self._mem_player_wx, self._mem_player_wy = got[0], got[1]
                pf["mem"] += time.perf_counter() - _t
            # Vision-based cheese-on-back (primary)
            self._cb_history.append(has_cb)
            if len(self._cb_history) > CB_WINDOW:
                self._cb_history.pop(0)
            cb_count = sum(self._cb_history)
            # 2026-10-26：cb 没塌下去就作废观察计时（见 CB_EXIT_HOLD_S）。
            if cb_count > CB_EXIT_HOLE:
                self._cb_low_since = 0.0

            if self.state == 0 and cb_count >= CB_ENTER_HOLE:
                self.state = 1
                print("[Bot] GOT_CHEESE -> HOLE")
                self.stuck_frames = 0
            elif self.state == 1 and cb_count <= CB_EXIT_HOLE and self._cb_low_ok():
                self.state = 0
                # 2026-09-30 加：原来这里只打 GOT_HOLE -> CHEESE，**看不出这块芝士是
                # 送进洞了还是掉了**——进洞是机器人唯一真正的得分动作，而日志里一直
                # 没有这个事实，所以 2026-09-29 跑完只能报「背起过几次」，
                # 回答不了「到底送进去几个」。
                # 判据复用这个分支本身：它已经是「滑动窗口里的芝士帧塌到 <=2」=
                # 芝士确实离开背了，不是单帧检测抖动。此时看人贴不贴着某个洞口：
                # 洞的检测只给中心点（vision.detect_holes 返回 (cx,cy)，没有半径），
                # 所以用 HOLE_DELIVER_PX 兜一下——洞口直径约 26~40px、老鼠身宽约 20px，
                # 中心距要 <约 34px 才重叠得上，再远就只可能是掉了/被抢/回合结束。
                near = None
                if holes:
                    near = min(holes, key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
                    if (near[0] - px) ** 2 + (near[1] - py) ** 2 \
                            > (HOLE_DELIVER_PX * scale) ** 2:
                        near = None
                if near is not None:
                    self._delivered += 1
                    print(f"[HOLE] 送进洞 #{self._delivered} ✓ 洞口({near[0]},{near[1]}) "
                          f"人({px},{py})")
                else:
                    self._dropped += 1
                    print(f"[HOLE] 芝士离开背但不在洞口附近（人({px},{py})）"
                          f"→ 掉了/被抢/回合结束，不计入送洞")
                # 顺手落一次环形缓冲事件，把「判成送洞还是掉」这个结论直接写进文件名和
                # labels.csv 备注 —— 2026-10-07 那场 14 次「离开背」全判 0 送洞，事后想复盘
                # 却一帧都没有（DBG_RING 默认关）。跑采集时置 DBG_RING=1 就有前后各 15 帧。
                # （放在 state_x_toy 那个事件之前，所以那一帧会标成更有信息量的 hole_*。）
                self._ring.event(
                    "hole_" + ("deliver" if near is not None else "drop"),
                    note="cb=%d px=%d py=%d holes=%d near=%s"
                         % (cb_count, px, py, len(holes),
                            ("%d,%d" % near) if near else "-"))
                print("[Bot] GOT_HOLE -> CHEESE")
                self.stuck_frames = 0

            # ====== Anti-fall ======
            if ground_info is None:
                self.no_ground_frames += 1
            else:
                self.no_ground_frames = 0
            falling = self.no_ground_frames > NO_GROUND_FALL_FRAMES
            if falling:
                if self._fall_since == 0.0:
                    self._fall_since = time.time()
                    # 记下「开始判掉落」这一刻的屏幕 y：真掉落 y 会一路变大，
                    # 拿它当铁证（见下面 FALL_GIVEUP_S 那段）
                    self._fall_py0 = py
            else:
                self._fall_since = 0.0

            # ====== Anti-stuck ======
            # 判据不能用「相邻两帧位移 < 2px」：模板跟踪取的是整数 argmax，老鼠原地
            # 踏步时精灵动画会让它每帧抖 ±2~4px，每帧都把计数器清零。实测 bot 顶着墙
            # 按了十几秒方向键，STAT 里 pos 一直是 (444,294)、keys=['right']、stuck 却
            # 始终是 0，永远触发不了脱困。改成锚点式：只有相对锚点的累计漂移超过
            # STUCK_MOVE_TOL 才算「真的在走」并重设锚点，小幅抖动累积不到阈值。
            #
            # 但**只看屏幕坐标还不够**：长图里相机是跟着角色滚的，角色在屏幕上本来就
            # 钉住不动——实测有一次锁定 29 分钟里 pos 只从 (826,149) 挪到 (702,149)，
            # 期间报了 1108 次 [STUCK]，而它其实一直在跑。所以同时看第二路证据：
            # **背景滚没滚**。相机跟随时背景跟着滚，角色真在走世界就一直在动；
            # 顶死在墙角时两路都不动，才是真卡住。
            #
            # 背景滚没滚这条证据**只在屏幕坐标判据不成立时才去算**：`moved` 是「或」，
            # 位置一动结论就已经定了，再跑一遍全帧相位相关纯属白花钱。实测（_probe_perf.py
            # 2026-09-19）world_shift 中位 21.3ms，而锁定中 [PERF] 只记到 15.9ms 却只跑出
            # 12.7fps——差的这 ~63ms/帧大半是它和 detect_player 没被计时。
            # 语义不变：位置在动 → moved 必为 True，bg_shift 取不取都一样。
            pos_moved = (abs(px - self.stuck_x) >= STUCK_MOVE_TOL
                         or abs(py - self.stuck_y) >= STUCK_MOVE_TOL)
            if pos_moved:
                self._bg_shift, self._bg_n = 0, 0
                self._bg_hits = 0
            else:
                # 走到这儿才会真的比一次。锚点在上一次 moved 时已经被清成 None，
                # 所以这一帧是「重设锚点、返回 0」，下一帧起才拿到真值——真在滚的时候
                # 只是晚一帧发现，stuck 要连累 STUCK_FRAMES 帧才会触发，够用。
                self._track_bg_shift(gray)
                # **背景证据要连着两帧成立才算数**（2026-10-27 02:22 现场）：
                # 万圣节图满屏落叶/蝙蝠/发光南瓜，全帧相位相关偶尔单帧就报十几像素的
                # 假位移。单帧假位移就够把这个计数器清零，实测「顶着墙一动不动 90 秒、
                # 一次 [STUCK] 都不报」，stuck_frames 永远在 41~59 之间来回——于是探索
                # 分支只会在原地一下一下地跳，永远等不到「换向」那一步。
                if self._bg_shift >= STUCK_MOVE_TOL:
                    self._bg_hits += 1
                else:
                    self._bg_hits = 0
            moved = pos_moved or self._bg_hits >= 2
            if moved:
                self.stuck_frames = 0
                self._stuck_streak = 0         # 真的挪了窝，脱困计数清零
                self.stuck_x, self.stuck_y = px, py
                self._anchor_gray = None       # 锚点跟着走，下一帧重设
            else:
                self.stuck_frames += 1
            stuck = self.stuck_frames > STUCK_FRAMES

            # ====== Target selection ======
            # USE_MEM_TARGETS=False 时 use_mem 在 start() 就是 False，整段等于不存在。
            # 关掉的原因见那个常量的注释：内存折算出的目标会跑到画面外把 bot 带出地图。
            tgt = None
            src = "vision"
            if self.use_mem and self._mem_player_addr:
                addrs = self._mem_cheese if self.state == 0 else self._mem_hole
                pts = self._mem_positions(addrs) if addrs else []
                if pts:
                    wx, wy = min(pts, key=lambda t: (t[0] - self._mem_player_wx) ** 2
                                 + (t[1] - self._mem_player_wy) ** 2)
                    dx_w = self._mem_key_sign * (wx - self._mem_player_wx)
                    dy_w = (self._mem_dy_sign or -1) * (wy - self._mem_player_wy)
                    # 兜底：地址缓存过期时 world 坐标会串成离谱的值，折算出的屏幕目标
                    # 会飞很远。宁可退回视觉，也别照着垃圾坐标一路狂奔。
                    if abs(dx_w) <= MEM_MAX_OFFSET and abs(dy_w) <= MEM_MAX_OFFSET:
                        tgt = (px + dx_w, py + dy_w)
                        src = "mem"
            if tgt is None:
                cands = cheeses if self.state == 0 else holes
                # 够不着被拉黑的目标先剔掉（见 TGT_BAD_HITS 那段说明）。
                # 剔空了就返回空表，下面 tgt=None → 走无目标探索，比原地跳强。
                if self._bad_tgts:
                    _nb = time.time()
                    self._bad_tgts = [b for b in self._bad_tgts if b[2] > _nb]
                    if self._bad_tgts:
                        cands = [c for c in cands
                                 if not any(abs(c[0] - b[0]) <= TGT_BAD_R
                                            and abs(c[1] - b[1]) <= TGT_BAD_R
                                            for b in self._bad_tgts)]
                # A/B：TGT_HYST=auto 时每 TGT_AB_PERIOD_S 秒翻一次，同场跑完两种配置。
                # 翻转时把迟滞状态清掉——否则「关→开」那一帧会抱着上一段留下的旧锁定
                # 继续用，两组样本就串味了。两边都从干净态起跑才可比。
                self._tgt_on = self._tgt_hyst_on()
                if self._tgt_on != self._tgt_on_prev:
                    self._tgt_on_prev = self._tgt_on
                    self._tgt_state = tgt_mod.new_state()
                    self._tgt_far, self._tgt_far_xy = False, None
                    print("[TGT] A/B 切换 -> 迟滞 = %s" % ("on" if self._tgt_on else "off"))
                # 「同平台优先」的两层过滤搬进了 tgt.py（_same_platform），这里不再重复。
                # tgt 为 None 只在候选为空时出现，下面的运动分支本来就处理了这种情况。
                tgt = tgt_mod.pick_target(
                    cands, self.state, ground_info, px, py, scale,
                    self._tgt_state, time.time(),
                    C=self._tgt_C, on=self._tgt_on, log=self._tgt_log)

            # ====== Movement ======
            dx_dead = int(DX_DEAD * scale)
            dy_jump = int(DY_JUMP * scale)

            if falling:
                if time.time() - self._fall_since > FALL_GIVEUP_S \
                        and py - self._fall_py0 < FALL_STATIC_DY:
                    # 2026-10-27 01:1x：**「脚下没地面」不等于「在掉」。**
                    # 活动图（万圣节）的地台是深蓝石板、木桥、南瓜，`build_terrain_mask`
                    # 只认草地/土色，`find_ground_below` 在这类图上每帧都返回 None，
                    # 站在平地上也被判成掉落；满 3s 就走下面那条「丢锁重标定」，于是
                    # 每 3~6s 白丢一次好锁（01:07:01 实测：刚标定成功 4.6s 就丢在这）。
                    # 真掉落的铁证是**屏幕 y 一路变大**；y 纹丝不动而地面检测说没地面，
                    # 只能是检测失灵 —— 这种情况不该丢锁，清掉计数继续干活就行。
                    print(f"[Bot] 判定掉落已持续 {time.time() - self._fall_since:.1f}s，"
                          f"但 y 只动了 {py - self._fall_py0}px → 判定为地面检测失灵"
                          f"（活动图地台不认），不丢锁，继续跑")
                    self.no_ground_frames = 0
                    self._fall_since = 0.0
                    self._fall_py0 = py
                elif time.time() - self._fall_since > FALL_GIVEUP_S:
                    # 判「脚下没地面」判了这么久还没落回地面，说明这个判据本身已经不成立：
                    # 要么角色早就摔出地图/变成幽灵，要么锁定框飘到了别处，find_ground_below
                    # 在错坐标上永远找不到地形。实测这时候 bot 每帧只发跳跃、keys=[]，
                    # 能这样一直站到回合结束，中间没有任何机制会把它拉回来（_live_run.log
                    # 里 `fall=1 keys=[]` 一路持续到日志结束就是这么来的）。
                    # 唯一能自愈的动作是**承认定位已经废了**：丢掉锁定退回 NO-LOCK，
                    # 让主循环重新标定。
                    print(f"[Bot] 判定掉落已持续 {time.time() - self._fall_since:.1f}s 仍没"
                          f"落到地面 → 认定定位失效，丢弃锁定重新标定")
                    self.vision.clear_player_lock()
                    self.no_ground_frames = 0
                    self._fall_since = 0.0
                    self._anchor_gray = None
                    self._ring_push_early(bgr)   # 这一支 continue 掉末尾那次 push
                    continue
                # 判定掉落：**千万别松开方向键**。原来这里是 release_all()，等于放弃空中
                # 水平控制，角色会笔直往下掉进虚空摔死（摔死 = 整局报废，见 EDGE_GUARD_PX
                # 那段说明）。保留朝目标的水平转向，才有机会落到下面那层平台上；
                # 空中再按 up 是没用的（没有二段跳），但贴地那一下能顶一下，留着。
                if tgt is not None:
                    dx = tgt[0] - px
                    if dx > dx_dead:
                        self._key("left", False)
                        self._key("right", True)
                    elif dx < -dx_dead:
                        self._key("right", False)
                        self._key("left", True)
                    else:
                        self._key("left", False)
                        self._key("right", False)
                self._ktap("up", JUMP_TAP_MS)

            elif stuck:
                # 有目标却半天没挪窝 = 撞墙/卡住。这一支必须排在追目标前面，否则
                # 只要还看得见目标就永远不会触发，bot 会一直顶着墙按方向键。
                self.stuck_frames = 0
                self._stuck_count += 1
                # STAT 每 2s 才采一次样，而这一支每 STUCK_FRAMES+1 帧才可能进一次，
                # 光看 STAT 里的 stuck 字段永远不知道脱困逻辑有没有真的触发过。
                print(f"[STUCK] 卡住 #{self._stuck_count}：{STUCK_FRAMES} 帧没离开锚点 "
                      f"({self.stuck_x},{self.stuck_y}) pos=({px},{py}) → 跳一下并反向转身 "
                      f"STUCK_TURN_MS={STUCK_TURN_MS}ms")
                self._stuck_streak += 1
                # 同一个目标反复卡死 → 拉黑，别再跟它耗（见 TGT_BAD_HITS 那段说明）
                if tgt is not None:
                    _nt = time.time()
                    _k = (int(round(tgt[0] / float(TGT_BAD_CELL))),
                          int(round(tgt[1] / float(TGT_BAD_CELL))))
                    _n0, _t0 = self._stuck_tgt.get(_k, (0, 0.0))
                    if _nt - _t0 > TGT_BAD_FORGET_S:
                        _n0 = 0
                    _n0 += 1
                    self._stuck_tgt[_k] = (_n0, _nt)
                    if _n0 >= TGT_BAD_HITS:
                        self._bad_tgts.append((float(tgt[0]), float(tgt[1]),
                                               _nt + TGT_BAD_KEEP_S))
                        self._stuck_tgt[_k] = (0, _nt)
                        print(f"[TGT] 这块 ({tgt[0]:.0f},{tgt[1]:.0f}) 已经 {_n0} 次卡死够不着 "
                              f"→ 拉黑 {TGT_BAD_KEEP_S:.0f} 秒，改追别的")
                if self._stuck_streak >= STUCK_STREAK_RECHECK:
                    # 跳+转身几次都还在原地，未必是撞墙——也可能是**这个锁已经不跟手了**。
                    # 实测 2026-09-19 两次（08:22 锁在 (641,523) score=0.80、08:30 锁在
                    # (786,631) score=0.60）：锁飘到屏幕固定的计分板/技能面板文字上（掩膜
                    # F1 对这类静态金字同样是高分），pos 于是钉死不动、[STUCK] 连报 157 次，
                    # 而脱困那一下「松开所有键 + 反向转身」反而把正在走的角色拽回来，永远
                    # 到不了目标。这时唯一能自愈的是承认锁废了、退回 NO-LOCK 重新标定。
                    # 判据复用标定那套 `_verify_lock_moves`（按一下看模板跟不跟手），它返回
                    # None 表示「相机在滚/没等到新帧，判不了」——长图上必须放过，不能冤杀。
                    self._stuck_streak = 0
                    v = self._verify_lock_moves()
                    if v is False:
                        print("[STUCK] 连续 %d 次脱困都没挪窝，且探针量到「按键挪不动锁定框」"
                              "→ 判定锁在不响应按键的东西上（计分板/技能面板这类屏幕固定 UI "
                              "文字），丢弃锁定重新标定" % STUCK_STREAK_RECHECK)
                        self.vision.clear_player_lock()
                        self.stuck_frames = 0
                        self._anchor_gray = None
                        self._ring_push_early(bgr)   # 这一支 continue 掉末尾那次 push
                        continue
                self.release_all()
                self._ktap("up", JUMP_TAP_MS)
                # 2026-09-29 修：原来用 `_stuck_count % 2` 每次脱困都换向，也就是
                # 「往右点 100ms → 往左点 100ms → 往右 …」，位移两两抵消，净位移恒为 0。
                # 实测 _bot_run.log：1035 次 STUCK 全挤在 (850,550)±50px 这一格里，
                # 73 分钟没挪出去——**脱困动作本身保证了这个点出不去**。
                # 改成同一个方向连着用 STUCK_SAME_DIR 次再换向，才有净位移。
                self._ktap("right" if (self._stuck_count // STUCK_SAME_DIR) % 2 == 0
                           else "left", STUCK_TURN_MS)

            elif tgt:
                dx = tgt[0] - px
                dy = tgt[1] - py

                # 目标站在**另一层**（上下差超过 SAME_PLATFORM_DY）时把水平死区收窄。
                # 水平死区的前提是「水平对齐 = 到了」，那要求目标和我同层；目标在我下面
                # 70px 的时候，水平再对齐也走不到它，站住就是干等。
                # 实测 2026-09-29（_bot_run.log，73 分钟）：带目标的 1815 次采样里，
                # 「|dx| <= DX_DEAD 且 dy > 0」的占 732 次，这些帧 want 恒为 0、
                # keys 恒为 []，bot 就这么站了 73 分钟，一次芝士都没送进洞。
                # 收窄到 ONLEVEL_DEAD_PX 后，只要还有一点水平偏差就会继续朝目标点方向走。
                dead_x = dx_dead
                # 2026-10-26 二修：判断「洞在另一层」。见 HOLE_LEVEL_HOLD_S 处的说明。
                # 计时/锁死都用 _hole_lvl_since；目标一换（或回到找芝士）就作废。
                _now_mv = time.time()
                if self.state != 1:
                    self._hole_lvl_since = 0.0
                    self._hole_lvl_tgt = None
                elif self._hole_lvl_tgt is not None and (
                        abs(tgt[0] - self._hole_lvl_tgt[0]) > 40
                        or abs(tgt[1] - self._hole_lvl_tgt[1]) > 40):
                    self._hole_lvl_since = 0.0
                    self._hole_lvl_tgt = None
                if (self.state == 1 and 15 <= dy <= SAME_PLATFORM_DY
                        and abs(dx) <= int(HOLE_ENTER_DEAD_PX * scale)):
                    if self._hole_lvl_since == 0.0:
                        self._hole_lvl_since = _now_mv
                        self._hole_lvl_tgt = (tgt[0], tgt[1])
                off_level = abs(dy) > SAME_PLATFORM_DY
                if self._hole_lvl_since and _now_mv - self._hole_lvl_since >= HOLE_LEVEL_HOLD_S:
                    off_level = True
                if off_level:
                    # 目标在另一层：**放宽**成一条带，别再拿 6px 去微调（见 OFFLEVEL_ALIGN_BAND）。
                    # 带外照常朝目标走；带内交给下面的「找平台边缘」。
                    dead_x = max(dead_x, int(OFFLEVEL_ALIGN_BAND * scale))
                elif self.state == 0:
                    # 找芝士时死区必须**小于拾取半径**。芝士是「走过去碰到才捡起来」，
                    # 不像洞口只要人在旁边就行，所以停在不远处等于白走。
                    # DX_DEAD=25 换算到帧上是 int(25*1.196)=30px（实测这个场 scale≈1.196），
                    # 而老鼠和芝士的贴图各约 20/14 px，中心距要 <约17px 才重叠得上。
                    # 实测 2026-09-29（_bot_run_2min.log，18.5 分钟）：离目标 <=30px 的采样
                    # 11 次，**cb 全是 0**；反复停在 23~29px 就不动了，一次奶酪都没背起来。
                    # 收到 8px 后只要还有一点偏差就会继续压上去。
                    dead_x = min(dead_x, int(DX_DEAD_CHEESE * scale))
                elif self.state == 1:
                    # 2026-10-26：背着芝士往洞口走时同样把死区收到 7px。洞心天然比脚底下
                    # 20~50px（洞是地表上的坑），「目标略低 + 水平已对齐」在 dx_dead=30 下
                    # 每一帧都判成「到了」→ want=0 → 站着不动，就是用户说的「在洞口徘徊」。
                    # 实测全天 129 次这个姿态里 126 次 keys=[]。收到 7px 后它会继续朝洞心压，
                    # 越过洞口边缘掉进去，cb 自然掉 0 → 判送洞。
                    # 注：dy 在 20~60 之间时不触发上面的悬崖保护（那是 < EDGE_DOWN_SLACK 才拦）。
                    dead_x = min(dead_x, int(HOLE_ENTER_DEAD_PX * scale))

                # 悬崖保护：要去的方向正好是脚下平台的边缘、而目标又不在下方时，再按下去
                # 就是走出平台。宁可原地停一下（停了会被 [STUCK] 接管去脱困），也别照着
                # 够不到的目标一路走出地图摔死。目标在下方时放行——那本来就是该往下走。
                #
                # 2026-09-29 修：这里的判据本来是 `dy > -EDGE_DOWN_SLACK`，**符号是反的**。
                # dy = tgt_y - pos_y，屏幕坐标往下为正，所以 dy>0 就是「目标在下方」。
                # 原式在 dy > -20 时全部触发，也就是**恰恰在目标明显在下方（该放行）时
                # 把方向键拦掉**，只在目标高出 20px 以上时才不拦——跟上面两行注释
                # （以及 EDGE_DOWN_SLACK 自己的注释）说的正好相反。
                # 后果实测在 _bot_run.log（73 分钟）：带目标的 1815 次采样里 259 次
                # 「水平明明在死区外（dx=+36/+42/-35/-39）却 keys=[]」，全是这里拦的。
                # 目标 (856,649) 在脚底(607)下方 42px 的下层平台上，于是永远走不过去。
                # 现在改成只在「目标不低于脚底 EDGE_DOWN_SLACK」时守边缘。
                want = 1 if dx > dead_x else (-1 if dx < -dead_x else 0)

                # 目标在**另一层**、水平又已经对齐 —— **别站着等**。
                # 实测 2026-10-07（_bot_run_1008_0045_goldgate.log，撑了 81 秒那次锁）：
                # 奶酪在脚下 133px 的下层平台上、只有那一块（cheese=1），bot 水平对着它
                # 站了整整 81 秒（|dx| 一直在 0~8 之间打转，39 次采样里只有 2 次按下过方向键），
                # 全靠 [STUCK] 每 60 帧点一下，一块芝士都没吃到。人在这时候会**沿平台左右走走、
                # 找一个能下去的口子**——照做：朝一个方向走 OFFLEVEL_SCAN_S 秒，再换向。
                # 只在「水平已对齐」时接管：有方向可走时还是照着目标走（那条路更快）。
                # 注意悬崖保护在上面那道 `dy < EDGE_DOWN_SLACK` 里已经放行「目标在下方」的
                # 走出边缘，所以扫描时掉下去=朝目标那一侧掉，是想要的结果。
                if want == 0 and off_level:
                    if dy > 0 and ground_info:
                        # 2026-10-26：目标在**下面那一层**、水平也成列了 —— 要做的不是等，
                        # 是走到脚下这条平台的边上掉下去。取**最近的那个边**：走的路最短，
                        # 也最不容易被中间的平台结构挡住。原来这里只会「左右各走 1.5 秒」
                        # 来回晃，宽平台上永远走不到头（19:59 那次 45 秒一格没挪）。
                        _, gl, gr = ground_info
                        want = 1 if (gr - px) <= (px - gl) else -1
                        self._scan_dir = want
                    else:
                        # 目标在上方：下不去，只能左右试探；照旧按时换向。
                        if time.time() - self._scan_since > OFFLEVEL_SCAN_S:
                            self._scan_since = time.time()
                            self._scan_dir = -self._scan_dir
                        want = self._scan_dir
                else:
                    self._scan_since = time.time()

                edge_blocked = False
                if want and ground_info and dy < EDGE_DOWN_SLACK:
                    _, gl, gr = ground_info
                    edge = int(EDGE_GUARD_PX * scale)
                    if (want > 0 and gr - px <= edge) or (want < 0 and px - gl <= edge):
                        want = 0
                        edge_blocked = True

                # Horizontal
                if want > 0:
                    self._key("left", False)
                    self._key("right", True)
                elif want < 0:
                    self._key("right", False)
                    self._key("left", True)
                else:
                    self._key("left", False)
                    self._key("right", False)

                # Jump toward target
                want_jump = dy < -dy_jump

                # Jump across gaps when target is far。被悬崖保护挡下时不要再起跳——
                # 从边缘起跳同样是离开平台，一样是掉下去。
                if (not edge_blocked) and ground_info and abs(dx) > int(FAR_GAP_DX * scale):
                    _, gl, gr = ground_info
                    if gr - gl < int(PLATFORM_MIN_WIDTH * scale):
                        want_jump = True

                # **按住** up，而不是每帧 _ktap("up", 40ms)。两点原因：
                # 1. 游戏机制上按住 up 就是连续跳（用户 2026-09-19 确认），按住和人手一致；
                # 2. 原来的 _ktap 内部 `time.sleep(ms/1000)` 会**阻塞主循环**：目标在头顶时
                #    每帧都触发，实测有目标的帧里 47% 如此，每帧白卡 40ms。锁定中帧率被压到
                #    11fps（_live_run7/8），而 [PERF] 的 total 只有 17ms——差的 ~70ms/帧
                #    这就是大头。上面「跨缝隙」那条同理，条件持续成立时也会每帧卡 40ms，
                #    所以合并成同一个 want_jump。
                self._key("up", want_jump)

            else:
                # Exploration：**朝一个方向走，撞住了就换边**（原来是恒定向右）。
                # 2026-10-27 02:22 现场：没目标时 bot 永远按住右，撞到墙之后既不
                # 报 [STUCK]（背景假位移在刷计数器，见上面）、也不会换向——整局
                # 就跟一堵墙较劲，一局下来一块芝士都没碰过。这是「看着像机器人」
                # 的最大来源：一个能跑的生物不会对着墙站两分钟。
                if (self.stuck_frames > STUCK_FRAMES // 2
                        and time.time() - self._explore_t > EXPLORE_TURN_S):
                    self._explore_dir = -self._explore_dir
                    self._explore_t = time.time()
                    print(f"[EXPLORE] 没目标又挪不动窝 → 探索方向换成"
                          f"{'右' if self._explore_dir > 0 else '左'}", flush=True)
                # 2026-10-27 03:0x：**探索方向也要守边缘**。EDGE_GUARD_PX 原来只装在
                # 「有目标」那条路上，没目标时它按住一个方向一路走出平台、掉进虚空摔死。
                # 现场两帧（02:36:32 / 02:40:2x）的聊天栏都是「现在 X 是你的萨满」紧跟
                # 一句「你没有奶酪吃了！^_^」，顶栏 老鼠: 24~26 —— 也就是**每回合开场几秒
                # 就死**；死了以后是幽灵，幽灵不带动镜头，`_identify_player` 永远
                # 「按键没有让画面动」→ 可操作=False → 整局白跑。这是「一局下来一块芝士
                # 都没碰过」的上游原因，比目标选择更靠前。
                # 换向之间至少隔 EXPLORE_EDGE_COOL_S 秒，免得站在窄平台上每帧来回抖。
                if (ground_info is not None
                        and time.time() - self._explore_t > EXPLORE_EDGE_COOL_S):
                    _, gl, gr = ground_info
                    edge = int(EDGE_GUARD_PX * scale)
                    d = self._explore_dir
                    if (d > 0 and gr - px <= edge) or (d < 0 and px - gl <= edge):
                        self._explore_dir = -d
                        self._explore_t = time.time()
                        print(f"[EXPLORE] 前面就是平台边缘（脚下平台 {gl}~{gr}，人在 {px}）"
                              f"→ 掉头往{'右' if -d > 0 else '左'}", flush=True)
                self._key("left", self._explore_dir < 0)
                self._key("right", self._explore_dir > 0)
                # 同样是按住而不是每帧点一下：越过 STUCK_EXPLORE_JUMP 之后这个条件**每帧**
                # 都成立，_ktap 会每帧卡 40ms。
                self._key("up", self.stuck_frames > STUCK_EXPLORE_JUMP)

            # ====== 帧环形缓冲：事件触发 + 每帧入队（见 _ring.py）======
            # 放在 _dump_frame 之前：触发时缓冲里正好是「出事前 N 帧」，事件这一帧
            # 作为 post[0] 落盘，前后文都拿得到。DBG_RING 没开时全是空转。
            if self.state != self._state_prev:
                self._ring.event("state_%dto%d" % (self._state_prev, self.state),
                                 note="cb=%d px=%d py=%d" % (cb_count, px, py))
            elif cb_count >= CB_RING_EDGE and self._cb_prev < CB_RING_EDGE:
                self._ring.event("cb_up", note="cb %d->%d" % (self._cb_prev, cb_count))
            elif cb_count <= CB_RING_FALL and self._cb_prev > CB_RING_FALL:
                self._ring.event("cb_down", note="cb %d->%d" % (self._cb_prev, cb_count))
            # 贴到目标跟前那一下（边沿触发：一直贴着不重复存）。
            # 这一段是关键证据来源 —— 机器人挨着芝士、但 has_cheese_on_back 报什么，
            # 只有把这些帧留下来才看得见。见 NEAR_TARGET_PX 处的说明。
            _nd = (abs(tgt[0] - px) + abs(tgt[1] - py)) if tgt is not None else None
            _near = _nd is not None and _nd <= NEAR_TARGET_PX
            if _near and not self._near_prev:
                self._ring.event("near_target",
                                 note="d=%d px=%d py=%d tgt=%d,%d cb=%d cheese=%d"
                                      % (_nd, px, py, tgt[0], tgt[1], cb_count,
                                         len(cheeses)))
            self._near_prev = _near
            self._state_prev = self.state
            self._cb_prev = cb_count
            if self._ring.on:       # 关着的时候连 meta 都不构造，保证零开销
                _rmeta = self._ring_meta(px, py, cb_count, falling, stuck, bbox)
                if time.time() - self._bg_sample_t >= DBG_SAMPLE_S:
                    self._bg_sample_t = time.time()
                    self._ring.sample(bgr, _rmeta, "bg")
                self._ring.push(bgr, _rmeta)

            if time.time() - self._last_dump >= DBG_DUMP_S:
                self._last_dump = time.time()
                self._dump_frame(bgr, player, cheeses, holes, tgt)

            self._dbg = (f"state={self.state} pos=({px},{py}) feet={pfeet} "
                         f"cheese={len(cheeses)} selfdrop={self._selfdrop} "
                         f"hole={len(holes)} cb={cb_count} "
                         f"tgt={'None' if tgt is None else (int(tgt[0]), int(tgt[1]))} "
                         f"src={src} keys={sorted(self._keys)} fall={int(falling)} "
                         f"stuck={int(stuck)} bg={self._bg_shift}/{self._bg_n} "
                         f"score={getattr(self.vision, '_last_score', 0.0):.2f} "
                         # 标签召回率：窗里（any）和身体候选头顶上（exp）。两个都低 =
                         # 名字压根没画出来；any 高而 exp 低 = 名字在，是我的位置算错了。
                         f"cov={getattr(self.vision, '_last_cov_any', -1.0):.2f}/"
                         f"{getattr(self.vision, '_last_cov_exp', -1.0):.2f} "
                         f"ring={'on' if self._ring.on else 'off'}/"
                         f"{self._ring.n_frames}/{self._ring.n_events} "
                         # 迟滞开着没、当前目标守了多少帧/跟丢多少帧 —— 「目标跳变」
                         # 这个核心指标只有配上这三个才有解释力（跳变是抢锁还是跟丢？）
                         f"hyst={'on' if self._tgt_on else 'off'} "
                         f"tage={self._tgt_state['age']} tmiss={self._tgt_state['miss']}")
            self.vision.store_frame(gray)

        print(f"[HOLE] 本场合计：送进洞 {self._delivered} 次，"
              f"芝士离开背但不在洞口 {self._dropped} 次")
        print(f"[TGT] 本场合计：目标事件 {self._tgt_ev} 次，其中抢锁换目标 "
              f"{self._tgt_switches} 次（TGT_HYST={TGT_HYST}）")
        print(f"[RING] {self._ring.status()}")
        self._ring.close()
        print("[Bot] Stopped")


def _other_bot_pids():
    """返回其它正在跑 bot.py 的进程 pid（不含自己和自己的父进程）。

    实测 2026-09-13：前面几轮「重启」没杀掉旧进程，先后有 4 个 bot 同时活着好几分钟，
    全都往同一个角色注入按键。标定探针量到的位移里混着另一个实例的操作，`_world_dx`
    时有时无、锁定结果全部不可信（lock 落到计分板 UI 上就是这么来的）。同时跑两个实例
    没有任何意义，启动时直接拦住。

    **必须先把进程限定成 Python**：只按「命令行里有没有 bot.py 这个词」判会把启动它的
    那个 shell 也算进来——`python -u bot.py > log &` 这条 bash 命令行本身就以 `bot.py`
    为一个词，于是每次从终端启动都会自我拦截（2026-09-15 实测踩到，报的 pid 是那条
    已经退出的 bash）。限定解释器后，误报没了，真正的「两个 python 都在跑 bot.py」
    照样能抓到。
    """
    me, parent = os.getpid(), os.getppid()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in (me, parent):
                continue
            if not (p.info["name"] or "").lower().startswith("python"):
                continue
            cl = p.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(os.path.basename(a) == "bot.py" for a in cl[1:]):
            out.append(p.info["pid"])
    return out


# ── 运行期间不许熄屏／不许锁屏 ───────────────────────────────────────────────
# 2026-10-07：机器人一跑就是几十分钟到几小时。中途只要 Windows 熄屏→睡眠→唤醒，
# 就会弹出**要输密码的锁屏**：dxcam 抓到的那一帧整个变成锁屏画面，视觉全线崩，
# 而且按键还在往前面那个窗口注入。系统层的电源方案已经全改成「永不」，但联想
# Fn+Q／Legion Zone 切性能模式时会**整套换电源方案**，切到「平衡」「自定义模式」
# 就又是 5 分钟熄屏 / 15 分钟睡眠。SetThreadExecutionState 是进程自己提的申请，
# 与当前用哪个电源方案无关，最稳：
#   ES_CONTINUOUS       = 0x80000000  一直有效，直到进程退出
#   ES_SYSTEM_REQUIRED  = 0x00000001  不许进睡眠
#   ES_DISPLAY_REQUIRED = 0x00000002  不许关显示器
# 进程结束这个申请自动失效，不需要手动撤销。
def keep_awake():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            0x80000000 | 0x00000001 | 0x00000002)
        print("[SYS] 已申请保持唤醒：本次运行期间不熄屏、不睡眠、不锁屏")
    except Exception as e:                       # 申请失败不影响正常跑
        print("[SYS] 保持唤醒申请失败（不影响运行）:", e)


if __name__ == "__main__":
    keep_awake()
    others = _other_bot_pids()
    if others:
        print(f"[FATAL] 已经有 bot 在跑（pid {others}）：两个实例会往同一个角色注入按键，"
              f"标定和跟踪结果都不可信。先关掉那个再启动。")
        sys.exit(1)
    b = Bot()
    if b.start():
        b.run()
