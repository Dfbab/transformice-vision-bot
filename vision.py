"""Pure vision detection for Transformice — no memory access."""
import cv2
import numpy as np
import json
import os
import sys
import time


def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def save_png(path, img):
    """存 PNG。**不要用 cv2.imwrite**：路径里带中文（`06_项目`）时它在 Windows 上
    静默返回 False、不抛异常——本目录的 `_lost_window.png` / `_calib_fail_*.png`
    从 2026-10-08 起就一张都没更新过，调试时还以为「没走到存盘那段」。
    走 imencode + 二进制写，与路径编码无关。"""
    ok, buf = cv2.imencode(".png", img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    return bool(ok)


def _mask_f1(win_m, tpl_m):
    """两个「文字像素掩膜」在窗内每个位置的 F1 图（float mask，值域 0~1）。

    F1 = 2*命中 / (2*命中 + 多出来的 + 少掉的)。只在**掩膜**上比，不看原图像素——
    名字标签底下那个半透明气泡是会随地形变的，拿原图比只能拿 0.32~0.75（见
    `set_player_lock` 里记录的失败数据），而掩膜比是恒定的 1.00。bot.py 的
    `_mask_shift` 用的是同一个公式，别改一处的口径。
    """
    tp = cv2.matchTemplate(win_m, tpl_m, cv2.TM_CCORR)
    ker = np.ones(tpl_m.shape, np.float32)
    tot = cv2.filter2D(win_m, -1, ker, anchor=(0, 0))[:tp.shape[0], :tp.shape[1]]
    ts = float(tpl_m.sum())
    return 2 * tp / np.maximum(2 * tp + (tot - tp) + (ts - tp), 1e-6)


def _entry_hw(e):
    """模板条目占的 (高, 宽)。掩膜条目存的是 `tpl_m`，灰度条目才是 `tpl`。"""
    return (e["tpl_m"] if e.get("mask") else e["tpl"]).shape[:2]


# ===== 跟踪模板库参数（阈值都是从实测分布里划的，改前先看图核对）=====
LOCK_STRONG = 0.55     # 这个分数以上算「匹配上了」
LOCK_REFRESH = 0.90    # 这么好才把它当新姿势收进模板库
BANK_CAP = 10          # 库里最多存这么多模板（含 2 个锚点）
WEAK_MIN = 0.40        # 弱匹配下限：实测锁定后第一帧起跳就是 0.51（第二峰 0.35）
WEAK_MARGIN = 0.08     # 峰要明显高于次峰，否则那只是背景噪声
WEAK_JUMP = 40         # 弱匹配位置离上一帧不能超过这么多 px
WEAK_CONFIRM = 12      # 连续两帧弱匹配位置差在这个范围内，才把它学进库
WEAK_MAX = 60          # 弱匹配最多连续撑这么多帧（80fps 下约 0.75s）
LOCK_LOST_GRACE_S = 0.5  # 连续失配多少**秒**才真的放弃锁定（为什么按时间、不按帧数见 track_locked）
# 灰度模板匹配到的窗口，灰度标准差低于这个值就当作「平坦块」作废（见 track_locked 里的说明）。
# 实测 2026-09-19：锁漂到一块平坦木地板时该处 std=12.96（同帧真老鼠处 std=41.75）。
# 取 18 = 比实测的坏样本高 1.4 倍、比好样本低 2.3 倍，两边都留了余量。样本只有一对，
# 如果以后 跟丢 明显变多（把好匹配也误杀了），先回来调这个值。
LOCK_MIN_PATCH_STD = 18
# 灰度候选按标签偏移量推出来的那块位置上，金色名字标签的**召回率**下限（见 _label_cover）。
# 只算召回、不算精确率，因为地图上的金色美术（岩浆亮芯、沙地、狐狸尾巴）总会钻进标签框，
# 算精确率的话真身也会掉到 0.38（2026-09-19 run13 实测）。真身召回≈1.00，漂到 UI/别人
# 老鼠身上≈0，取 0.6 让部分遮挡（岩浆盖住半行字）也还能过，剩下的靠 LOCK_LOST_GRACE_S 兜。
LABEL_COVER_MIN = 0.6
# 掩膜锚点（金色名字标签）这一条**自己**的分数，改用**召回**而不是 F1。
# 为什么：F1 = 2·命中/(2·命中 + 多出 + 少掉)，分母里那个「多出」是**窗里多出来的金像素**，
# 而同一块标签在不同帧里的金像素数根本不稳定（文字抗锯齿、底下半透明气泡后面的地形颜色
# 都在变）。2026-10-07 空房那场实测：标签就在预期位置上、944 个金像素，锁定时模板只有
# ~46 个（掩膜入库门槛是 gm.sum()>=40，刚够过线），命中 46 / 多出 190 → **F1=0.32**，
# 而**召回=1.00**。于是 best_mask 只有 0.27~0.35，mx 全被压在 LOCK_STRONG=0.55 以下，
# 锁在 1.1~1.6s 内必丢、11 次标定无一例外（crowded 场同样症状，空房并不更好）。
# 同一块掩膜在 `_label_cover` 里早就改成只看召回了（见那边的实测 0.38/0.66），这里漏改
# ——「锁不稳」的直接原因就是这一处自相矛盾。
LABEL_MASK_RECALL_MIN = 0.80   # 真标签召回实测≈1.00，留余量；对不上就别认
# 召回是形状无关的：一整块金斑（岩浆亮芯、顶栏那条金条）盖上标签框也能拿 1.00。所以再量
# 一下「标签框里有多少金像素**不是**我这几笔字」占整框的比例。真标签只占百分之几到三成
# （2026-10-07 实测 190/1541 = 12%），整块金斑能占满 → 占满就整条丢掉（不是降分）。
LABEL_MASK_EXTRA_MAX = 0.60
# 2026-10-26：名字/称号文字的**亮度**下限。为什么不用色相：本机客户端上自己的名字是
# 蓝白 BGR(255,243,206)、别人的是暖金/纯白，写死的金色阈值一个都命不中（实测整帧
# gold_mask 在标签框内 = 0 像素）。文字与气泡底的亮度差是唯一稳定的判据：
# 文字 >185，气泡底 BGR(22,45,64) 最高通道也才 64。
LABEL_TEXT_V_MIN = 185
# 建模板时从标签上**采**到的文字色容差（每通道）。跟踪时只在「离这几种色 ≤ 容差」
# 的像素上算召回——这样别的玩家同亮度的名字、工具栏那条近白都不会算进模板里。
LABEL_COLOR_TOL = 30
# 丢锁后「整幅画面搜自己的名字」的门槛。
# ⚠ **必须用 TM_CCOEFF_NORMED 比字形，不能用召回（TM_CCORR）**：召回只看「我的字
# 有几个像素被对上」，而满屏都是文字（聊天栏、顶栏、别的玩家名字），任何一个文字区
# 都能拿到 1.00 —— 实测 live 帧 rec=1.00、次峰也 1.00，判据直接失效。CCOEFF 会减
# 均值再比形状，字形之间的**空白**也是信息，于是同一帧上：真标签 1.00 / 次峰 0.31。
# 2026-10-26 实测 _probe_A/B/C：1.000@(140,333) / 0.879@(167,333) / 0.992@(155,333)，
# 位移与按键完全一致，次峰 0.31~0.37。
# ⚠ 2026-10-27 01:2x 实测 154 帧（1.5x 舞台、52x15 字形）分档：
#   **暖色（自己）n=96：中位 0.784，最低 0.540**；**冷色（别人的名字）n=58：最高 0.338**。
#   原来 0.70 正好切在自己分布的尾巴上 —— 人群里一挤就掉到 0.65~0.69，重挂失败，
#   4 分钟里 6 次重挂 / 5 次跟丢。0.52 离冷色上限 0.338 还有 0.18 的余量，安全。
# ⚠ 2026-10-27 01:4x **回调到 0.60**：0.52 那版实跑 4 分钟里 9 次重挂 / 10 次跟丢，
#   比改之前还差，而且挂上去的位置会跑到**舞台右边缘外 1px**（pos=(1369,y)，
#   stage x1=1368）——那根本不是老鼠能站的地方，说明放进来的就是假峰。
#   真标签在正常帧上是 0.73~0.99，只有被别人名字压住时才掉到 0.54~0.69；而那种帧
#   本来位置就不可信。宁可这一帧挂不上（下一帧还能再挂，代价≈0），也别挂错
#   （整局都歪，代价极大）。位置远近另有一道闸门见 `LABEL_REACQ_NEAR`。
# ⚠ 2026-10-27 02:2x **提到 0.68**，依据是同一晚量到的两堆数：
#   真标签（老鼠活着、没被人压住）：0.72 / 0.79 / 0.80 / 0.86 / 0.87 / 0.88 / 0.89 / 0.93
#   假峰（别人的名字、左下聊天栏里自己名字那几行、并成一排的多行名字）：0.40~0.62
#   0.60 的线正好切在假峰上沿，于是「重挂成功」经常挂到别人头上，撑 5 秒就丢——
#   一晚上大半时间耗在「锁上→丢→退避→再锁」的循环里。0.68 两边都留了余量。
LABEL_REACQ_SCORE = 0.68
# 重挂/跟踪时候选点离「上一次已知位置」多远之内算「近」。自己的老鼠不可能瞬移，
# 而相机是跟着自己走的，屏幕上它基本不动；会跳这么远的只可能是别人的名字。
LABEL_REACQ_NEAR = 260
# 离上次位置太远的候选必须是「一眼就是我」的分数才认（地图切换/重生时确实会瞬移，
# 所以不能一刀切禁掉，只能加价）。
LABEL_REACQ_FAR_SCORE = 0.72
LABEL_REACQ_MARGIN = 0.25
# 扫峰时「值得再看一眼」的最低分。低于它的峰连颜色都不用判——人群里 0.5 以下的
# 峰成片，扫下去纯浪费时间。
LABEL_WARM_FLOOR = 0.50
# ⚠ 2026-10-27 02:1x：**地图缩放。** 客户端会把整张地图缩放着放进 800x600 的舞台，
#   缩放比随地图大小变，于是名字标签的字号也跟着变。同一晚同一局里实测到三种：
#   **52x15 / 身体 72**（1.00x）、**42x13 / 身体 58**（0.81x）、**24x8**（更小的图）。
#   字形模板是按 52x15 存的，拿去匹配 42x13 的活标签峰值只有 0.33~0.40 —— 低于重挂
#   门槛 0.60，于是**换一张大图就整局挂不上**（02:02 那次连续 1000+ 帧 NO-LOCK，
#   而同一时刻点开截图看得清清楚楚：Dfbaa 的奶油色名字就在画面上）。
#   修法：整幅搜索时按下面这组倍率各试一次，命中就记住这个倍率，之后每帧只试它一个
#   （挂着的帧代价不变，只有「挂不上」的帧才会把 7 个倍率都试一遍）。
LABEL_ZOOM_STEPS = (0.62, 0.70, 0.79, 0.89, 1.00, 1.12, 1.26)
# 扫完整组倍率要 ~85ms（实测 1200x900 舞台），不能每帧都扫 —— 挂不上的时候帧率
# 反而是 bot 唯一在做的事，扫太勤会把标定也拖慢。0.5s 扫一次，中间只试当前倍率。
LABEL_ZOOM_SWEEP_S = 0.5
# 当前倍率下打到这么高就不用再扫别的倍率了（「一眼就是我」）。低于它才值得花 85ms
# 把整组扫一遍挑最好的 —— 0.60~0.85 这一段正是「别人的名字/假峰」能混进来的区间。
LABEL_ZOOM_GOOD = 0.85
# **换倍率**的门槛，比重挂门槛（0.60）高得多。老鼠活着时真标签实测 0.79~0.93，
# 而假峰（左下聊天栏里自己的名字、计分板）只有 0.46~0.62。用 0.60 当「换倍率」的
# 判据，一死就必然被假峰改歪倍率，而且改歪之后**真标签再也试不到**（见 02:1x 的现场）。
LABEL_ZOOM_ADOPT = 0.75
# 每帧跟踪的「峰值要比次峰高多少」。比 0.25 松：一屋子 40 个白名字时，次峰常常
# 0.6~0.7，而自己的暖金字峰值只有 0.85 —— 按 0.25 算就永远不够差。
# 现在**冷暖色那道闸门**（_label_warm）已经把人筛掉了，差一点就够用。
LABEL_TRACK_MARGIN = 0.08
LABEL_REACQ_EXCL = 24          # 判次峰时避开主峰多大的方框（字形宽 28~43px，取 24 够）
# 每帧跟踪也可以直接走标签（见 _track_by_label）。门槛比重挂高一点：重挂失败只是
# 等下一帧，误挂会把位置钉在别人头上，代价大得多。
# 2026-10-27 01:4x：0.55 → 0.60，和重挂统一（理由同上一条：0.55 那版里
# 「刚锁上就被丢」10 次/4 分钟，锁上的是假名字，位置一直贴在舞台右边缘）。
LABEL_TRACK_SCORE = 0.60
LABEL_TRACK_JUMP = 220         # 一帧之内位置跳超过这么多 = 挂错人了，拒掉
LABEL_ANCHOR_JUMP = 200        # 跟踪中拿名字标签纠偏时，允许一次掰多远（超过就当标签认错了）
LABEL_TPL_KEEP = 0.92          # 旧字形在这帧上还打得这么好，就别让新标定顶掉它
# 「存档字形对上了 → 直接锁定」的门槛。实测种子模板在同局另一帧上 0.948 / 次峰 0.440
# （亮度掩膜口径），所以峰值门槛留 0.90、次峰差留 0.35 —— 原来写 0.95/0.45 卡得太死，
# 0.948 会被判「不够」而退回按键标定。
LABEL_TRUST_SCORE = 0.90
LABEL_TRUST_MARGIN = 0.35
# 「名字看不见，但身体匹配得又高又没挪窝」的豁免门槛。名字消失不等于候选是错的：
# 半透明聊天栏/道具栏压在身上、或者精灵正好走到面板后面时，金色掩膜是 0，而身体
# 灰度能到 0.85~0.98 且位置一动不动（2026-09-19 run23 实测两条：0.98/离上一帧0px、
# 0.85/38px）。所以不能一刀切压分。区分「真身被压住」和「匹配飘到隔壁周期纹理上」
# 只靠**位置连续性**——分数区分不了（那个漂移样本原始分也有 0.57~0.60）。
# 取 0.70：明显高于漂移样本的 0.57~0.60，又低于真身样本的 0.85~0.98，两边都有余量。
# ⚠ 2026-10-07 从 0.70 降到 0.55（LOCK_STRONG）：上面那个 0.70 是按「真身 0.85~0.98 /
# 漂移样本 0.57~0.60」划的，但当晚一整场实测真身被地形或别的老鼠遮住时只有 0.53~0.57
# （23:28:58 那次：灰 0.57、离上一帧 2px、标签召回 0.11 —— 明摆着就是它，却刚好卡在
# 门外被压成 0.45 丢掉，7 次跟丢一个形状）。当初要防的漂移样本是「一下跳到 38px 外」，
# 拦住它的是 NO_LABEL_JUMP 而不是分数，所以这里放开是安全的：
#   · 静态背景（分数 ~1.00、原地不动）在 0.70 时也照样能过这条，跟这次改动无关；
#   · 分数落在 0.55~0.70 的只有「精灵被挡住/半截露出来」这一种情况。
NO_LABEL_MIN = 0.55
NO_LABEL_JUMP = 20     # 豁免时位置离上一帧不能超过这么多 px（漂移样本是 38px）
# 而且**整个搜索窗里都得找不到自己的名字**（_last_cov_any < LABEL_COVER_MIN）才算
# 「名字被压住了」。窗里名字在、只是不在候选头顶 ⇒ 那是别的老鼠（同房老鼠长得一样）。
#
# ⚠ 豁免必须**限量**（2026-10-07 实跑才发现的漏洞）：上面那条「名字被压住」的设计前提是
# **一两秒后名字会回来**（run17 实测 cov 掉到 0.00~0.10、一两秒后回到 1.00）。但豁免是
# 无限续期的——只要窗里一直找不到金子（cov_any 恒为 0），一个 0.99 的**静态背景**就能
# 一帧一帧把自己续下去：它分数高、原地不动，三条判据全过。那晚三次锁定分别豁免了
# 109 / 474 / 494 帧，锁钉在背景边上，bot 按着方向键打了半分钟空气；
# `_nolabel_keep` 原来只被打印、**从来没当上限用过**。
# 做法是**预算 + 回满**（不是纯计时）：每豁免一帧扣一格，**任何一帧看到标签就把预算
# 回满**。静态背景那条路因此是死胡同——它永远产生不了验证帧，预算用完就再也豁免不了；
# 而「真身被面板短暂压住」每次都有标签回来那一帧把预算回满，短时遮挡一次都不受影响。
# 用帧数不用秒：见 LOCK_LOST_GRACE_S 的说明，锁定中帧率 9~18fps 飘得厉害，帧数简单够用。
NOLABEL_KEEP_MAX = 45

# 查标签召回率时允许身体匹配位置有几像素误差（名字是 1~2px 粗的抗锯齿笔画，单点比对
# 错开一像素召回就掉一半）。见 _label_cover。
#
# ⚠ 2026-10-07 从 4 放大到 10：4 是按「匹配位置只有几像素误差」估的，**实测误差比这大**。
# 23:20:22 那条跟丢：真身灰度 0.82、离上一帧只有 5px（明摆着就是它），可 cov 只量到
# 0.51 < LABEL_COVER_MIN，被压成 0.45 丢掉——回头看标签召回图，标签峰就在 5px 外，
# 而 ±4 的搜索框差 1px 没够着。根因是**精灵走路/跳跃的动画**：名字标签挂在角色的锚点上
# （静止），而灰度模板匹配到的「身体左上角」会落在精灵的不同部位上，同一只老鼠前后两帧
# 能差 ±10px。所以搜索半径得按这个抖动量给，不能按「笔画粗细」给。
LABEL_COVER_SLACK = 10

# ===== 名字标签条目的入库存量检查 =====
# 标签是贴在身体**正上方**的静态文字，正常偏移只有几十像素（实测 (-13,-5)、块 30x9 或
# 35x23）。一旦标定把远处的另一块金字并进来（见 bot.py `_identify_by_label` 的并块判据），
# 偏移会飙到几百、块也会被拉成一条：2026-09-28 实跑量到过 ox=339、条 736 宽。
# 那种条目一进库，搜索窗就按它撑到整幅（[LOCK] 搜索窗异常大），灰度模板在哪儿都能凑出
# 高分、掩膜也总能找到金像素，锁立刻被拽走。这里挡一道，别让任何调用方污染模板库。
# ⚠ 2026-10-27 02:30 收紧：160x40 太松，28 人的房间里名字互相压着，标签块会被
# **好几行名字并成一条**——现场实测建库时拿到过 `标签(253,707,151,14)`（三行并排）
# 和 `标签(1126,156,110,52)`（两层叠着），这不是「尺寸略大」，这压根不是一块标签：
# 拿它当锚点，搜索窗撑成一条 151 宽的长条，跟两步就飘走。真标签实测只有两种：
# 52x15（1.00x）和 41x13（0.79x），留到 90x30 已经绰绰有余。
TAG_MAX_OFFSET = 110   # 标签左上角离身体左上角最远这么多 px
TAG_MAX_W = 90         # 标签块本身的尺寸上限
TAG_MAX_H = 30

# ===== 舞台（Flash 画布）以外的客户端底色条 =====
# 窗口比舞台大：本机实测窗口 1114x815，舞台只有 800x600（居中，左上角 (157,119)），
# 四周那条 BGR(149,116,106) 的底色是客户端自己画的。玩家只可能出现在舞台里，所以
# 跟踪的搜索窗和位置都要夹进舞台。2026-09-19 run20 实跑：锁飘进右边那条底色带之后
# 整窗变成纯色，`_lost_window_c_final.png` 就是一张纯色图（跟丢最后的 0.45 分来自它）。
STAGE_TOL = 26           # 判定「这一像素还是底色条」的每通道颜色容差
STAGE_MIN_RUN = 8        # 底色带至少这么宽才算数（否则只是地图上碰巧同色的像素）
STAGE_MIN_FRAC = 0.40    # （已不用；舞台占比会随窗口大小变，见 stage_rect 的校验说明）
STAGE_MIN_W = 320        # 检测出来的舞台至少这么大，否则当失败退回整帧
STAGE_MIN_H = 240        # 真舞台是 800x600，这里留很大余量，只用来挡「一条窄带」那种坏结果


class Vision:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(_app_dir(), "vision_config.json")
        self.cfg = self._load_config(config_path)
        self._last_player = None       # (cx, cy, feet_y, bbox)
        self._player_lost_frames = 0
        self._prev_gray = None
        # 输入相关法锁定的本地玩家模板（与皮肤无关）
        self._lock_patch = None
        self._lock_bbox = None
        self._lock_lost = 0
        self._lock_lost_since = 0.0    # 本次「连续失配」是从什么时候开始的（0 = 没有在失配）
        self._lock_win_warned = True   # 「搜索窗撑爆」只报一次；True = 还没锁定，不用报
        self._lock_bank = []           # 多姿势模板库：锚点 + 镜像 + 途中学会的姿势
        # 锁定那一刻从**自己名字标签**上采到的文字色（见 dominant_text_colors）。
        # 跟踪时用它代替写死的金色阈值；空 = 还没锁过，按亮度兜底。
        self._label_colors = []
        # 名字标签的**字形掩膜** + 它到身体框的偏移。整场有效、不随锁清掉：
        # 丢锁后可以直接在整幅画面上搜这块字形，几毫秒就能把锁挂回去（见
        # reacquire_by_label）——不用再按键、不用等两三秒的标定。
        self._label_tpl_m = None       # float32 0/1 掩膜（已换算到当前舞台缩放比）
        self._label_body_off = None    # (dx, dy) = 身体框左上角 − 标签左上角
        self._label_body_size = None   # 当时锁的身体框边长
        # 存盘时的**原生**字形 + 几何 + 当时的舞台缩放比（见 _apply_label_scale：
        # 舞台从 1x 变 1.5x 时同一个字的字形也大 1.5 倍，不换算是永远挂不上的）
        self._label_tpl_raw = None
        self._label_off_raw = None
        self._label_size_raw = None
        self._label_scale_raw = 1.0
        self._label_scale_cur = None
        # 舞台缩放（帧尺寸/800x600）和**地图缩放**是两回事：前者恒定，后者随地图大小
        # 变（见 LABEL_ZOOM_STEPS）。两个都要乘进字形尺寸里。
        self._label_stage_scale = 1.0
        self._label_zoom = 1.0
        self._label_sweep_t = 0.0      # 上次扫完整组地图缩放倍率的时刻（见 LABEL_ZOOM_SWEEP_S）
        self._load_label_tpl()         # 上次进程留下的字形：开局第一帧就能重挂
        # 背芝士的「跟不跟手」记忆：(质心 − 身体框左上角, 身体框左上角)。
        # 万圣节地图满地南瓜灯，都是黄色块，站在它旁边四道闸门全能过——
        # 判据换成「这块黄跟着我走没有」：跟着走 = 我在驮它；不跟 = 地图装饰。
        self._cb_rel = None
        self._weak_frames = 0
        self._weak_cand = None
        self._stuck_history = []
        # 2026-10-27 00:2x：**真芝士的图样模板。**
        # 活动期的芝士是「青绿霉芝士」，色相和地图上的树叶装饰**一模一样**
        # （实测都落在 H≈44/S≈158）：`_snap_a1_raw.png` 右侧那片大绿叶被当成芝士，
        # 机器人一路追过去、够不着就原地跳，连续几分钟 [STUCK]。
        # 色相分不开，但**图样分得开**：拿 `_full_now3.png` 里那块真芝士（44x25，787px）
        # 裁成 46x28 模板去匹配 —— 真芝士 1.000，那片叶子 0.300，全屏扫最高才 0.468。
        # 所以色掩膜只当「候选」，能不能算芝士由这里一锤定音。
        self._cheese_tpl = None
        self._load_cheese_tpl()
        # Pre-allocate reusable kernels
        self._k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        self._k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._k10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        # Terrain mask cache
        self._terrain_cache = None
        self._terrain_frame = -1
        # 舞台矩形缓存：按帧尺寸缓存（窗口被重建/改尺寸会换尺寸，跟 absdiff 同一条路）
        self._stage_key = None
        self._stage_rect = None
        self._stage_fail_t = 0.0   # 上次「舞台矩形检测失败」的时刻（失败结果只缓存 1 秒）
        self._stage_good = None    # 最近一次**测成功**的舞台矩形（含帧尺寸键），失败时兜底用
        self._stage_good_key = None
        self._noted_label_less = False   # 「无金标签但有灰度高分」每次锁定只报一条
        self._nolabel_keep = 0           # 本次锁定里被 NO_LABEL_MIN 豁免过多少帧
        self._nolabel_run = 0            # 本次「看不见标签」已经豁免掉多少帧（看到标签就回满）
        self._lock_shape = None          # 建锁时那一帧的灰度尺寸（换尺寸即作废）
        self._last_score = -1.0          # 最近一次 track_locked 的分数（-1 = 这次没得出结论）
        self._last_nolabel_run = -1      # 上面那个豁免预算的对外快照（日志/存图用）

    def _load_config(self, path):
        defaults = {
            "cheese_hsv_lower": [18, 140, 140],
            "cheese_hsv_upper": [35, 255, 255],
            # 2026-10-26：万圣节活动把芝士换成了**发绿的霉芝士**（截帧 _zoom_pump.png 实测
            # H 41~56 / S 121~180 / V 84~224），原来的黄区间（H 18~35）一格都罩不住。
            # 结果整晚机器人看到 3 个「芝士」全是地图上的南瓜灯，真芝士一个没认出来 ——
            # 0 送洞的直接原因就是这个，不是跟丢、也不是背带判据。附加区间把绿芝士收进来。
            # 2026-10-26 22:5x 实测收窄：原来写的 [38,105,75]~[58,210,240] 太宽，
            # 树上的**叶簇**（H38~41 / S184 / V136~203）照样命中，机器人整节都去追树叶。
            # 同一批帧里量到的**真芝士**是 H44 / S158~166 / V150~173（六块全是这个色），
            # 两边在 S 上差 20、H 上差 3~6，所以收成 [42,120,100]~[49,180,200]：
            # 六块真芝士全留、叶簇全走。
            "cheese_hsv_extra": [[42, 120, 100], [49, 180, 200]],
            # 2026-10-26 22:5x：**只认绿芝士**。活动期的芝士是灰绿色霉菌块，
            # 而没有一样真芝士是黄的了 —— 但黄区间（H18~35 S/V≥140）恰好把
            # **地上的草**（实测 H20~28 S174~212 V175~219）全罩住。
            # 后果非常隐蔽：detect_cheeses 报的一堆「芝士」全是草；
            # has_cheese_on_back 里老鼠**脚边那撮草**正好落在身体框内 → 四道闸门全过
            # → 判定「背上了芝士」→ 掉头往洞口跑 → 两秒后草没跟住 → 记一次「掉芝士」。
            # 用户看到的「在洞口徘徊、不搬芝士」就是这个循环。
            # 活动结束后把这一项改回 false 就恢复黄芝士。
            "cheese_green_only": True,
            "player_hsv_lower": [4, 30, 40],
            "player_hsv_upper": [28, 220, 240],
            "hole_gray_threshold": 35,
            "hole_min_area": 80,
            "hole_circularity_min": 0.4,
            # 2026-10-26 23:3x 实测：万圣节的洞口是一圈**洋红色发光描边** + 黑腔体
            # （_z_hole2.png）。描边实测 HSV H150 / S74 / V205，BGR≈(204,145,205)。
            # 灰度阈值那套在这批地图上完全失效 —— 它把**砖缝里的小暗点**当洞口，
            # 一帧能报 20~25 个（间距整齐的 50px 网格，见 _obs2302_08.png 的蓝圈），
            # 于是「拿着芝士往哪跑」永远跑错。
            "hole_magenta_lower": [138, 45, 110],
            "hole_magenta_upper": [168, 255, 255],
            "hole_magenta_fill_max": 0.32,   # 洞是**空心环**，实心块不是洞
            "hole_magenta_dark_min": 0.30,   # 环内必须是黑的（实测真洞 0.34~0.74，杂项 ≤0.01）
            # 2026-10-26 23:5x：**改回 false。** 上面那句「只认洋红」是一晚的临时禁令，
            # 实测代价太大：活动图里只有一小部分地图的洞带洋红边（还有带青色边的，
            # 见 _last_01.png；也有完全不发光的），于是整晚 STAT 里 hole 恒等于 0，
            # 送洞数也就恒等于 0。现在按「洋红洞优先，找不到再退回灰度法」跑，
            # 灰度法那边靠 UI_RECTS 的修正已经不会再满屏砖缝了（实测一帧 3 个候选，
            # 而修正前是 20~25 个）。
            "hole_magenta_only": False,
            "scale_thresholds": True,
            "play_area_top": 0.05,
            "play_area_bottom": 0.645,
        }
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
                defaults.update(loaded)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return defaults

    def _scale(self, frame, base_px):
        """Scale a pixel threshold to current frame size (baseline 600px min dimension)."""
        if not self.cfg["scale_thresholds"]:
            return base_px
        h, w = frame.shape[:2]
        return max(1, int(base_px * min(w, h) / 600.0))

    def _play_area(self, frame):
        """游玩区在整帧里的 y 范围（像素）。

        顶栏、聊天框、右下角技能/背包面板都是**屏幕固定**的 UI，里面全是黄橙色
        图标，不过滤掉的话 detect_cheeses 会把它们当芝士——实测 8 个"芝士"里
        有 5 个就是这些 UI 图标和升级奖励图标（见 _cheese_montage.png）。
        """
        h = frame.shape[0]
        return (int(h * self.cfg["play_area_top"]),
                int(h * self.cfg["play_area_bottom"]))

    def _stage_scale(self, bgr):
        """舞台相对 800x600 的缩放比（150% DPI 的机器上是 1.5）。判不出来就当 1.0。"""
        try:
            x0, y0, x1, y1 = self.stage_rect(bgr)
            s = (y1 - y0) / 600.0
            if 0.5 <= s <= 4.0:
                return s
        except Exception:
            pass
        return 1.0

    def _load_cheese_tpl(self):
        """读芝士图样模板。**两档尺寸**，缺一不可：

        * `tpl_cheese_1027.png`      46x28  —— 小块芝士（44x25 楔形）
        * `tpl_cheese_big_1027.png`  69x38  —— 大块芝士（67x36，地图上的「大芝士」）

        2026-10-27 00:2x 实测：只用小模板时**大块真芝士只拿 0.309**，和那片假叶子
        的 0.300 完全分不开；换成大模板后大芝士 1.000 / 叶子 0.258 / 全屏最高 0.427。
        所以两档一起打分取最大，谁近就用谁。

        读不到就退回空列表：整条模板校验自动不生效，行为与改前完全一致。
        """
        tpls = []
        for fn in ("tpl_cheese_1027.png", "tpl_cheese_big_1027.png"):
            p = os.path.join(_app_dir(), fn)
            try:
                t = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                t = None
            if t is not None and t.size:
                tpls.append(t)
        self._cheese_tpl = tpls or None

    def _cheese_tpl_score(self, bgr, bbox):
        """这块绿/黄斑长得像不像真芝士（0~1，取各档模板最高分）。没模板返回 1.0（不拦）。

        ⚠ 2026-10-27 01:0x 改法：**先把斑块缩放到模板尺寸再比，别在带背景的大窗口里比。**
        上一版是「斑块四周留一圈背景 → 在那块窗口里滑模板」，结果是**背景在拖分**：
        同一块大芝士在深蓝地图上 0.650、浅蓝地图 0.512、黄土地上 0.389、深紫上 0.286
        —— 四张图放大看**全是真芝士**（`_blob_cmp.png`），只是周围底色不同。
        阈值 0.55 于是把 4 块真芝士里的 3 块直接毙了，这正是「芝士只在 2.5% 的帧里看得见」
        的主因：不是没芝士，是**看见了不认**。
        实测（pad 留 0 / 2 像素，把斑块 resize 到模板尺寸后比）：
            真芝士 0.824 / 0.823 / 0.567 / 0.768
            树叶   0.284 / 0.331
        中间空了 0.23，阈值放 **0.45** 两头都稳。
        """
        if not self._cheese_tpl:
            return 1.0
        x, y, cw, ch = bbox
        if cw < 8 or ch < 6:
            return 0.0
        best = 0.0
        for pad in (0, 2):
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(bgr.shape[1], x + cw + pad)
            y1 = min(bgr.shape[0], y + ch + pad)
            reg = bgr[y0:y1, x0:x1]
            if reg.shape[0] < 6 or reg.shape[1] < 8:
                continue
            for tpl in self._cheese_tpl:
                th, tw = tpl.shape[:2]
                # 尺寸归一：斑块多大都缩到模板大小，天然对缩放免疫
                g = cv2.resize(reg, (tw, th), interpolation=cv2.INTER_AREA)
                r = cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED)
                _, mv, _, _ = cv2.minMaxLoc(r)
                best = max(best, float(mv))
        return best

    # ====== Cheese Detection ======
    # 2026-10-05 实测记录：**别再试「圆度 / 面积下限 / H 方差」来压误检**，三条都试过：
    #   · 圆度下限（照 detect_holes 的 hole_circularity_min 那套）：在**芝士堆**地图上
    #     检出数 11 → **0**，真芝士全灭。原因是挨着的芝士会被 MORPH_CLOSE 连成一个
    #     大团（实测 area=1463 circ=0.27、area=993 circ=0.23、area=869 circ=0.18），
    #     团越大连成一片圆度越低 —— 而木板假阳性的圆度是 0.42~0.70，跟真芝士
    #     （0.70）完全重叠。圆度**不区分**这两类，只会杀真阳性。
    #   · 面积下限（想滤掉 55~150 的小碎块）：芝士堆里单块芝士面积就是 55/77/150，
    #     一起被滤掉。
    #   · 色块内 H 方差：木板假阳性 H 标准差 0.0~1.7，跟平色贴图的真芝士一样平，分不开。
    #   （形态学开运算本来就已经在这两行下面做了，plan 里那条是重复项。）
    # 唯一实测有效的是把 cheese_hsv_lower 的 S 从 120 降到 100（真芝士像素 S 的
    # 5% 分位正好=100，S=120 会切掉它 38% 的黄色像素，把一块 44x24 的团削成
    # 44x14 的细条、长宽比 3.14 被 0.4~2.5 那道闸挡掉）——但代价是全场检出数
    # 从 7 涨到 18，且**没有带标注的语料能证明召回没掉**，所以默认值不动。
    def _cheese_mask(self, hsv):
        """芝士色掩膜：主区间（黄）+ 附加区间（万圣节的青绿霉芝士）。

        平时两段并起来；但**活动期芝士整块变成灰绿色**时，黄区间只会把地上的草
        收进来（见 cheese_green_only 那段说明），这时就只留绿区间。
        """
        ex = self.cfg.get("cheese_hsv_extra")
        if ex and self.cfg.get("cheese_green_only"):
            return cv2.inRange(hsv, np.array(ex[0], dtype=np.uint8),
                               np.array(ex[1], dtype=np.uint8))
        lo = np.array(self.cfg["cheese_hsv_lower"], dtype=np.uint8)
        hi = np.array(self.cfg["cheese_hsv_upper"], dtype=np.uint8)
        m = cv2.inRange(hsv, lo, hi)
        if ex:
            m = cv2.bitwise_or(m, cv2.inRange(
                hsv, np.array(ex[0], dtype=np.uint8), np.array(ex[1], dtype=np.uint8)))
        return m

    def detect_cheeses(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = self._cheese_mask(hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k7)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cheeses = []
        h, w = bgr.shape[:2]
        ytop, ybot = self._play_area(bgr)
        # ⚠ 2026-10-27 01:0x：**下面所有面积/边长阈值都是按 800x600 舞台写死的。**
        # 这台机器 150% DPI，bot.py 量到的帧是 1536x1050、舞台 1201x901（s≈1.5），
        # 芝士面积要乘 s²≈2.25：实测那块大芝士 1721px@1x → 3872px@1.5x，
        # 原来 `area > 3000` 直接把它毙了。所以面积统一折回 1x 再比。
        _s = self._stage_scale(bgr)
        _a2 = _s * _s
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 面积过小=噪点，过大=大块黄色地形/UI（阈值乘 s² 折回不同缩放下等价）
            if area < 30 * _a2 or area > 3000 * _a2:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            # 贴边=被截断的残影或 UI，丢弃
            if x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2:
                continue
            aspect = cw / max(ch, 1)
            extent = area / (cw * ch) if cw * ch > 0 else 0
            if aspect > 2.5:
                # 芝士接近圆/方，纵横比过滤掉细长条。
                # 2026-10-08 例外（实测 _diag_look.png 那块 68x35 的大芝士）：
                # 带色调的地图（紫/粉）把芝士**下半部分**压成 H≈11 S≈190 V≈55
                # ——和地面一个色，掩膜里只剩上中部一条 66x23 的亮带（纵横比 2.87），
                # 原本被这道闸丢掉 → 整张图检出 0 个芝士。而亮带形心 (725,406)
                # 离芝士真实中心 (722,401) 只差 5px，**照用亮带形心就行，不用补框**
                # （第一版按「高=宽」补成方框、形心下移 21px，反而落到芝士下面的地上，
                #  5 倍放大图一看就露馅了）。
                # 尺寸卡这么紧是为了不把这道闸本来挡着的**木板**假阳性放进来：
                # 木板又长又扁（实测纵横比十几），aspect<=5 就进不来。
                if not (20 <= cw <= 90 and 8 <= ch <= 32
                        and aspect <= 5.0 and extent >= 0.35):
                    continue
            elif aspect < 0.4:
                continue
            # 饱满度=面积/外接矩形面积，太低说明形状松散
            if extent < 0.3:
                continue
            # 万圣节的**绿芝士**要额外收紧形状。绿色在地图里到处都是（树叶/苔藓/草地），
            # 光靠色相分不开：实测 `_full_now.png`（绿色多的那张图）绿区间炸出 19 个假阳性，
            # 而真芝士在 `_full_now3.png` 上是 **44x26 / 798px / 饱满度 0.70**（实心楔形）。
            # 假阳性饱满度只有 0.17~0.57、面积普遍 <300。所以绿块另立一条：实心 + 够大。
            # ⚠ 2026-10-26 23:0x 实测把上一版的门槛推翻了：原来要求「饱满度≥0.55 且 ≥250px」，
            # 但当天那张图上**驮在老鼠背上的真芝士**实测是 375px/0.47、314px/0.34、
            # 298px/0.40、279px/0.39、202px/0.40、193px/0.42 —— 六个全是 0.34~0.47，
            # 一条都过不了 0.55；反倒是草地（H20~28）因为色相 ≤36 完全绕开了这道闸。
            # 现在按实测分布放宽（面积 ≥150、饱满度 ≥0.32、纵横比 0.9~2.4），
            # 草地则靠 cheese_green_only 在掩膜层就挡掉了。
            _ys, _xs = np.nonzero(mask[y:y + ch, x:x + cw])
            if len(_ys):
                _hue = float(np.median(hsv[y + _ys, x + _xs, 0]))
                if _hue > 36 and (extent < 0.32 or area < 150 * _a2
                                  or not (0.9 <= aspect <= 2.4)):
                    continue
                # 绿的候选再过一遍图样校验（见 _load_cheese_tpl 那段说明）。
                # 只对绿块做：平时的黄芝士和图样模板不是一回事，套上去会全灭。
                # 阈值 0.55 → 0.45（见 _cheese_tpl_score 里的实测分布：
                # 真芝士最低 0.567、树叶最高 0.331，原来的 0.55 会误杀真芝士）
                if _hue > 36 and self._cheese_tpl_score(bgr, (x, y, cw, ch)) < 0.45:
                    continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # UI 区（顶栏/聊天/技能面板）里的黄色图标不是芝士
                if not (ytop <= cy <= ybot):
                    continue
                cheeses.append((cx, cy))
        return cheeses

    # ====== Hole Detection ======
    def detect_holes(self, bgr, gray):
        """洞口：先试万圣节那套洋红描边，找不到再退回老的灰度阈值法。

        为什么要有两套：活动期所有地图的洞口都套了一圈洋红发光描边（形状像
        一个拱门，里面是黑的），这是**别的地图元素没有的**特征；而老的灰度法
        在活动地图上会把砖缝暗点全当洞口（实测一帧 20~25 个），拿它导航等于
        往随机方向跑。活动结束后描边消失，自动退回老方法。
        """
        holes = self._detect_holes_magenta(bgr)
        if holes or self.cfg.get("hole_magenta_only"):
            return holes
        return self._detect_holes_gray(bgr, gray)

    def _detect_holes_magenta(self, bgr):
        lo = np.array(self.cfg["hole_magenta_lower"], dtype=np.uint8)
        hi = np.array(self.cfg["hole_magenta_upper"], dtype=np.uint8)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        h, w = bgr.shape[:2]
        ytop, ybot = self._play_area(bgr)
        V = hsv[:, :, 2]
        fill_max = float(self.cfg["hole_magenta_fill_max"])
        dark_min = float(self.cfg["hole_magenta_dark_min"])
        out = []
        for i in range(1, n):
            area = int(st[i, cv2.CC_STAT_AREA])
            if area < 100:
                continue
            x, y, bw, bh = [int(v) for v in st[i, :4]]
            if bw < 16 or bh < 16 or bw > 120 or bh > 120:
                continue
            asp = bw / float(max(bh, 1))
            if not (0.55 <= asp <= 2.4):
                continue
            fill = area / float(bw * bh)
            if fill > fill_max:          # 实心块 = 装饰/图标，不是空心拱门
                continue
            # 环内（bbox 中部偏下，正好是黑洞腔的位置）必须是黑的
            y0, y1 = y + int(bh * 0.30), y + int(bh * 0.95)
            x0, x1 = x + int(bw * 0.25), x + int(bw * 0.75)
            if y1 <= y0 or x1 <= x0:
                continue
            dark = float((V[y0:y1, x0:x1] < 45).mean())
            if dark < dark_min:
                continue
            cx, cy = int(cen[i][0]), int(cen[i][1])
            if not (ytop <= cy <= ybot):
                continue
            if self._in_ui(cx, cy, w, h):
                continue
            out.append((cx, cy, float(area), dark))
        # 一局最多 3 个洞口，按「面积大 + 内部黑」排，多的丢掉（假阳性通常又小又不够黑）
        out.sort(key=lambda t: -(t[2] * (0.5 + t[3])))
        return [(x, y) for x, y, _, _ in out[:3]]

    def _detect_holes_gray(self, bgr, gray):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = self.cfg["hole_gray_threshold"]
        _, binary = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._k5)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._k3)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        holes = []
        h, w = bgr.shape[:2]
        ytop, ybot = self._play_area(bgr)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            min_area = self.cfg["hole_min_area"]
            # 面积过小=噪点，过大=整块黑背景
            if area < min_area or area > 8000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # 圆形度：洞近似圆，1.0=正圆；太低=不规则形状
            if circularity < self.cfg["hole_circularity_min"] or circularity > 1.2:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)
            if aspect < 0.3 or aspect > 3.0:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            # 凸度=面积/凸包面积，洞应是实心圆，太低说明有凹陷缺口
            if solidity < 0.6:
                continue
            # 贴边=被截断，丢弃
            if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # 聊天框/技能面板里的深色圆形图标同样不是洞口
                if not (ytop <= cy <= ybot):
                    continue
                # 2026-10-27 00:0x：**洞口底下必须有实地。**
                # 重新放开灰度法之后（hole_magenta_only=False），最大的假阳性来源是
                # 「夜空里的蝙蝠 / 暗云」——它们的灰度、面积、圆度、凸度全都合格，
                # 一帧报 3~6 个，全在天空里（_z_track3.png 的蓝圈：218,163 / 255,157 /
                # 573,188 全是蝙蝠和云）。洞是**长在地面或墙里的**，正下方必然是
                # 泥土/木板/草地这类亮而艳的地形；天空是又暗又不艳的一片。
                below = hsv[y + ch + 1:y + ch + 12, x:x + cw + 1]
                if below.shape[0] < 4:
                    continue
                ground = float(((below[:, :, 1] > 90) & (below[:, :, 2] > 70)).mean())
                if ground < 0.35:
                    continue
                holes.append((cx, cy))
        return holes

    # ====== Input-correlation player identification ======
    # 颜色法在自定义皮肤下失效（掩膜命中的是草地/UI/树花），所以改为「按方向键看
    # 谁动」。相机是静止的（实测远处地形位移中位数 0），所以屏幕上会动的就是玩家
    # 自己。名字标签（描边白字，永远在老鼠正上方、与皮肤无关）用来辅助确认锁定位置。
    # 下面这些只看像素形状/位移，不看颜色。

    # 屏幕固定 UI 的区域（占窗口的比例）——这些地方也有白字，但不是老鼠标签
    # 2026-10-26 23:5x：**这四条矩形以前把整张地图的下半截都判成了 UI。**
    # 老值 (0.00,0.42,0.40,1.00) / (0.60,0.42,1.00,1.00) / (0.00,0.60,1.00,1.00)
    # 合起来等于「y>42% 全封、y>60% 更封」——只有 y<0.42·600=252px 这一条横带是「地图」。
    # 而 Transformice 的**洞口有大量地图就落在下半屏**，于是 detect_holes 永远返回 0
    # （2026-10-26 一整晚 STAT 里 hole 全是 0，送洞数自然是 0）。副作用还有：
    # 老鼠走到左下角时自己的名字标签被当聊天框文字剔掉 → 反复跟丢。
    # 下面这组是按 _b_3.png 实测的：顶栏 0~14px；聊天框 x0~0.60、y399~652；
    # 右侧物品栏/积分板 x0.60~1.0、y399~655。UI 真正的上边缘是舞台 y≈399 = 0.665。
    UI_RECTS = (
        (0.00, 0.00, 1.00, 0.035),      # 顶栏：地图名/剩余时间/老鼠数/萨满
        (0.00, 0.665, 0.60, 1.00),      # 聊天框（左下，半透明但字是白的，会污染标签检测）
        (0.60, 0.665, 1.00, 1.00),      # 物品栏 + 积分板 + 底部按钮（右下）
    )

    def _in_ui(self, cx, cy, w, h):
        for x0, y0, x1, y1 in self.UI_RECTS:
            if x0 * w <= cx <= x1 * w and y0 * h <= cy <= y1 * h:
                return True
        return False

    def detect_mouse_labels(self, bgr):
        """找老鼠头顶的名字标签。返回 [(label_cx, label_bottom, (x, y, w, h))]。

        判据是「白色像素 + 紧贴的一圈深色描边 + 扁长条形状 + 不在 UI 区域」。
        偏宽松是故意的：多检出几个候选不致命（世界里的白云/白花会跟着世界动，
        第 2 步选不中），漏检才致命（真的本地玩家没进候选池）。
        """
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        white = ((gray > 190) & (sat < 80)).astype(np.uint8) * 255
        # 横向闭合，把同一行的字符合成一条
        lines = cv2.morphologyEx(white, cv2.MORPH_CLOSE, self._k_line)
        cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if not (7 <= ch <= 26 and 18 <= cw <= 420):
                continue
            if self._in_ui(x + cw // 2, y + ch // 2, w, h):
                continue
            fill = float((white[y:y + ch, x:x + cw] > 0).mean())
            if not (0.08 <= fill <= 0.80):
                continue
            pad = 2
            y0, y1 = max(0, y - pad), min(h, y + ch + pad)
            x0, x1 = max(0, x - pad), min(w, x + cw + pad)
            wroi = white[y0:y1, x0:x1]
            ring = (cv2.dilate(wroi, self._k3) > 0) & ~(wroi > 0)
            if ring.any() and float((gray[y0:y1, x0:x1][ring] < 130).mean()) >= 0.15:
                out.append((x + cw // 2, y + ch, (x, y, cw, ch)))
        return out

    @staticmethod
    def own_label_mask(bgr):
        """自己名字标签的掩膜：**金色文字 = 普通老鼠，青色文字 = 自己当萨满**。

        本地玩家的名字会**随身份换色**（2026-09-19 截图逐像素核对，_zz_crop_3.png）：
        当普通老鼠时是暖金 BGR(206,243,255)（和白色其实只差一点点）；当上萨满之后整
        行变成青色（文字核心 RGB(0,248,248)，实测纯色像素 325 个），旁边多一个白圈
        等级徽章和一颗奶酪图标。

        只认金色的话，**萨满那几局永远找不到自己的标签**——跟踪库里的掩膜锚点全废、
        cov 恒为 0，分数被一路压到 0.45，锁撑不过几秒。2~3 只老鼠的小房间里这种局
        占一多半（记忆里 project_shaman_mode 那条也早就记过：小房间要么全空必是萨满、
        要么约一半回合是萨满），所以这不是边缘情况，是主路径。

        金色判据（r>240/g>220/b>165/r-b>=25）是**对着真标签色 BGR(206,243,255) 卡的**，
        见下面代码里的实测数字：白名字（r-b=1）和淡粉美术（g=214）都进不来。顺带提醒，
        别人的**金**名字还是会命中——没关系：「这个候选是不是自己」由按键探针（跟着
        按键动的才是自己）判，比颜色硬得多。青色同理也会命中别的萨满的名字，同样交给探针。
        """
        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        r = bgr[:, :, 2].astype(np.int16)
        # ⚠ 判据收紧过一次（2026-10-07 当晚实测）：原来只是 r>240/g>205/b>150 的方框，
        # 而**客户端自己的标题栏/菜单栏是一条 1008x49 的近白**（BGR 242,246,243，38803 px）
        # 也能过——整帧「金像素」40258 个里 96% 是它；地图上那层淡粉美术（BGR 154,214,254）
        # 的亮边也成片地过（一帧搜索窗里量到 8172 px，占 7.6%，标签框压上去「杂」就 0.69
        # > LABEL_MASK_EXTRA_MAX，真标签反而被当成金斑整条丢掉）。
        # 真名字标签的像素是**固定色 BGR(206,243,255)**（2026-10-07 又一次逐像素核对：
        # 全帧金连通块里属于标签的那 4 块 (577,343)(585,343)(630,344)(638,344) 中位色全是
        # 它；docstring 里 2026-09-19 记的也是这个值，两次独立测量一致）。
        # 所以用「暖 + 亮」把标签和白/粉分开：白 r-b=1、淡粉 g=214，标签 r-b=49 / g=243。
        # 模板和窗口**用同一个掩膜**，收紧对召回率没有影响（分子分母一起缩）。
        gold = (r > 240) & (g > 220) & (b > 165) & ((r - b) >= 25)
        cyan = (b > 150) & (g > 170) & (r < 140)
        # 2026-10-26：**上面两条在本机当前客户端上一条都不命中**。当场抓帧逐像素量的：
        # 自己的标签 "Dfbaa/小老鼠" 是**蓝白 (B255,G243,R206)**（r-b = **-49**，
        # 跟「暖金 r-b>=25」正好反着），别人的名字反而是暖金 (B151,G225,R234) 和纯白。
        # 也就是说：颜色随客户端版本/称号变，任何写死的色相阈值迟早失效——而它失效的表现
        # 正是这一轮的现象：「金色名字标签 4 个候选全是掩膜太稀」、标定反复不过、锁撑不过
        # 几秒，71%~83% 的时间没锁。
        # 改成**亮度**判据（名字/称号文字一律是亮的、气泡底是暗的，实测框内底色
        # BGR(22,45,64)）：与色相无关，白/金/青/蓝白全收。代价是**所有玩家**的
        # 名字都成候选——本来就该如此，哪个是自己交给按键探针判（见 _identify_by_label）。
        bright = (b >= LABEL_TEXT_V_MIN) | (g >= LABEL_TEXT_V_MIN) | (r >= LABEL_TEXT_V_MIN)
        return gold | cyan | bright

    @staticmethod
    def gold_mask(bgr):
        """历史名字，等价于 own_label_mask（顶栏萨满名那种青色也算）。"""
        return Vision.own_label_mask(bgr)

    @staticmethod
    def dominant_text_colors(patch, k=2, min_px=12):
        """从一小块像素里取「文字色」：最亮的几簇颜色（BGR float）。

        为什么不能写死颜色：本机客户端上自己的名字是蓝白 BGR(255,243,206)、别人的名字
        是暖金 BGR(151,225,234)、称号又各不一样，换版本/换身份还会再变。建模板的那一刻
        从**自己的标签**上量，比任何常量都准（2026-10-26 实测：采样色建出来的掩膜，
        三帧里召回 1.00 / 杂 0.00，见 _probe_A/B/C）。
        """
        p = patch.reshape(-1, 3).astype(np.int16)
        v = p.max(axis=1)
        br = p[v >= LABEL_TEXT_V_MIN]
        if len(br) < min_px:
            return []
        q = br // 24                       # 量化后聚类，避免抗锯齿把一簇拆成几十种色
        keys, counts = np.unique(q, axis=0, return_counts=True)
        order = np.argsort(-counts)
        out = []
        for i in order[:k]:
            if counts[i] < min_px:
                continue
            sel = np.all(q == keys[i], axis=1)
            out.append(tuple(float(x) for x in br[sel].mean(axis=0)))
        return out

    @staticmethod
    def color_mask(bgr, colors, tol=LABEL_COLOR_TOL):
        """「离采到的文字色 ≤ tol」的像素掩膜（float32 0/1）。"""
        m = np.zeros(bgr.shape[:2], np.float32)
        if not colors:
            return m
        b = bgr.astype(np.int16)
        for c in colors:
            d = np.abs(b - np.array(c, np.int16)).max(axis=2)
            m[d <= tol] = 1.0
        return m

    def text_mask(self, bgr):
        """跟踪用的标签掩膜：一律用**亮度**判据（own_label_mask）。

        ⚠ 2026-10-26 21:52 实测推翻了「采样色精确匹配」：同一个种子模板（从 _v_B.png
        采的 (206,243,255)）在它自己那帧上 1.000，换到**同一局另一张**帧上掉到 **0.26**
        —— 名字文字的像素色随底图/抗锯齿在变，精确色掩膜只在采样那一帧成立。
        而亮度掩膜在同一张帧上是 **0.948**（位置正好压在 Dfbaa 的标签上）。
        精确色不可用还带来一连串副作用：`_tpl_score_on` 量旧模板永远不到 0.92 →
        「好模板别被顶掉」那道闸门形同虚设 → 每次按键标定都把好模板换成一块 38x9 的碎字，
        越换越差。掩膜口径必须是**跨帧稳定**的，亮度这条才是。
        """
        return self.gold_mask(bgr)

    def detect_own_labels(self, bgr):
        """找**自己**的名字标签（金色文字行）。返回 [(x, y, w, h)]，按从上到下排序。

        和 detect_mouse_labels 的区别除了颜色，还有**不做 UI 区域排除**：实测玩家站在
        画面左下角时（那正是聊天框的地盘，UI_RECTS 覆盖了 y>42% 的左半屏），标签照样
        画在聊天框之上、照样跟着按键动，被当成 UI 剔掉就等于这个位置永远标定不了
        （_k2_0_before.png 就是这么漏的）。这里的假阳性不靠位置排除，靠「按方向键看它
        动不动」的探针兜底——那是比任何区域规则都硬的判据。

        返回的是文字行的紧致框：名字一行约 35x10，紧跟着下面 14px 是称号一行 28x9，
        两行都会作为候选一起返回，选哪个都行（身体都在它们下方）。
        **萨满那几行两行会被闭运算粘成一块**（实测 73x28：青色的「Dfbaa」+「小老鼠」
        行距比金字的窄），所以高度上限放到 34 才收得到它——原来卡在 26 的那版会把
        萨满标签整个滤掉，等于又退回「萨满局找不到自己名字」。

        唯一的一条例外是**舞台矩形之外一律不要**（见 stage_rect）：那不是游戏内容，
        跟踪也夹在舞台内，选那儿当锚点必然 4 次探针全 0.45、整次标定白跑。
        """
        h, w = bgr.shape[:2]
        sx0, sy0, sx1, sy1 = self.stage_rect(bgr)
        m = self.gold_mask(bgr).astype(np.uint8) * 255
        lines = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._k_line)
        cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if not (20 <= cw <= 240 and 7 <= ch <= 34):
                continue
            if cw / max(ch, 1) < 2.0:
                continue
            if y < 26 or y > h * 0.75:      # 顶栏（计时/地图名）和最底下的物品栏
                continue
            cx, cy = x + cw * 0.5, y + ch * 0.5
            if not (sx0 <= cx <= sx1 and sy0 <= cy <= sy1):
                # 舞台外那条客户端底色带 / 窗口标题栏里也可能有金色大字（实测标题栏
                # 就是一整块 1098x30 的金斑）。那些锚点永远匹配不上（跟踪已夹进舞台），
                # 选它当标签就会让 4 次验证探针全部 0.45 → 整次标定白跑。
                continue
            out.append((x, y, cw, ch))
        out.sort(key=lambda b: (b[1], b[0]))
        return out

    def match_translation(self, a_gray, b_gray, search=80, patch=48, grid=18):
        """把 A 帧网格上的每块 patch 在 B 帧内搜索匹配，返回帧间位移。

        返回 [(ax, ay, dx, dy, score)]，ax/ay 是 patch 在 A 帧里的左上角坐标。
        静止的物体 dx=dy=0；跟着按键移动的物体会有明显位移。
        search 必须大于按住方向键期间角色在屏幕上的位移，否则匹配会撞到窗口
        边缘（这种被截断的匹配直接丢弃，见下面的边界判断）。
        """
        h, w = a_gray.shape
        top = int(h * self.cfg["play_area_top"])
        bottom = int(h * self.cfg["play_area_bottom"])
        out = []
        lo_y = max(top, search)
        hi_y = min(bottom, h - search - patch)
        lo_x = search + 20
        hi_x = w - search - patch - 20
        for ay in range(lo_y, hi_y, grid):
            for ax in range(lo_x, hi_x, grid):
                p = a_gray[ay:ay + patch, ax:ax + patch]
                if p.std() < 12:          # 平坦区域（天空/纯色）没有可匹配的特征
                    continue
                win = b_gray[ay - search:ay + search + patch,
                             ax - search:ax + search + patch]
                r = cv2.matchTemplate(win, p, cv2.TM_CCOEFF_NORMED)
                _, mx, _, ml = cv2.minMaxLoc(r)
                if mx < 0.7:
                    continue
                dx, dy = ml[0] - search, ml[1] - search
                if abs(dx) >= search - 3 or abs(dy) >= search - 3:
                    continue              # 撞到搜索窗边缘：真实位移可能更大，不可信
                out.append((ax, ay, dx, dy, float(mx)))
        return out

    def world_shift(self, a_gray, b_gray, search=30, patch=48, grid=54, min_score=0.75):
        """两帧之间**背景整体**平移了多少，返回 (dx, dy, n)。正 dx = 内容向右移。

        和 match_translation 的唯一区别是目的：这里只要一个标量，所以采样稀疏得多
        （grid 54 vs 18、search 30 vs 110），单帧开销约为它的 1/50，可以每帧都跑。
        它回答的是一个很具体的问题——**相机有没有在跟角色滚、世界到底动没动**。

        取中位数：背景块在数量上占绝对多数，玩家（屏幕固定或独自移动）是少数派，
        带不偏中位数。n 是采到的可信块数，太小说明画面太素（纯天空），调用方应把
        这次结果当「未知」而不是「没动」。
        """
        h, w = a_gray.shape
        top = int(h * self.cfg["play_area_top"])
        bottom = int(h * self.cfg["play_area_bottom"])
        lo_y = max(top, search)
        hi_y = min(bottom, h - search - patch)
        lo_x = search
        hi_x = w - search - patch
        if hi_y <= lo_y or hi_x <= lo_x:
            return 0, 0, 0
        ds, es = [], []
        for ay in range(lo_y, hi_y, grid):
            for ax in range(lo_x, hi_x, grid):
                p = a_gray[ay:ay + patch, ax:ax + patch]
                if p.std() < 12:          # 平坦区域（天空/纯色）没有可匹配的特征
                    continue
                win = b_gray[ay - search:ay + search + patch,
                             ax - search:ax + search + patch]
                if win.shape[0] < patch or win.shape[1] < patch:
                    continue
                r = cv2.matchTemplate(win, p, cv2.TM_CCOEFF_NORMED)
                _, mx, _, ml = cv2.minMaxLoc(r)
                if mx < min_score:
                    continue
                ds.append(ml[0] - search)
                es.append(ml[1] - search)
        if not ds:
            return 0, 0, 0
        ds.sort()
        es.sort()
        return int(ds[len(ds) // 2]), int(es[len(es) // 2]), len(ds)

    def set_player_lock(self, gray, cx, cy, size=48, bgr=None, tag=None):
        """把 (cx, cy) 附近的像素块锁为本地玩家模板，之后逐帧模板跟踪。

        平坦区域（纯天空）不能当模板：TM_CCOEFF_NORMED 对常量/近常量模板是退化情形，
        实测一块 std=2.15 的天空块在 60px 外也能拿到 1.0000 —— 表现就是「score 恒等于
        1.00、位置乱跳、pos 冻死」。这种位置直接拒绝，让调用方换下一个候选。
        """
        h, w = gray.shape
        x = int(round(cx - size / 2.0))
        y = int(round(cy - size / 2.0))
        x = max(0, min(x, w - size))
        y = max(0, min(y, h - size))
        if gray[y:y + size, x:x + size].std() < 12:
            return False
        self._lock_patch = gray[y:y + size, x:x + size].copy()
        self._lock_shape = gray.shape     # 帧尺寸变了（换地图/最大化）这套坐标就作废
        # 多锚点模板库。每个条目是 {tpl, ox, oy, anchor}：(ox, oy) 把该模板的匹配左上角
        # 换算成**身体 bbox 的左上角**——身体和名字标签的尺寸/位置都不一样，必须有这个
        # 偏移才能放在同一个库里比分数。
        #
        # 锚点（真身 + 镜像 + 名字标签）**永不淘汰**：老鼠转回站姿时还能靠锚点高分匹配，
        # 避免「一路学会跳跃姿势、再也认不出站姿」的单向漂移。镜像直接当独立模板放进库，
        # 取代原来「正向失配才试镜像」的特判——主模板不翻转，也就没有 2 帧极限环。
        self._lock_bank = [
            {"tpl": self._lock_patch.copy(), "ox": 0, "oy": 0, "anchor": True},
            {"tpl": cv2.flip(self._lock_patch, 1).copy(), "ox": 0, "oy": 0, "anchor": True},
        ]
        if tag is not None:
            tx, ty, tw, th = tag
            bad = (abs(x - tx) > TAG_MAX_OFFSET or abs(y - ty) > TAG_MAX_OFFSET
                   or tw > TAG_MAX_W or th > TAG_MAX_H)
            if bad:
                # **整把锁作废**，不只是「这条不进库」。尺寸/偏移离谱说明这压根不是一块
                # 名字标签（28 人的房间里几行名字并成一条就会这样），拿这种块当锚点，
                # 跟踪必然飘走——宁可这一把锁不要，让调用方换个候选/等下一帧。
                print(f"[LOCK] 标签 {tag} 偏移/尺寸离谱（身体 ({x},{y})，上限 "
                      f"±{TAG_MAX_OFFSET}px / {TAG_MAX_W}x{TAG_MAX_H}）→ 整把锁作废；"
                      f"这种块多半是几行名字并出来的，当不了锚点", flush=True)
                self.clear_player_lock()
                return False
            elif (0 <= tx and 0 <= ty and tx + tw <= w and ty + th <= h
                    and gray[ty:ty + th, tx:tx + tw].std() >= 12):
                self._lock_bank.append({"tpl": gray[ty:ty + th, tx:tx + tw].copy(),
                                        "ox": x - tx, "oy": y - ty, "anchor": True})
            # 同一块文字再存一个**掩膜**版本。名字标签底下是半透明气泡，气泡后面
            # 的地形/天空随角色移动一直变，所以原图像素匹配实测只有 0.32~0.75
            # （2026-09-19 实跑量到 0.56），拿它当验证判据必然被 min_score=0.70 否掉；
            # 掩膜匹配恒为 1.00。模板库里真正管用的就是这块静态文字，所以它必须用
            # 掩膜口径——这也正是「金色名字标签法」能锁住的原因。
            if (not bad and bgr is not None
                    and 0 <= tx and 0 <= ty and tx + tw <= w and ty + th <= h):
                # 2026-10-26：模板不再走「写死的金色阈值」，改成**从这块标签上采色**，
                # 再按采到的颜色建掩膜。实测（_probe_A/B/C 三帧）：召回 1.00、杂 0.00，
                # 位移轨迹 +28 / -15 与按键完全一致；而金色阈值在这块标签上是 0 像素。
                patch = bgr[ty:ty + th, tx:tx + tw]
                cols_ = self.dominant_text_colors(patch)     # 只作记录，不再参与掩膜
                # 掩膜与跟踪侧必须同一口径（都是亮度），否则库里的掩膜条目和每帧喂进来的
                # 窗口掩膜不是一回事，分数没有意义。见 text_mask。
                gm = self.gold_mask(bgr)[ty:ty + th, tx:tx + tw].astype(np.float32)
                if float(gm.sum()) >= 40:      # 文字像素太少，匹配分没有意义
                    # 2026-10-26 23:4x：**已经种好的字形在内存里也不再被顶掉。**
                    # 磁盘那份冻结之后，这个问题转到了内存：按键标定挑中的那块常常不是
                    # 名字行，实测 22:55 被换成 51x9、22:58 又被换成 28x9 的称号行
                    # （CCOEFF 0.49~0.69），之后每次重挂都歪 20~35px。
                    # 只有「手上根本没有模板」时才接收标定结果；有就一律不动，
                    # 想重新种 → 删掉 _label_tpl.npz 再跑 _seed_label_1026.py。
                    self._label_tpl_prev = (self._label_tpl_m,
                                            self._label_body_off,
                                            self._label_body_size,
                                            self._label_colors)
                    if self._label_tpl_m is None:
                        self._label_colors = cols_
                        self._label_tpl_m = gm.astype(np.float32)
                        self._label_body_off = (x - tx, y - ty)
                        self._label_body_size = size
                    # ⚠ **不在这里存盘**：set_player_lock 也被「名字标签重挂」调用，
                    # 而重挂可能是错的（CCOEFF 偶尔会挑中别人的名字）。存盘只认
                    # 「按键标定 + 反向验证通过」那一次（见 save_label_tpl 的调用点）
                    # ——否则一次错的重挂会把自己写成模板，下次开机继续错，越滚越偏。
                    self._lock_bank.append({"tpl_m": gm, "ox": x - tx, "oy": y - ty,
                                            "anchor": True, "mask": True,
                                            "colors": cols_})
            # 标定给出的标签宽高是**活标签的实际像素尺寸**，而字形是按别的缩放存的。
            # 两者一比就是地图缩放比（41/52 ≈ 0.79x），当场校正，别等下一次扫描。
            self._label_zoom_from_tag(tag, bgr)
        self._weak_frames = 0
        self._weak_cand = None
        self._cb_area_min = None     # 刚锁上：还不知道身上「本底」多大一块绿
        # 同位置的彩色模板，只在「首次失配」时跟着存盘：灰度图看不出模板里到底是老鼠、
        # 名字标签还是背景（灰度下白色文字和浅色帽子长得一样，前几轮就是这么猜错的）。
        self._lock_bgr = bgr[y:y + size, x:x + size].copy() if bgr is not None else None
        self._lock_bbox = (x, y, size, size)
        # 建库时把每块模板的偏移打出来。名字标签条目的偏移 = 身体左上角 − 标签左上角，
        # 而搜索窗是按**所有**模板 ✕ search 撑开的，所以偏移一旦离谱（标定给错标签、
        # 或者 tag 和 body 不配套），窗就会撑到整幅那么大——窗一大，灰度模板在哪儿都能
        # 凑出高分、掩膜也总能找到金像素，锁立刻被拉到无关位置。2026-09-19 run12 实测
        # 出现过 852x453 的窗（帧 1106x787，身体框只有 48x48），靠这一行才能定位是谁。
        print(f"[LOCK] 建库 {len(self._lock_bank)} 块 身体({x},{y},{size}) "
              f"标签{tag} 偏移{[(e['ox'], e['oy'], _entry_hw(e)[1], _entry_hw(e)[0], bool(e.get('mask'))) for e in self._lock_bank]}")
        self._lock_lost = 0
        self._lock_lost_since = 0.0
        self._lock_dbg_done = False
        self._lock_win_warned = False
        self._noted_label_less = False
        self._nolabel_keep = 0
        self._nolabel_run = 0
        self._last_player = (x + size // 2, y + size // 2, y + size, (x, y, size, size))
        self._player_lost_frames = 0
        return True

    def _lock_dump(self, win, off, ml, bank_n, r=None, suffix="", box=None, win_g=None):
        """锁定后**第一次**失配时把模板和搜索窗存盘，供人工核对到底跟丢了什么。

        没有这个就只能对着 score 曲线猜（前几轮猜错过好几次）；有图能直接看出模板
        是不是老鼠、老鼠在窗内哪个位置、是不是被 HUD 或别的老鼠挡住、是不是向下掉了。
        灰度图里「白色名字文字」和「浅色帽子」长得一样，所以彩色模板/窗口也一并存，
        并把匹配图的前 5 个峰打出来——能看出真老鼠位置压根没进候选，还是被别的东西压过。
        每次锁定只存一次，不会刷屏。
        """
        d = os.path.dirname(os.path.abspath(__file__))
        bh, bw = self._lock_patch.shape[:2]
        x0, y0 = off
        vis = cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, ml, (ml[0] + bw, ml[1] + bh), (0, 0, 255), 2)   # 正向最佳匹配
        save_png(os.path.join(d, f"_lost_window{suffix}.png"),
                    cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        save_png(os.path.join(d, f"_lost_template{suffix}.png"),
                    cv2.resize(self._lock_patch, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        # 名字标签的**掩膜**也要存：2026-10-07 空房那场所有线索都指向它（召回 0.44、
        # 杂 0.16 —— 不是金斑误配，是真对不上），但「锁定时的掩膜」和「失配时的掩膜」
        # 一张都没留下来，只能对着 0.44 这个数猜是字形变了还是位移。两张并排放一起，
        # 一眼就能看出是行距变了、少了一行、还是笔画粗细变了。
        # 灰掩膜里 0=无金像素、255=有，直接放大 4 倍存盘。
        e_lab = self._label_entry()
        if e_lab is not None:
            tpm = (e_lab["tpl_m"] > 0).astype(np.uint8) * 255
            save_png(os.path.join(d, f"_lost_labmask_tpl{suffix}.png"),
                        cv2.resize(tpm, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        if win_g is not None:
            wm = (np.asarray(win_g) > 0).astype(np.uint8) * 255
            save_png(os.path.join(d, f"_lost_labmask_win{suffix}.png"),
                        cv2.resize(wm, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        if getattr(self, "_last_bgr", None) is not None:
            cb = self._last_bgr[y0:y0 + win.shape[0], x0:x0 + win.shape[1]]
            if cb.shape[:2] == win.shape[:2]:
                cvis = cb.copy()
                cv2.rectangle(cvis, ml, (ml[0] + bw, ml[1] + bh), (0, 0, 255), 2)
                save_png(os.path.join(d, f"_lost_window_c{suffix}.png"),
                            cv2.resize(cvis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
            if self._lock_bgr is not None:
                save_png(os.path.join(d, f"_lost_template_c{suffix}.png"),
                            cv2.resize(self._lock_bgr, None, fx=4, fy=4,
                                       interpolation=cv2.INTER_NEAREST))
        peaks = ""
        if r is not None:
            rr = r.copy()
            top = []
            for _ in range(5):
                _, mxv, _, mlv = cv2.minMaxLoc(rr)
                top.append((mlv, round(float(mxv), 2)))
                x_, y_ = mlv
                rr[max(0, y_ - 8):y_ + 9, max(0, x_ - 8):x_ + 9] = -1
            peaks = f" 匹配图前5峰 {top}"
        print(f"[LOCK] {('最终失配' if suffix else '首次失配')}已存图：模板 {bw}x{bh} / "
              f"窗 {win.shape[1]}x{win.shape[0]} 窗原点({x0},{y0})"
              f" 身体框{box}"
              f"（最佳 {ml} score={self._last_score:.2f}"
              f" 金标签召回{getattr(self, '_last_best_mask', -1):.2f}"
              f"(杂{getattr(self, '_last_best_mask_extra', -1):.2f})"
              f" 灰度{getattr(self, '_last_best_gray', -1):.2f}"
              f" 标签召回{getattr(self, '_last_cov_any', -1):.2f}/"
              f"{getattr(self, '_last_cov_exp', -1):.2f}"
              f" | 赢家库#{getattr(self, '_last_win_i', -1)}"
              f"灰{getattr(self, '_last_win_gray', -1):.2f}"
              f"cov{getattr(self, '_last_win_cov', -1):.2f}"
              f"d{getattr(self, '_last_win_dist', -1)}"
              f" | 灰最优{getattr(self, '_last_bg_pos', None)}"
              f"cov{getattr(self, '_last_bg_cov', -1):.2f}"
              f"d{getattr(self, '_last_bg_dist', -1)}"
              # 无标签豁免预算：撑到 NOLABEL_KEEP_MAX 就说明锁是靠「窗里没有金子」硬撑的
              f" 豁免{getattr(self, '_last_nolabel_run', -1)}/{NOLABEL_KEEP_MAX}"
              f" 模板库{bank_n}块）{peaks}")

    def clear_player_lock(self):
        self._lock_patch = None
        self._lock_bbox = None
        self._lock_shape = None
        self._lock_lost = 0
        self._lock_lost_since = 0.0
        self._lock_bank = []
        self._weak_frames = 0
        self._weak_cand = None
        self._cb_area_min = None     # 新的一把锁 = 重新学「身上本来多大一块绿」
        # 采到的标签颜色**不随锁一起清掉**：它描述的是「这台客户端上我自己的名字长什么样」，
        # 跟这一把锁无关。跟丢后重新标定时还能拿它当掩膜兜底（text_mask）。
        # 只有在 set_player_lock 采到新色时才覆盖。

    def _label_tpl_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_label_tpl.npz")

    @staticmethod
    def label_tpl_sane(tpl, off, colors):
        """这块字形值不值得当「我的名字」用。

        标定偶尔会把别人的名字/顶栏文字认成自己的（实测 20:57 那次并出来的标签
        双向跟手 +52 / **-110**，位移都不对称），那种模板存下来只会让重挂指到
        画面别处。这里只卡**几何**，卡不住的交给 CCOEFF 的分数和次峰差。
        实测好模板：28~43 宽 / 9~10 高，身体在标签**下方** 11~18px。
        """
        if tpl is None or off is None or not colors:
            return False
        th, tw = tpl.shape
        ox, oy = off
        return (6 <= th <= 16 and 18 <= tw <= 90
                and -40 <= ox <= 40 and 4 <= oy <= 45
                and float(tpl.sum()) >= 40)

    def _tpl_score_on(self, bgr, tpl, cols=None):
        """这块字形在当前帧上的 CCOEFF 峰值（整幅舞台搜）。用来判断「谁更好用」。

        `cols` 已不参与（见 text_mask 的说明：精确色掩膜跨帧不稳），保留参数只为兼容
        调用点。
        """
        if tpl is None or bgr is None:
            return 0.0
        sx0, sy0, sx1, sy1 = self.stage_rect(bgr)
        win = self.gold_mask(bgr)[sy0:sy1, sx0:sx1].astype(np.float32)
        th, tw = tpl.shape
        if win.shape[0] < th or win.shape[1] < tw:
            return 0.0
        return float(cv2.minMaxLoc(
            cv2.matchTemplate(win, tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED))[1])

    def save_label_tpl(self, bgr=None):
        """把名字字形存盘。**为什么跨进程存**：这台客户端的名字是固定文字（同字体
        同字号，只有位置会变），存下来下次启动第一帧就能重挂，不用再按方向键标定
        一遍。存不下也不影响运行，只是下次还得用按键标定。

        `bgr` 用来**比新旧两版字形谁好用**：标定挑中的那一行未必是名字行，实测挑中过
        28x9 的称号「小老鼠」，CCOEFF 只有 0.49~0.69（名字行 35x10 是 1.00）。旧模板
        在这帧上还打得 ≥0.92 就说明它没问题，别拿难用的顶掉好用的。
        """
        if self._label_tpl_m is None or self._label_body_off is None:
            return
        # 2026-10-26 23:2x：**磁盘上的字形一旦存在就不再改写**。
        # 试过「同一帧比新旧谁高」，仍会被按键标定顶掉：人多的时候自己的标签被压住，
        # 好模板那一帧打不高，而标定刚挑中的那块在**它自己那帧**永远 1.00 —— 比不过。
        # 实测被换成了偏移 (-15,15)/(-12,16) 的版本（指向名字下面那行称号），
        # 重挂出来的身体框歪 20~35px，跟踪跟着歪。
        # 这个文件是「标准答案」，只由 _seed_label_1026.py 从干净截帧重新种；
        # 想换就删掉文件再种一次（README 级别的操作，不要在这里自动改）。
        try:
            if os.path.exists(self._label_tpl_path()) and \
                    os.path.getsize(self._label_tpl_path()) > 0:
                return
        except Exception:
            pass
        prev = getattr(self, "_label_tpl_prev", None)
        if bgr is not None and prev is not None:
            old_tpl, old_off, old_size, old_cols = prev
            s_old = self._tpl_score_on(bgr, old_tpl, old_cols)
            s_new = self._tpl_score_on(bgr, self._label_tpl_m, self._label_colors)
            # 2026-10-26 23:1x：原来只判「旧模板这帧还打得 ≥0.92」就保住旧的——
            # 可人多的时候自己的标签被别的老鼠压住，旧模板那一帧**打不高**，闸门就形同虚设，
            # 22:27 那次于是把 35x10 的好字形换成了 28x9 的称号行（偏移 (-15,15)，
            # 指向的是名字下面那行「小老鼠」），之后重挂分数一路掉到 0.4~0.6。
            # 改成**同一帧上比新旧谁高**：新的要 ≥0.95 且比旧的高 0.05 才准顶掉。
            if old_off is not None and (s_old >= LABEL_TPL_KEEP
                                        or s_new < 0.95 or s_new < s_old + 0.05):
                self._label_tpl_m = old_tpl
                self._label_body_off = old_off
                self._label_body_size = old_size
                self._label_colors = old_cols
                print(f"[LOCK] 这次标定挑中的标签块不如上次的字形好用"
                      f"（旧 {s_old:.2f} / 新 {s_new:.2f}，同一帧）→ 继续用旧的，不覆盖")
                return
        if not self.label_tpl_sane(self._label_tpl_m, self._label_body_off,
                                   self._label_colors):
            # 几何就不像「头顶上的名字」→ 不存。宁可下次多标定一次，也别让一块
            # 坏模板在进程之间传播（实测坏模板会让重挂指到聊天栏/顶栏）。
            return
        try:
            cols = np.array(self._label_colors or [], np.float32).reshape(-1, 3)
            # ⚠ 存盘空间的约定：字形/偏移/身体框一律存成**只含舞台缩放、不含地图缩放**
            # 的几何（`_label_scale_raw` 的口径）。标定是在当前这张图上量的，而当前图
            # 可能带着 0.81x 的地图缩放，直接存会把那层缩放焊死进「标准答案」——
            # 下次换回 1.00x 的图，模板就大了一圈，永远对不上。所以存之前先按当前的
            # 地图缩放除回去。（zoom=1.00 时这两个分支完全等价，不影响既有文件。）
            _z = float(self._label_zoom or 1.0)
            tpl_s = self._label_tpl_m
            off_s = self._label_body_off
            size_s = int(self._label_body_size or 48)
            scale_s = float(self._label_stage_scale or 1.0)
            if abs(_z - 1.0) > 0.01:
                th_, tw_ = tpl_s.shape
                tpl_s = cv2.resize(tpl_s, (max(4, int(round(tw_ / _z))),
                                           max(3, int(round(th_ / _z)))),
                                   interpolation=cv2.INTER_NEAREST)
                off_s = (int(round(off_s[0] / _z)), int(round(off_s[1] / _z)))
                size_s = int(round(size_s / _z))
            with open(self._label_tpl_path(), "wb") as f:
                np.savez(f, tpl=tpl_s,
                         off=np.array(off_s, np.int32),
                         size=np.array([int(size_s)], np.int32),
                         cols=cols,
                         # 存的是**舞台缩放比**（不含地图缩放），读回来要按它换算
                         scale=np.array([scale_s], np.float32))
        except Exception as e:
            print(f"[LOCK] 名字字形存盘失败（不影响运行）：{e}")

    def _load_label_tpl(self):
        """读回上次存的名字字形 + **采到的文字色**。

        ⚠ 文字色必须一起存：`text_mask` 在没有采色时会退回写死的金色阈值，而本机
        客户端上自己的名字是蓝白的，金色阈值**一个像素都命不中**——只存字形不存色，
        重挂时算出来的掩膜整幅为 0，等于什么都没存。

        ⚠ 2026-10-27：连 **scale**（存盘时舞台相对 800x600 的倍数）一起存。
        旧文件没有这个字段，按 1.0 算。读进来先按原生几何过 sane，再交给
        `_apply_label_scale` 换算到当前帧的尺寸——见那里的说明。
        """
        p = self._label_tpl_path()
        if not os.path.exists(p):
            return
        try:
            with np.load(p) as z:
                tpl = z["tpl"].astype(np.float32)
                off = tuple(int(v) for v in z["off"])
                size = int(z["size"][0])
                cols = [tuple(float(x) for x in c) for c in z["cols"]] \
                    if "cols" in z.files else []
                scale = float(z["scale"][0]) if "scale" in z.files else 1.0
        except Exception:
            return
        if not (0.5 <= scale <= 4.0):
            scale = 1.0
        # sane 的边界（oy 4~45、宽 18~90）是按 **1x 舞台**量的，带缩放的模板要
        # 先把几何折回 1x 再判，否则 1.5x 的 oy=53 会被当成坏模板丢掉。
        if tpl.ndim != 2 or not self.label_tpl_sane(
                tpl, (off[0] / scale, off[1] / scale), cols):
            print("[LOCK] 上次存的名字字形不可信（尺寸/偏移不像头顶标签）→ 不用它")
            return
        self._label_tpl_raw = tpl
        self._label_off_raw = off
        self._label_size_raw = size
        self._label_scale_raw = scale
        self._label_colors = cols
        self._label_scale_cur = None
        self._apply_label_scale(scale)
        print(f"[LOCK] 载入上次的名字字形 {tpl.shape[1]}x{tpl.shape[0]}"
              f"（原生 {scale:.2f}x，偏移 {off}，身体 {size}px，"
              f"文字色 {[(round(c[0]),round(c[1]),round(c[2])) for c in cols]}）"
              f"——开局即可重挂")

    def _apply_label_scale(self, s, quiet=False):
        """把存盘的字形 / 偏移 / 身体框边长换算到**当前**舞台缩放比。

        为什么必须有这一步：字形模板是按某一次的帧尺寸量的（这一份是在
        1024x700 的帧上量的，舞台 1x）。客户端一旦换成 150% DPI，帧变成
        1536x1050、舞台 1.5x，同一个「Dfbaa」的字形也跟着大 1.5 倍。拿 35x10 的
        旧字形去 CCOEFF 匹配 52x15 的活字形，峰值只有 **0.359**（2026-10-27
        实测），低于重挂门槛 0.70 → 不按键就永远锁不上；换成 1.5x 后同一帧
        峰值 **0.833**、位置正好在 (272,425) 的标签上。
        放大用 INTER_NEAREST：掩膜是 0/1 的，插值会把笔画糊成灰边，实测
        LINEAR 0.699 / CUBIC 0.680 都明显差于 NEAREST 0.813。
        """
        if getattr(self, "_label_tpl_raw", None) is None:
            return
        try:
            s = float(s)
        except Exception:
            s = 1.0
        if not (0.5 <= s <= 4.0):
            s = 1.0
        self._label_stage_scale = s
        # 总倍率 = 舞台缩放 × 地图缩放（见 LABEL_ZOOM_STEPS）；_label_scale_cur 记的是
        # 这个总倍率，所以换地图缩放时得先把它清掉才会重算。
        s = s * float(self._label_zoom or 1.0)
        if self._label_scale_cur is not None and abs(s - self._label_scale_cur) <= 0.02:
            return
        k = s / float(self._label_scale_raw or 1.0)
        tpl = self._label_tpl_raw
        th, tw = tpl.shape
        if abs(k - 1.0) <= 0.02:
            self._label_tpl_m = tpl
            self._label_body_off = self._label_off_raw
            self._label_body_size = self._label_size_raw
        else:
            nw = max(4, int(round(tw * k)))
            nh = max(3, int(round(th * k)))
            self._label_tpl_m = cv2.resize(tpl, (nw, nh),
                                           interpolation=cv2.INTER_NEAREST)
            ox, oy = self._label_off_raw
            self._label_body_off = (int(round(ox * k)), int(round(oy * k)))
            self._label_body_size = int(round((self._label_size_raw or 48) * k))
        self._label_scale_cur = s
        if abs(k - 1.0) > 0.02 and not quiet:
            print(f"[LOCK] 字形按舞台 {s:.2f}x 换算："
                  f"{tw}x{th} → {self._label_tpl_m.shape[1]}x{self._label_tpl_m.shape[0]}"
                  f"（偏移 {self._label_body_off}，身体 {self._label_body_size}px）")

    def _label_set_zoom(self, z, quiet=True):
        """换一个地图缩放倍率（字形/偏移/身体框一起换）。不打印，免得扫倍率时刷屏。"""
        if getattr(self, "_label_tpl_raw", None) is None:
            return
        try:
            z = float(z)
        except Exception:
            return
        if not (0.3 <= z <= 2.5) or abs(z - float(self._label_zoom or 1.0)) <= 0.01:
            return
        self._label_zoom = z
        self._label_scale_cur = None
        self._apply_label_scale(self._label_stage_scale, quiet=quiet)

    def _label_zoom_from_tag(self, tag, bgr):
        """按键标定找到的**活标签实测尺寸**，直接拿来校准地图缩放。

        2026-10-27 02:12 的现场：标定在一块 41x13 的标签上成功（这张图缩放 0.79x），
        可字形还停在 1.00x 的 52x15 上，紧跟着的 [ANCHOR] 峰只有 0.39 → 锁撑 5 秒就丢，
        一晚上就在「标定成功 → 立刻丢 → 退避 → 再标定」里打转。宽度比 41/52 就是
        缩放比，当场换算比等下一次扫描快整整一局。

        只认「换完之后**这一帧**确实打得更高」的倍率：标定偶尔会挑中称号行（实测
        51x9 / 28x9），那种倍率换上去只会更差，必须能自己退回来。
        """
        if self._label_tpl_m is None or bgr is None or tag is None:
            return
        try:
            tw_det = int(tag[2]); th_det = int(tag[3])
        except Exception:
            return
        th_cur, tw_cur = self._label_tpl_m.shape
        if tw_cur <= 0 or tw_det <= 0:
            return
        z_now = float(self._label_zoom or 1.0)
        z_new = z_now * (float(tw_det) / float(tw_cur))
        if not (0.45 <= z_new <= 2.0) or abs(z_new - z_now) <= 0.06:
            return
        m = self.text_mask(bgr)
        rect = self.stage_rect(bgr)
        r0 = self._label_match_one(bgr, m, *rect, LABEL_REACQ_MARGIN)
        s0 = r0[0] if r0 is not None else 0.0
        self._label_set_zoom(z_new)
        r1 = self._label_match_one(bgr, m, *rect, LABEL_REACQ_MARGIN)
        s1 = r1[0] if r1 is not None else 0.0
        if s1 >= max(LABEL_REACQ_SCORE, s0 + 0.05):
            print(f"[LOCK] 地图缩放按标定标签 {tw_det}x{th_det} 校正："
                  f"{z_now:.2f}x → {z_new:.2f}x（峰 {s0:.2f} → {s1:.2f}，"
                  f"字形 {self._label_tpl_m.shape[1]}x{self._label_tpl_m.shape[0]}）",
                  flush=True)
            self._label_sweep_t = time.time()   # 刚校过，这半秒别再扫
        else:
            self._label_set_zoom(z_now)
            self._apply_label_scale(self._label_stage_scale, quiet=True)

    def _label_raw(self, bgr, mask=None):
        """整幅舞台搜字形，返回 `(峰, 次峰, tx, ty)`；不能算时返回 None。"""
        if self._label_tpl_m is None or bgr is None:
            return None
        sx0, sy0, sx1, sy1 = self.stage_rect(bgr)
        m = mask if mask is not None else self.text_mask(bgr)
        win = m[sy0:sy1, sx0:sx1].astype(np.float32)
        th, tw = self._label_tpl_m.shape
        if win.shape[0] < th or win.shape[1] < tw:
            return None
        cov = cv2.matchTemplate(win, self._label_tpl_m, cv2.TM_CCOEFF_NORMED)
        _, mv, _, mlv = cv2.minMaxLoc(cov)
        sec = self._second_peak(cov, mlv, LABEL_REACQ_EXCL)
        return (float(mv), float(sec), sx0 + mlv[0], sy0 + mlv[1])

    def label_confidence(self, bgr, mask=None):
        """刚标出来的字形在整屏上「独一无二」到什么程度：(峰, 次峰)。"""
        r = self._label_raw(bgr, mask)
        return None if r is None else (r[0], r[1])

    def _label_match(self, bgr, mask=None, margin=LABEL_REACQ_MARGIN):
        """在整幅舞台里找**自己名字的字形**。返回 `(score, tx, ty)` 或 None。

        `mask` 是已经算好的全帧文字掩膜（`detect_player` 每帧本来就要算一次，
        传进来省一趟 1~2ms 的 HSV/颜色距离）。

        判分用 `TM_CCOEFF_NORMED` 而不是召回率：召回率在有文字的地方恒定 1.00
        （聊天栏、顶栏、别人的名字全是），次峰也 1.00，判据直接塌掉——2026-10-26
        实跑就是这么满屏乱挂的。CCOEFF 看的是**字形笔画本身**，实测三种帧上
        峰值 1.000 / 0.879 / 0.992，次峰 0.31~0.37。

        ⚠ 2026-10-26 23:4x 补了一道**冷暖色**闸门：字形模板是**亮度**掩膜，
        而满屏几十个玩家的名字都是亮字，形状一凑就过 0.55 的线 —— 重挂就这么
        挂到别人头上（帧 `_sq_11.png` 里挂中的是白字的 "F_a"，正是别人的名字）。
        实测自己名字是暖色（BGR 207/231/241，R−B≈+34），别人的是冷白
        （R−B≈−15~−23），差 50 上下，用「R−B ≥ 6」一刀切即可。
        撞上白名字不返回 None，而是把它从匹配图里挖掉再找下一个峰 —— 自己的
        标签常被别人压住，直接放弃就整局挂不上。
        """
        if self._label_tpl_m is None or bgr is None:
            return None
        sx0, sy0, sx1, sy1 = self.stage_rect(bgr)
        m = mask if mask is not None else self.text_mask(bgr)
        # **地图缩放**：先按上次命中的倍率试一次。分≥0.85 就是「一眼就是我」，
        # 直接返回（锁着的帧代价和不改之前一样）。低于 0.85 —— 可能是被别人的名字
        # 压住，也可能是**这张图的缩放变了、0.60 的线只是碰巧被某个假峰过了**——
        # 扫一圈挑分最高的那个倍率。实测 02:07 就是这么挂错的：缩放变了，旧倍率
        # 撞上个 0.60 出头的假峰就锁上了，跟了两分钟没离开过锚点。
        base = float(self._label_zoom or 1.0)
        self._label_set_zoom(base)
        r0 = self._label_match_one(bgr, m, sx0, sy0, sx1, sy1, margin)
        if r0 is not None and r0[0] >= LABEL_ZOOM_GOOD:
            return r0
        # 锁着时平时不扫（每帧 85ms 会把跟踪拖死，好坏交给跟踪门槛）。**例外**：已经
        # 连续失配 3 帧以上——换图后地图缩放会变（52x15 → 41x13 实测），不重扫就只能
        # 干等丢锁再重挂，白扔十几秒。
        if self._lock_patch is not None and getattr(self, "_lock_lost", 0) < 3:
            return r0
        if time.time() - self._label_sweep_t < LABEL_ZOOM_SWEEP_S:
            # 刚扫过一整圈。这时**不要拿 0.60~0.85 的候选凑数** —— 2026-10-27 02:11
            # 就是这么挂错的：真实标签是 41x13（这张图缩放 0.81x），而旧倍率在别处凑出
            # 个 0.6 出头的假峰，挂上去跟 5 秒就丢，来回折腾。宁可这一帧空着。
            # （锁着时不适用：那时候 r0 至少还在原倍率上，喂给跟踪比什么都没有强。）
            return r0 if self._lock_patch is not None else None
        self._label_sweep_t = time.time()
        best, best_z = r0, base
        for z in LABEL_ZOOM_STEPS:
            if abs(z - base) <= 0.01:
                continue
            self._label_set_zoom(z)
            r = self._label_match_one(bgr, m, sx0, sy0, sx1, sy1, margin)
            if r is not None and (best is None or r[0] > best[0] + 0.02):
                best, best_z = r, z
        self._label_set_zoom(best_z)
        if best is None or best[0] < LABEL_ZOOM_ADOPT:
            # **扫一圈也没打到 0.75 → 不认这个倍率。** 换倍率必须由**真命中**驱动：
            # 2026-10-27 02:1x 现场实测，老鼠一死，左下**聊天栏**里自己名字那几行
            # 在 0.89x 上打出 0.46~0.62 夺冠（真标签回来时在 1.00x 上打 0.93，
            # 但倍率已经被改歪，那个 1.00x 压根没被试）——结果一晚上都在聊天栏里
            # 找自己，「锁上 5 秒就丢」。倍数宁可不换，也不能被假峰改。
            self._label_set_zoom(base)
            return r0
        if abs(best_z - base) > 0.01:
            print(f"[LOCK] 地图缩放 {best_z:.2f}x 命中（字形 "
                  f"{self._label_tpl_m.shape[1]}x{self._label_tpl_m.shape[0]}，"
                  f"身体 {self._label_body_size}px，峰 {best[0]:.2f}）"
                  f"——这张图的名字比以前{'小' if best_z < 1 else '大'}"
                  f"{abs(1 - best_z) * 100:.0f}%", flush=True)
        return best

    def _label_match_one(self, bgr, m, sx0, sy0, sx1, sy1, margin):
        """在**一个固定缩放**下找名字字形（判据见 _label_match 的说明）。"""
        win = m[sy0:sy1, sx0:sx1].astype(np.float32)
        th, tw = self._label_tpl_m.shape
        if win.shape[0] < th or win.shape[1] < tw:
            return None
        cov = cv2.matchTemplate(win, self._label_tpl_m, cv2.TM_CCOEFF_NORMED)
        # 2026-10-27 00:0x：**峰间差这道闸门在人群里是错的判据，改成「暖色优先」。**
        # 原来先算 sec（全屏次峰），只要 `峰 − 次峰 < margin` 就整条 return None。
        # 在 30 人的 vanilla1 里次峰天天是 0.6~0.8（别人的白名字），于是**自己的标签
        # 明明打到 0.95 也被一票否决**——实测 `_watch_label_1027`：00:02:29 分=0.95
        # 暖=True @(144,337)，同一时刻机器人 STAT 却是 NO-LOCK 连续 4000+。
        # 真正的判据是颜色：自己的名字是暖金（R−B≈+34），别人的是冷白（R−B≈−15~−23），
        # 差 50 上下，`_label_warm` 一刀切就够。所以：
        #   ① 扫前 8 个峰，把「暖 / 冷」分开收集；
        #   ② 暖色峰里取分最高的，只要 ≥ 门槛、且**比最强的冷色峰高 0.05**（或自己 ≥0.92，
        #      高到这个份上暖色已经不可能认错）就认；
        #   ③ 一个暖色峰都没有时，退回原来的「峰−次峰」老判据（个别地图/称号会把颜色冲淡）。
        rr = cov.copy()
        warm_c, best_cold = [], 0.0
        for _ in range(8):
            _, mv, _, mlv = cv2.minMaxLoc(rr)
            if mv < LABEL_WARM_FLOOR:
                break
            tx, ty = sx0 + mlv[0], sy0 + mlv[1]
            if self._label_warm(bgr, tx, ty, tw, th):
                warm_c.append((float(mv), tx, ty))
            else:
                best_cold = max(best_cold, mv)
            rr[max(0, mlv[1] - LABEL_REACQ_EXCL):mlv[1] + LABEL_REACQ_EXCL + 1,
               max(0, mlv[0] - LABEL_REACQ_EXCL):mlv[0] + LABEL_REACQ_EXCL + 1] = -1.0
        # ① 分数够（且有暖冷对比）② 位置对得上上次的自己。两条都过才认。
        # warm_c 已经按分数降序，所以「近而分低」的会排在「远而分高」的后面 ——
        # 加上远近闸门以后，近处那个更重要，所以要遍历完再挑，不能见到第一个就回。
        ref = self._last_player
        for mv, tx, ty in warm_c:
            if mv < LABEL_REACQ_SCORE or not (mv - best_cold >= 0.05 or mv >= 0.92):
                continue
            if ref is not None:
                d = max(abs(tx + tw // 2 - ref[0]), abs(ty + th // 2 - ref[1]))
                # 离上次位置很远：只有「一眼就是我」的分才认（换图重生确实会瞬移，
                # 所以不能一刀切禁掉，只能加价）
                if d > LABEL_REACQ_NEAR and mv < LABEL_REACQ_FAR_SCORE:
                    continue
            return (mv, tx, ty)
        if not warm_c:
            # 老路：冷色都没有可比对象时按峰间距判
            _, _, _, mlv0 = cv2.minMaxLoc(cov)
            sec = self._second_peak(cov, mlv0, LABEL_REACQ_EXCL)
            _, mv, _, mlv = cv2.minMaxLoc(cov)
            if mv >= LABEL_REACQ_SCORE and mv - sec >= margin:
                return (float(mv), sx0 + mlv[0], sy0 + mlv[1])
        return None

    @staticmethod
    def _label_warm(bgr, tx, ty, tw, th, need=6.0):
        """这块字是不是**暖色**（自己的名字）。别人的名字一律冷白。"""
        h, w = bgr.shape[:2]
        if tx < 0 or ty < 0 or tx + tw > w or ty + th > h:
            return False
        sub = bgr[ty:ty + th, tx:tx + tw]
        b = sub[:, :, 0].astype(np.int16)
        r = sub[:, :, 2].astype(np.int16)
        g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        m = g >= LABEL_TEXT_V_MIN
        if int(m.sum()) < 6:
            return False
        return float((r[m] - b[m]).mean()) >= need

    def reacquire_by_label(self, bgr):
        """不按键、纯视觉地把锁重新挂回自己的名字标签上；挂不上返回 None。

        为什么要这条路：标定要按住方向键、隔两帧、再做反向投票，一次好几秒；而一局
        只有 ~2 分钟，标定失败还会退避。实测 20:49 那场：标定成功 → 4.3s 后跟丢 →
        退避 60s → 整局白跑。名字标签是**静态像素**（同字体同字号，不随角色动画变），
        一旦留下字形掩膜，「在画面上找它」比「找身体」稳得多，而且零按键、几毫秒。

        返回 `(cx, cy, size, tag)`，tag = 标签左上角 + 尺寸，原样喂给 set_player_lock。
        """
        if self._label_body_off is None:
            return None
        r = self._label_match(bgr)
        if r is None:
            return None
        _, tx, ty = r
        ox, oy = self._label_body_off
        size = self._label_body_size or 48
        bx, by = tx + ox, ty + oy
        h, w = bgr.shape[:2]
        if not (0 <= bx and 0 <= by and bx + size <= w and by + size <= h):
            return None
        th, tw = self._label_tpl_m.shape
        return (bx + size // 2, by + size // 2, size, (tx, ty, tw, th))

    def _track_by_label(self, bgr, mask=None):
        """每帧跟踪的主路径：在整幅舞台上找自己的名字字形，按固定偏移推出身体框。

        ⚠ 和 `reacquire_by_label` 的区别只是门槛和一道「别跳太远」的闸门 ——
        重挂错了下一帧还能再挂，跟踪挂错整局都歪，所以这里要 `LABEL_TRACK_SCORE`
        和 `LABEL_TRACK_JUMP` 两道。
        """
        if self._label_tpl_m is None or self._label_body_off is None or bgr is None:
            return None
        r = self._label_match(bgr, mask, margin=LABEL_TRACK_MARGIN)
        if r is None or r[0] < LABEL_TRACK_SCORE:
            return None
        _, tx, ty = r
        ox, oy = self._label_body_off
        size = self._label_body_size or 48
        bx, by = tx + ox, ty + oy
        h, w = bgr.shape[:2]
        if not (0 <= bx and 0 <= by and bx + size <= w and by + size <= h):
            return None
        cx, cy = bx + size // 2, by + size // 2
        last = self._last_player
        if last is not None:
            if max(abs(cx - last[0]), abs(cy - last[1])) > LABEL_TRACK_JUMP:
                return None
        self._lock_bbox = (bx, by, size, size)
        self._last_player = (cx, cy, by + size, (bx, by, size, size))
        self._player_lost_frames = 0
        self._nolabel_run = 0
        return self._last_player

    def has_lock(self):
        return self._lock_patch is not None

    def _bank_add(self, patch):
        """把一个新姿势（连同镜像）收进模板库；库里已经有很像的就不重复存。

        只收身体尺寸的模板（ox=oy=0）；锚点条目永不淘汰——见 set_player_lock 的说明。
        """
        if self._lock_patch is None or patch.shape != self._lock_patch.shape:
            return
        if float(patch.std()) < 12:
            return
        for e in self._lock_bank:
            t = e.get("tpl")           # 掩膜条目只有 tpl_m，形状也和身体块不同
            if t is None:
                continue
            if t.shape == patch.shape and \
                    float(cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED).max()) > 0.95:
                return
        self._lock_bank.append({"tpl": patch.copy(), "ox": 0, "oy": 0})
        self._lock_bank.append({"tpl": cv2.flip(patch, 1).copy(), "ox": 0, "oy": 0})
        while len(self._lock_bank) > BANK_CAP:
            for i, e in enumerate(self._lock_bank):
                if not e.get("anchor"):
                    del self._lock_bank[i]      # 从最旧的学来姿势开始丢
                    break
            else:
                break

    @staticmethod
    def _second_peak(r, ml, radius):
        """匹配图上「避开最佳峰一整块」后的最高峰，用来判断最佳峰是不是够突出。
        两个作用：

        1. 挡掉退化匹配——平坦块（纯天空/冰面渐变）整张匹配图都接近 1.0，
           次峰也≈1.0，差值≈0。
        2. **半径必须够大**（用 bw//2）。实测镜像模板会在真身旁边 24px 处造出一个
           near-equal 的峰（(63,64) 0.52 vs (87,64) 0.47，锁定框宽 48）——那不是「另有
           一个物体」，是同一只老鼠的对称性。半径只给 8px 的话差值只剩 0.05，弱匹配
           会被自己人挡掉，锁就丢了。"""
        rr = r.copy()
        x_, y_ = ml
        rr[max(0, y_ - radius):y_ + radius + 1,
           max(0, x_ - radius):x_ + radius + 1] = -1.0
        return float(rr.max())

    def _label_entry(self):
        """库里那块金色名字标签的掩膜条目（没有就 None）。"""
        for e in self._lock_bank:
            if e.get("mask"):
                return e
        return None

    @staticmethod
    def _label_cover(cov_map, tpl_m, e_lab, body_win, slack=LABEL_COVER_SLACK):
        """灰度候选匹配在 body_win（**窗口内**的身体左上角）时，它头顶上标签该在的
        位置有多少金色文字。返回 0~1 的召回率。

        为什么不用掩膜自己的全局最高分当闸门（曾经的写法）：`_mask_f1` 是 F1，分母里
        含「窗里多出来的金像素」。地图上一有金色美术（岩浆亮芯、沙地），标签框里就总会
        多出金像素，F1 掉到 0.38 **而标签其实就在原地**——2026-09-19 run13 实测
        「金标签 0.38 / 灰度 0.66」，那个 0.66 是货真价实的身体，却被整帧作废。
        改成只看召回后，多出来的金像素不再扣分；位置不对（漂到 UI 或别的老鼠身上）
        才会掉到 0，那正是要拦的情况。

        必须允许 ±slack 的挪动：名字是 1~2px 粗的抗锯齿笔画，单点比对时错开一个像素
        召回就能掉一半，而灰度身体匹配本来就允许几像素的位置误差。`cov_map` 是
        `_label_cover_map` 一次性算好的整窗召回图，`cov_map[y][x]` = 标签左上角放在
        (x,y) 时的召回率。

        位置换算：偏移量存的是「身体左上角 − 标签左上角」，所以标签位置 = 身体位置 − 偏移。
        """
        lth, ltw = tpl_m.shape
        lx, ly = body_win[0] - e_lab["ox"], body_win[1] - e_lab["oy"]
        my, mx_ = cov_map.shape
        ys0, ys1 = max(0, ly - slack), min(my, ly + slack + 1)
        xs0, xs1 = max(0, lx - slack), min(mx_, lx + slack + 1)
        if xs0 >= xs1 or ys0 >= ys1:
            return 0.0                 # 标签位置整个跑到窗外 ⇒ 这个候选不可能裹着真身
        return float(cov_map[ys0:ys1, xs0:xs1].max() / max(float(tpl_m.sum()), 1e-6))

    @staticmethod
    def _label_cover_map(win_g, tpl_m):
        """整窗的金色标签召回图：`cov_map[y][x]` = 标签左上角放在 (x,y) 时，标签模板的
        金像素有多少落在窗内的金色文字上（TM_CCORR 就是逐位置内积，两边都是 0/1 掩膜，
        所以内积＝重合像素数）。"""
        return cv2.matchTemplate(win_g, tpl_m, cv2.TM_CCORR)

    def stage_rect(self, bgr):
        """舞台（Flash 画布）在帧里的矩形 (x0, y0, x1, y1)；检测不出来就返回整帧。

        为什么要它：窗口比舞台大，四周那条客户端底色不是游戏内容，玩家永远不在那里。
        跟踪窗一旦滑进去，整窗就是一片纯色——**灰度模板在纯色上照样能凑出 0.45~0.56
        的分数**（掩膜/标签这时候早就 0 了），锁就眼睁睁丢在那儿。2026-09-19 run20 的
        `_lost_window_c_final.png` 就是这么一张纯色图。

        做法：底色是**整条**连续同色的带子，所以按行/列找「最外侧那条足够长的同色带」。
        单条行列可能整条都是碰巧同色的地图内容（纯色天空），所以取 9 条采样线的公共结论
        （多数表决），再要求结果像个居中的舞台，否则当检测失败退回整帧。
        """
        h, w = bgr.shape[:2]
        full = (0, 0, w, h)
        if self._stage_key == (h, w) and self._stage_rect is not None:
            # 失败（退回整帧）的结论只认 1 秒，过期重测（见下面写缓存那段说明）
            if self._stage_rect != full or time.time() - self._stage_fail_t < 1.0:
                return self._stage_rect
        # 底色 = 左边缘中点那一小块（窗口边框在最外面几个像素，往里挪一点再采）
        px = min(w - 1, max(3, int(w * 0.01)))
        patch = bgr[h // 2 - 3:h // 2 + 4, px:px + 8].astype(np.int16)
        base = np.median(patch.reshape(-1, 3), axis=0)
        lb = (np.abs(bgr.astype(np.int16) - base).max(axis=2) <= STAGE_TOL)

        def side_bounds(mask_1d):
            """一维同色掩膜里，最外两侧那条长带的位置。返回 (内侧起点, 内侧终点)。"""
            runs, s = [], None
            for i in range(mask_1d.shape[0]):
                if mask_1d[i] and s is None:
                    s = i
                elif not mask_1d[i] and s is not None:
                    if i - s >= STAGE_MIN_RUN:
                        runs.append((s, i))
                    s = None
            if s is not None and mask_1d.shape[0] - s >= STAGE_MIN_RUN:
                runs.append((s, mask_1d.shape[0]))
            if not runs:
                return 0, mask_1d.shape[0]
            return runs[0][1], runs[-1][0]

        # 竖向：在舞台内部的 9 条列上多数表决；横向：9 条行上多数表决
        cs = [int(w * f) for f in np.linspace(0.25, 0.75, 9)]
        rs = [int(h * f) for f in np.linspace(0.25, 0.75, 9)]
        vcol = lb[:, cs].mean(axis=1) > 0.5
        hrow = lb[rs, :].mean(axis=0) > 0.5
        y0, y1 = side_bounds(vcol)
        x0, x1 = side_bounds(hrow)
        rect = (x0, y0, x1, y1)
        # 校验只看「像一个**居中**的矩形舞台」，**不能**用「舞台占画面多大比例」——
        # Flash 舞台恒为 800x600 不随窗口缩放，窗口一最大化，舞台就只剩整帧的三成宽，
        # 旧那版 (x0<=0.30w / x1>=0.70w) 反而把**正确的**检测判成失败（2026-09-20 实测
        # 帧 2576x1456、舞台 (888,439,1688,1039)：中心分毫不差，却因为 888>0.30w 被退回整帧）。
        # 居中性才是那个不变的特征，两种坏样本（天空色连成一片 → 矩形偏到一边；
        # 底色带太窄 → 矩形只有 23px 宽）也都被它挡住。
        cx0, cy0 = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        ok = (x1 - x0 >= STAGE_MIN_W and y1 - y0 >= STAGE_MIN_H
              and abs(cx0 - w * 0.5) <= 0.15 * w
              and abs(cy0 - h * 0.5) <= 0.15 * h)
        if not ok:
            rect = full
        # 2026-10-26 23:5x：**高度几乎永远是对的，宽度会飘，而且没人管。**
        # 同一台机器、同一个 1024x700 的窗口，连测 8 帧：5 帧给 800、1 帧给 854、
        # 1 帧给 802（_b_3.png = (149,61,1003,661)）。原因是地图左侧正好是一片
        # 和客户端底色相近的深蓝天空，同色带从画面边缘一路吃进舞台，左边界被推到 149。
        # 于是同一个 run 里所有坐标整体歪 37px —— 洞口、名字标签、身体框全跟着歪，
        # 而且 stage_rect 是**按帧尺寸缓存**的，一旦测歪就歪到底。
        # Flash 舞台恒为 800x600 且在客户端区里**水平居中**，所以只要高度对得上，
        # 宽度就别信检测值，直接按居中算回去。
        if rect != full:
            # ⚠ 2026-10-27 01:0x：**「800x600」这句写死了，窗口一缩放就全失效。**
            # 这台机器开 150% DPI 缩放：同一个窗口，没开 DPI 感知的进程量到 1024x700、
            # 开了的（bot.py 因为 import dxcam 会被设成 DPI 感知）量到 **1536x1050**，
            # 舞台跟着变成 1201x901。原来这条按 `abs(高-600)<=3` 才修正宽度，
            # 缩放后恒不成立 → 宽度不修正，而且下面缓存逻辑会把结果钉死。
            # 改成按「测到的高度 / 600」推出缩放比 s，宽度就按 800*s 反推 + 水平居中。
            _hh = y1 - y0
            _s = _hh / 600.0
            if 0.5 <= _s <= 4.0:
                _ww = int(round(800 * _s))
                if abs((x1 - x0) - _ww) <= max(3, int(0.12 * _ww)):
                    x0 = (w - _ww) // 2
                    x1 = x0 + _ww
                    y1 = y0 + _hh
                    rect = (x0, y0, x1, y1)
        # ⚠ 2026-10-27 01:0x：**检测失败的结果绝对不能缓存。**
        # 实测 00:41:29 一帧黑屏（窗口刚缩放/重绘）把「舞台矩形检测失败」的整帧结果
        # 缓存到了 (1050,1536) 这个键上，此后 6 分钟里每帧都直接命中缓存、再也不重测，
        # 舞台不夹紧 → 标定锚点/窗口全错 → 连续 9500 帧锁不上（[Bot] 已连续 N 帧没锁上）。
        # 成功的结果照旧按帧尺寸缓存（尺寸不变就一直用）；失败只记 1 秒，过期重测。
        self._stage_key = (h, w)
        self._stage_rect = rect
        if rect != full:
            self._stage_fail_t = 0.0
            self._stage_good = rect
            self._stage_good_key = (h, w)
            # 舞台量准了才敢换算字形尺寸（退回整帧时的高度是窗口高度，不是舞台高度，
            # 拿它当缩放比会把 1.5 算成 1.75；这时候宁可保持上一次的换算结果）
            _sc = (y1 - y0) / 600.0
            if 0.5 <= _sc <= 4.0:
                self._apply_label_scale(_sc)
            print(f"[LOCK] 舞台矩形 {x1 - x0}x{y1 - y0} @({x0},{y0})（帧 {w}x{h}，"
                  f"底色 BGR{tuple(int(v) for v in base)}）—— 跟踪/标定夹在这个矩形内")
        else:
            # 检测不出来时**先用同尺寸上一次测准的矩形**，别一上来就退回整帧。
            # 2026-10-27 05:4x 实测：窗口被缩到 1040x739（舞台变成 800x600@(120,81)，1.0x），
            # 之后换到一张背景色 BGR(137,107,98) 和客户端底色几乎一样的地图，同色带从画面
            # 边缘一路连进舞台 → `side_bounds` 只找到一条整行同色带，返回 (run_end, run_start)
            # **两头颠倒**（日志里是 `原始 (23,81,8,31)`），检测判失败 → 退回整帧 → 搜索窗
            # 撑到整幅、灰度和掩膜在哪儿都能凑高分 → **锁到左下角聊天框上**（现场
            # `_dbg_frame.png` 红框正压在「感谢 Apluska，我们获得了 0 块奶酪！」那行字上），
            # 标定连续 7500+ 帧全部「候选簇都没有跟着 ← 走」，一局都动不了。
            # 帧尺寸没变就说明窗口没动过，上一次测准的舞台矩形依然成立——用它夹紧，
            # 比退回整帧安全得多。只有「换过尺寸且这一帧又测失败」才退回整帧。
            self._stage_fail_t = time.time()
            good = self._stage_good if self._stage_good_key == (h, w) else None
            if good is not None:
                gx0, gy0, gx1, gy1 = good
                gx0 = max(0, min(gx0, w)); gx1 = max(gx0, min(gx1, w))
                gy0 = max(0, min(gy0, h)); gy1 = max(gy0, min(gy1, h))
                if gx1 - gx0 >= STAGE_MIN_W and gy1 - gy0 >= STAGE_MIN_H:
                    rect = (gx0, gy0, gx1, gy1)
                    self._stage_rect = rect
                    print(f"[LOCK] 舞台矩形检测失败（原始 ({x0},{y0},{x1},{y1}) 不像居中舞台，"
                          f"帧 {w}x{h}，底色 BGR{tuple(int(v) for v in base)}）—— "
                          f"沿用上次测准的 {gx1-gx0}x{gy1-gy0} @({gx0},{gy0}) 继续夹紧"
                          f"（1 秒后重测）")
                    return rect
            # 真的没有可用的旧矩形才退回整帧（不夹）。**地图背景色正好等于客户端底色时
            # 必然走到这里**（2026-09-19 run23 实测：一张地图的天空就是 BGR(149,116,106)，
            # 和字母框底色一模一样，于是同色带从画面边缘一直连进舞台，找不出舞台边界）。
            # 这时候夹紧是无效的，但也不该有副作用——所以只报一次，好让日志能区分
            # 「夹了」和「没夹」。
            print(f"[LOCK] 舞台矩形检测失败（原始 ({x0},{y0},{x1},{y1}) 不像居中舞台，"
                  f"帧 {w}x{h}，底色 BGR{tuple(int(v) for v in base)}）—— 无可用旧矩形，"
                  f"退回整帧，不做夹紧（1 秒后重测）")
        return rect

    def track_locked(self, gray, search=64, gold=None, bgr=None):
        """用模板库跟踪已锁定的玩家。连续失配超过 LOCK_LOST_GRACE_S 秒就放弃锁定。

        宽限期按**时间**而不是按帧数。为什么：帧率在这套 bot 里是**随状态剧烈变化**的，
        数帧数等于让宽限期跟着飘。实测 _live_run7.log（2026-09-19）——注意不能拿全局中位
        数当依据（全局中位 20fps 是被大量「无锁定空转帧」拉下来的，那些帧一帧活都不干）：
        真正**锁定中**的帧中位只有 12.7fps（p10 9.2 / p90 18.2），而无锁定时中位 48.4fps。
        也就是说「连续 8 帧」在锁定中约 0.63s、无锁定时只要 0.17s。顺手把锁定中最大的
        那笔开销（world_shift 每帧 21ms，见 bot.py anti-stuck）改成按需调用之后，锁定
        中的帧率会往上走，8 帧的宽限期又会跟着缩水——按时间就没这问题。

        `gold` 是当前帧的金色文字掩膜（float32）。模板库里有掩膜条目时必须传，
        否则那些条目会被跳过——见 set_player_lock 里为什么名字标签要存掩膜版本。
        """
        if self._lock_patch is None:
            return None
        # 这次调用一定会给出结论，先把上一次的分数清掉——早退的分支（尺寸不符、窗比
        # 身体框还小）不会走到下面赋值，留着上次数值会让调用方读到**别的帧**的分数。
        # 2026-09-20 就是这么崩的：窗口中途最大化，`_verify_lock_moves` 读到不存在的
        # `_last_score` 直接 AttributeError 打死主循环。
        self._last_score = -1.0
        h, w = gray.shape
        # 锁是在**旧尺寸**的帧里建的，换尺寸后 bbox 在新帧里没有意义（夹进舞台之后
        # 甚至会出现 x1<x0）。直接作废让调用方重新标定，别拿它去猜。
        if self._lock_shape is not None and gray.shape != self._lock_shape:
            self.clear_player_lock()
            return None
        x, y, bw, bh = self._lock_bbox
        # ---- 名字标签直接定坐标（2026-10-26）----
        # 字形是静态像素，整幅舞台搜一次只要几毫秒，峰值恒 1.00、次峰 0.25；而灰度
        # 模板在 40 人的房间里隔几秒就被别的老鼠/物品挡一次，一挡就飘、飘久了就丢锁。
        # 标签可见时**以它为准**，把「漂移 → 丢锁」这条失效路径整个绕过去；
        # 标签不可见（死了、萨满局字形/颜色不同）就照旧走下面的模板跟踪。
        # 掰动上限 LABEL_ANCHOR_JUMP：超过就当这次标签认错，不去动锁。
        if bgr is not None and self._label_body_off is not None:
            _a = self._label_match(bgr, mask=gold)
            if _a is None and time.time() >= getattr(self, "_anchor_dbg_t", 0):
                self._anchor_dbg_t = time.time() + 2.0
                _sx0, _sy0, _sx1, _sy1 = self.stage_rect(bgr)
                _m = gold if gold is not None else self.text_mask(bgr)
                _win = _m[_sy0:_sy1, _sx0:_sx1].astype(np.float32)
                _th, _tw = self._label_tpl_m.shape
                if _win.shape[0] >= _th and _win.shape[1] >= _tw:
                    _c = cv2.matchTemplate(_win, self._label_tpl_m, cv2.TM_CCOEFF_NORMED)
                    _, _mv, _, _ml = cv2.minMaxLoc(_c)
                    print(f"[ANCHOR] 标签没匹配上：峰 {_mv:.2f} @({_sx0+_ml[0]},{_sy0+_ml[1]})"
                          f" 次峰 {self._second_peak(_c, _ml, LABEL_REACQ_EXCL):.2f}"
                          f" gold={'有' if gold is not None else '无'}")
            if _a is not None:
                _sc, _tx, _ty = _a
                _ox, _oy = self._label_body_off
                _asx0, _asy0, _asx1, _asy1 = self.stage_rect(bgr)
                _bx = max(_asx0, min(_tx + _ox, _asx1 - bw))
                _by = max(_asy0, min(_ty + _oy, _asy1 - bh))
                if max(abs(_bx - x), abs(_by - y)) <= LABEL_ANCHOR_JUMP:
                    self._lock_bbox = (_bx, _by, bw, bh)
                    self._last_score = _sc
                    self._lock_lost = 0
                    self._lock_lost_since = 0.0
                    self._weak_frames = 0
                    self._weak_cand = None
                    self._last_player = (_bx + bw // 2, _by + bh // 2,
                                         _by + bh, (_bx, _by, bw, bh))
                    self._player_lost_frames = 0
                    return self._last_player
        # 搜索窗按**所有**模板撑开：名字标签模板挂在身体上方（oy 是负数），只按身体
        # bbox ± search 开窗的话标签会跑到窗外，永远匹配不上。
        x0, y0 = x - search, y - search
        x1, y1 = x + bw + search, y + bh + search
        for e in self._lock_bank:
            th, tw = _entry_hw(e)
            x0 = min(x0, x + e["ox"] - search)
            y0 = min(y0, y + e["oy"] - search)
            x1 = max(x1, x + e["ox"] + tw + search)
            # 这里必须是 **y**，不是 x。写成 x 的版本（2026-09-19 之前一直是）会让窗的
            # **下边缘**跟着角色的水平位置跑：角色越靠右，窗就越往下撑到画面底部。
            # 实测 run16 身体框 77x77、search=120 时窗高 443（正常 317），整块 HUD、
            # 下面的地面、别的老鼠全被圈进来——于是灰度模板能凑出 0.66~0.85 的假高分、
            # 掩膜也能抓到别人的白色名字，锁就被拽走了。定位飘忽的头号原因。
            y1 = max(y1, y + e["oy"] + th + search)
        # 夹两遍：先夹进帧（模板偏移离谱时窗会伸到画面外），再夹进**舞台**——舞台外
        # 那条客户端底色不是游戏内容，滑进去就是纯色窗（见 stage_rect 的说明）。
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        sx0, sy0, sx1, sy1 = self.stage_rect(bgr) if bgr is not None else (0, 0, w, h)
        x0, y0 = max(x0, sx0), max(y0, sy0)
        x1, y1 = min(x1, sx1), min(y1, sy1)
        if x1 - x0 < bw or y1 - y0 < bh:
            return None
        win = gray[y0:y1, x0:x1]
        win_g = gold[y0:y1, x0:x1] if gold is not None else None
        # 正常窗就是 身体框 + 2*search（标签条目把窗往上/左再撑 ox/oy 那么多）。比这个
        # 大出一大截就说明有模板条目的偏移量离谱，窗会盖住半屏——那时候灰度模板在哪儿
        # 都能凑出高分。只报一次，并把每个条目的偏移和尺寸都打出来。
        _exp_h = bh + 2 * search
        _exp_w = bw + 2 * search
        if not getattr(self, "_lock_win_warned", True) and \
                (win.shape[0] > _exp_h + 60 or win.shape[1] > _exp_w + 60):
            self._lock_win_warned = True
            print(f"[LOCK] 搜索窗异常大 {win.shape[1]}x{win.shape[0]}（正常 "
                  f"{_exp_w}x{_exp_h}；帧 {w}x{h}，身体框 ({x},{y}) {bw}x{bh}，"
                  f"窗原点 ({x0},{y0})）条目 "
                  f"{[(e['ox'], e['oy'], *_entry_hw(e), bool(e.get('mask'))) for e in self._lock_bank]}")
        # 库里每个模板都试一遍取最大值，位置统一换算成身体 bbox 左上角。
        # 单个模板挡不住：实测锁定帧站姿 0.94，下一帧老鼠起跳姿势全变只剩 0.51；
        # 而名字标签是静态 UI，同一段被动录的 80 帧里全程 1.00。
        mx, ml, r, tw_best = -1.0, (0, 0), None, bw
        mx_key = None          # 赢家的次序键 (分数, 离上一帧的负距离)，见下面压分档的说明
        nx_, ny_ = x, y        # 一个模板都没试成时别让下面炸 UnboundLocalError
        mx_gray, mx_cov, mx_dist = -1.0, 1.0, 0   # 赢家的灰度分/标签召回/离上一帧多远（诊断用）
        _win_exempt = False    # 赢家是**靠无标签豁免**活下来的（这种帧不许学模板）
        e_lab = self._label_entry()
        # 整窗的标签召回图只算一次，供每个候选查「自己头顶上有没有金字」。顺带把它的
        # 全局最大值记下来：**「窗里到底有没有自己的金色名字」**和**「名字在不在身体
        # 上方该在的地方」是两个完全不同的问题**，一个值区分不了。2026-09-19 好几次
        # 误判就是因为只有后者（而后者依赖前者的像素级对齐）。
        # 窗被夹进舞台之后可能**比标签模板还小**（角色贴着舞台边缘时就是这样），
        # matchTemplate 要求模板不大于图像，否则直接抛 cv2.error 把整轮 bot 带崩
        # （2026-09-19 run22 实测）。放不下就当「这一帧没有标签信息」。
        _lth, _ltw = e_lab["tpl_m"].shape if e_lab is not None else (0, 0)
        cov_map = (self._label_cover_map(win_g, e_lab["tpl_m"])
                   if (e_lab is not None and win_g is not None
                       and win_g.shape[0] >= _lth and win_g.shape[1] >= _ltw) else None)
        if cov_map is not None and cov_map.size:
            self._last_cov_any = float(cov_map.max() / max(float(e_lab["tpl_m"].sum()), 1e-6))
        else:
            self._last_cov_any = -1.0
        self._last_cov_exp = -1.0   # 最高分灰度候选头顶上的标签召回率
        best_mask = -1.0       # 掩膜锚点（金色名字标签）这一帧的最好**召回**（只用于日志）
        best_mask_extra = -1.0  # 上面那个位置上「框里多出来的金像素」占整框的比例（同只用于日志）
        best_gray = -1.0       # 灰度身体模板这一帧的最好匹配分（过完平坦块判据，但**没**过标签判据）
        # ⚠ 「灰度最优」和「最终赢家」是两回事，2026-10-07 才发现分不开：
        # 赢家可能是一条分不高但**没被压分**的条目；而真正找到身体的那条（灰度 0.96）
        # 被 cov 判据压成 0.45 之后**反而没赢**。上一轮只打了 best_gray 这个标量，
        # 无法判断它落在画面哪儿、头顶标签召回多少 —— 只能对着 0.45 猜。分开记：
        #   mx_*  = 赢家（决定 pos 的那条）
        #   bg_*  = 灰度最优那条（不管它最后赢没赢）
        mx_i = -1                              # 赢家是模板库里第几条
        bg_pos, bg_cov, bg_dist = None, -1.0, -1   # 灰度最优那条：位置 / 头顶标签召回 / 离上一帧
        for _ei, e in enumerate(self._lock_bank):
            th, tw = _entry_hw(e)
            if win.shape[0] < th or win.shape[1] < tw:
                continue
            _exempt = False        # 这一条候选是不是靠无标签豁免活下来的
            if e.get("mask"):
                if win_g is None:
                    continue
                # 掩膜锚点用**召回图**当分数，不用 F1（理由见 LABEL_MASK_RECALL_MIN）。
                # cov_map 就是 matchTemplate(win_g, tpl_m, TM_CCORR) = 逐位置的**重合像素数**，
                # 召回 = 重合数 / 模板金像素数，与 cov_map 同序 —— argmax 位置完全一样，
                # 所以直接换掉 rr 即可，下面的 minMaxLoc 一行都不用动。
                # 形状对不上（窗被夹得比标签还小）才退回 F1，反正那种帧也赢不了。
                used_rec = (cov_map is not None
                            and cov_map.shape == (win.shape[0] - th + 1,
                                                  win.shape[1] - tw + 1))
                rr = cov_map if used_rec else _mask_f1(win_g, e["tpl_m"])
            else:
                rr = cv2.matchTemplate(win, e["tpl"], cv2.TM_CCOEFF_NORMED)
            _, mxv, _, mlv = cv2.minMaxLoc(rr)
            raw_gray = -1.0        # 夹分**之前**的灰度分（诊断「无金标签的候选本来多高分」
            if e.get("mask"):
                ts = max(float(e["tpl_m"].sum()), 1e-6)
                rec = float(mxv) / ts if used_rec else float(mxv)
                box_gold = float(win_g[mlv[1]:mlv[1] + th, mlv[0]:mlv[0] + tw].sum())
                extra = max(0.0, box_gold - float(mxv)) / max(float(th * tw), 1.0)
                if rec > best_mask:
                    best_mask, best_mask_extra = rec, extra
                if rec < LABEL_MASK_RECALL_MIN or extra > LABEL_MASK_EXTRA_MAX:
                    continue       # 名字没对齐，或者这是一整块金斑不是文字 → 整条不要
                mxv = rec
            else:
                # 平坦块退化解：TM_CCOEFF_NORMED 会各自减均值再除以标准差，窗口越平坦
                # 分母越接近 0，比值就越没有意义——实测一块**几乎全平的木地板**能拿到
                # score=1.00 并把锁钉死在那儿。真老鼠身上是绝不会平的，所以低于
                # LOCK_MIN_PATCH_STD 的匹配直接作废（不是降分，是整条候选丢掉）。
                patch = win[mlv[1]:mlv[1] + th, mlv[0]:mlv[0] + tw]
                if float(patch.std()) < LOCK_MIN_PATCH_STD:
                    continue
                # 光形状像还不够：自己老鼠头顶上一定挂着金色名字标签，所以这个灰度候选
                # 按标签偏移量推出来的那块位置上必须真的有金字（见 _label_cover）。
                # run11 那个「锁漂到计分板面板」的坏匹配就是死在这一条上：灰度 0.57~0.60
                # 看着还行，但它头顶上没有金标签。反过来，真身 0.66 配上 1.00 的标签覆盖
                # 照样过——判据比原来那版「标签全局 F1 不够高就整帧作废」精确得多。
                cov = 1.0
                if cov_map is not None:
                    cov = self._label_cover(cov_map, e_lab["tpl_m"], e_lab,
                                            (mlv[0] + e["ox"], mlv[1] + e["oy"]))
                if mxv > best_gray:
                    best_gray = float(mxv)
                    self._last_cov_exp = cov
                    bg_pos = (x0 + mlv[0] + e["ox"], y0 + mlv[1] + e["oy"])
                    bg_cov = float(cov)
                    bg_dist = max(abs(bg_pos[0] - x), abs(bg_pos[1] - y))
                raw_gray = float(mxv)
                if cov < LABEL_COVER_MIN:
                    # 名字这一帧看不见（掉落中/被别的老鼠挡住）**不等于**这个候选是错的：
                    # run17 实测掉落过程中 cov 掉到 0.00~0.10，但身体就好好地在原地，而且
                    # 一两秒后又回到 1.00。所以不能直接丢掉它，只把分数**压到进不了强匹配**，
                    # 交给下面的弱匹配通道用「峰值突出 + 位置连续」去判它（最多活 WEAK_MAX 帧）。
                    # 真要拦的那种漂移（锁跑到固定 UI / 别人的白名字上）原始分本来就只有
                    # 0.57~0.60，压下去之后同样进不了强匹配，保护没丢。
                    #
                    # 例外：分数高到 NO_LABEL_MIN 以上、而且位置**几乎没挪窝**的，就是真身
                    # 被面板压住/名字没渲染出来，压分反而把锁白白丢掉。判据只看位置连续性
                    # （见 NO_LABEL_MIN 的说明：分数区分不了真身和周期纹理漂移）。
                    _jx = abs((x0 + mlv[0] + e["ox"]) - x)
                    _jy = abs((y0 + mlv[1] + e["oy"]) - y)
                    # 还有个前提：**整个窗里都找不到自己的名字**。窗里名字明明在（any 高）、
                    # 只是不在这个候选的头顶，那说明这个候选是**别的老鼠**——同房间的老鼠
                    # 长得一模一样，灰度模板对它照样能打 0.95。2026-09-19 run25 实测
                    # 09:18:08 那条就是：灰度 0.95、次峰只有 0.49，看着完美，可标签召回
                    # any=1.00 / 该在的位置只有 0.15——名字在窗里的**另一只老鼠头上**。
                    # 光靠「分数高 + 没挪窝」会把这个冒牌货认下来，锁就飘到别人身上了。
                    #
                    # ④ 而且这套豁免只能**连续**给 NOLABEL_KEEP_S 秒。静态背景能一帧一帧
                    # 把豁免续下去（分数高、原地不动、窗里也没金子，三条判据全对），不封顶
                    # 就是个永动机。超时之后按原路压分 → 交回弱匹配/重标定。
                    if (mxv >= NO_LABEL_MIN and max(_jx, _jy) <= NO_LABEL_JUMP
                            and self._last_cov_any < LABEL_COVER_MIN
                            and self._nolabel_run < NOLABEL_KEEP_MAX):
                        _exempt = True      # 计数放在循环外按帧算，见下面
                    else:
                        mxv = min(mxv, WEAK_MIN + 0.05)
            _cx, _cy = x0 + mlv[0] + e["ox"], y0 + mlv[1] + e["oy"]
            # 压分档里所有候选都恰好等于 WEAK_MIN+0.05，`mxv > mx` 于是只对**库里排最
            # 靠前**的那条成立——位置信息被抹平了（2026-10-07 23:04:31 那条：真身候选
            # 灰 0.69 排在后面，输给了排第一的灰 0.48）。补一个次序键：同分才比**离上一帧
            # 近**（压分档本来就没有标签证据，位置连续是唯一判据，同弱匹配通道）。
            # 注意 mx 本身**不动**：弱匹配拿 mx 当「次峰突出」的基准，动它会误伤。
            _key = (round(float(mxv), 6), -max(abs(_cx - x), abs(_cy - y)))
            if mx_key is None or _key > mx_key:
                mx_key = _key
                mx, ml, r, tw_best = float(mxv), mlv, rr, tw
                nx_, ny_ = _cx, _cy
                # 赢家要是「灰度很高但头顶没金标签」的候选，把它的原始灰度和离上一帧
                # 多远一起记下来——这两种情况修法相反：「身体还待在原地」（被半透明
                # 聊天框/道具栏面板压住名字）该让锁撑住，而「一下跳到别处」（灰度匹到
                # 旁边那块周期纹理/UI 上）必须拦。光看分数区分不出来，得看距离。
                mx_gray = raw_gray
                mx_cov = cov if not e.get("mask") else 1.0
                mx_dist = max(abs(nx_ - x), abs(ny_ - y))
                mx_i = _ei
                _win_exempt = _exempt
        # 供「首次失配」的存盘日志用：到底是被标签判据否掉的（best_gray 有分但没有
        # 候选过得了 _label_cover），还是灰度模板自己也匹配不上（best_gray 本身就低）。
        # 这两者的修法完全相反——前者要动标签/掩膜，后者要动模板库/搜索窗。
        # 2026-09-19 就是这么发现「金标签 0.38 / 灰度 0.66」的：灰度那个 0.66 是真身，
        # 却因为掩膜的全局 F1 被岩浆金像素拉低而被整帧作废。
        self._last_score = mx
        self._last_best_mask = best_mask
        self._last_best_mask_extra = best_mask_extra
        self._last_best_gray = best_gray
        self._last_win_i = mx_i
        self._last_win_gray = mx_gray
        self._last_win_cov = mx_cov
        self._last_win_dist = mx_dist
        self._last_bg_pos = bg_pos
        self._last_bg_cov = bg_cov
        self._last_bg_dist = bg_dist
        self._last_win_exempt = _win_exempt
        # 豁免预算按**帧**结算（候选循环里会同时给主块和镜像块发豁免，按候选算等于
        # 把预算砍半）：豁免来的赢家扣一格；赢家头上**看到了标签**的帧把预算回满。
        if _win_exempt:
            self._nolabel_keep += 1
            self._nolabel_run += 1
        elif mx_cov >= LABEL_COVER_MIN:
            self._nolabel_run = 0
        self._last_nolabel_run = self._nolabel_run
        # 「灰度高分但头顶没金标签」每次锁定只报一条。这是当前最可疑的跟丢机制：
        # 灰度 0.94 / 金标签 0.00 这种组合，是「名字被半透明面板压住（身体没动）」
        # 还是「匹配飘到隔壁那块周期纹理上（身体跳了）」——**看 mx_dist 就知道**，
        # 别再从分数上猜。keep=1 表示这一帧按 NO_LABEL_MIN 豁免认下了。
        if mx_gray >= LOCK_STRONG and mx_cov < LABEL_COVER_MIN and not self._noted_label_less:
            self._noted_label_less = True
            keep = (mx_gray >= NO_LABEL_MIN and mx_dist <= NO_LABEL_JUMP
                    and self._last_cov_any < LABEL_COVER_MIN)
            print(f"[TRACK] 灰度高分但无标签：灰度={mx_gray:.2f} cov={mx_cov:.2f} "
                  f"标签召回any={self._last_cov_any:.2f} 离上一帧={mx_dist}px "
                  f"→ {'认下(keep=1)' if keep else '压分（是别的老鼠/位置跳了）'} "
                  f"pos=({nx_},{ny_}) 上一帧=({x},{y})")
        # 身体框必须整个落在画面内。掩膜锚点（名字标签）最容易踩这个：标签挂在身体上方，
        # 角色贴到画面下沿时按 oy 往下推出来的「身体」会整个越界，nx/ny 于是带负数或
        # 超过帧高。夹紧到帧内而不是判失配——丢锁的代价（重标定要几秒~几十秒）远大于
        # 脚底 y 偏这几十像素。同样要夹进舞台：舞台外是客户端底色，玩家不可能在那儿。
        nx, ny = max(sx0, min(nx_, sx1 - bw)), max(sy0, min(ny_, sy1 - bh))

        if mx >= LOCK_STRONG:
            self._lock_lost = 0
            self._lock_lost_since = 0.0
            self._weak_frames = 0
            self._weak_cand = None
        else:
            # 弱匹配：**分数低不等于位置错**。实测起跳那一帧最佳匹配 (74,68) 分 0.51、
            # 次峰只有 0.35，红框正好套在老鼠身上。所以只要峰够突出、位置没瞬移，
            # 就认下这个新姿势并把这块像素学进库——下一帧同一姿势就能高分匹配。
            second = self._second_peak(r, ml, max(8, tw_best // 2)) if r is not None else 0.0
            if (mx >= WEAK_MIN and mx - second >= WEAK_MARGIN
                    and max(abs(nx - x), abs(ny - y)) <= WEAK_JUMP
                    and self._weak_frames < WEAK_MAX):
                if self._weak_frames == 0:
                    print(f"[TRACK] 弱匹配接下新姿势：score={mx:.2f} 次峰={second:.2f}"
                          f" pos=({nx},{ny}) 模板库{len(self._lock_bank)}块")
                if self._weak_cand is not None:
                    cx_, cy_ = self._weak_cand
                    if abs(nx - cx_) <= WEAK_CONFIRM and abs(ny - cy_) <= WEAK_CONFIRM:
                        # 连续两帧指向同一处才学，免得把一帧噪声当成新姿势存进库
                        self._bank_add(gray[ny:ny + bh, nx:nx + bw])
                self._weak_cand = (nx, ny)
                self._weak_frames += 1
                self._lock_bbox = (nx, ny, bw, bh)
                self._last_player = (nx + bw // 2, ny + bh // 2, ny + bh,
                                     (nx, ny, bw, bh))
                self._player_lost_frames = 0
                return self._last_player
            self._lock_lost += 1
            if self._lock_lost_since == 0.0:
                self._lock_lost_since = time.time()
            if not self._lock_dbg_done:
                self._lock_dbg_done = True
                self._lock_dump(win, (x0, y0), ml, len(self._lock_bank), r, box=(x, y, bw, bh),
                                win_g=win_g)
            if time.time() - self._lock_lost_since <= LOCK_LOST_GRACE_S:
                return self._last_player       # 短暂遮挡：先沿用上一帧
            # 真丢锁这一帧也要存一次图：第一次失配常常只是探针在按键（老鼠被按着走，
            # 单帧匹配不上很正常），那时候的分数代表不了「为什么最后真的跟丢了」。
            # 2026-09-19 run13 就是这么被误导的：存下来的 0.38/0.66 取自标定探针期间。
            self._lock_dump(win, (x0, y0), ml, len(self._lock_bank), r, suffix="_final",
                              box=(x, y, bw, bh), win_g=win_g)
            return None                        # 连续跟丢，交给调用方重新标定

        self._lock_bbox = (nx, ny, bw, bh)
        # ⚠ 靠豁免活下来的这一帧**不许学**（2026-10-07）。豁免的定义就是「这条候选没有
        # 标签证据」，而下面这一支是唯一的**高分数就无条件收**的入口（没有次峰突出度
        # 那种判据）：一条 0.99 的静态背景会在这里被收进库、还被 0.12 混进主模板。
        # 当晚那个 0.99 的库条目就是这么来的——它的匹配图上**一排五个 0.99**（背景纹理
        # 平移不变），之后每一帧都能无条件夺冠。不学它，它最多活 NOLABEL_KEEP_S 秒。
        if mx > LOCK_REFRESH and not _win_exempt:
            # 匹配很好时：少量混进主模板（吸收走路动画），同时把这一姿势收进库。
            # 这里切出来的块尺寸靠上面那次夹紧保证等于 (bh, bw)——掩膜锚点让 mx 每帧都
            # > LOCK_REFRESH，这个分支从「偶尔进」变成「每帧进」，不夹紧的话角色贴到
            # 画面边缘时切出来的短块会让 addWeighted 抛 cv2.error 打死主循环
            # （2026-09-19 实跑 08:06:33 就是这么崩的）。
            cur = gray[ny:ny + bh, nx:nx + bw].astype(np.float32)
            merged = cv2.addWeighted(self._lock_patch.astype(np.float32), 0.88, cur, 0.12, 0)
            self._lock_patch = merged.astype(np.uint8)
            self._bank_add(gray[ny:ny + bh, nx:nx + bw])
        self._last_player = (nx + bw // 2, ny + bh // 2, ny + bh, (nx, ny, bw, bh))
        self._player_lost_frames = 0
        return self._last_player

    # ====== Player Detection ======
    def detect_player(self, bgr, gray):
        self._last_bgr = bgr               # 供 _lock_dump 存彩色窗口用
        # 窗口被重建/改尺寸后（游戏重启会换分辨率）上一帧形状对不上，absdiff 会抛
        # cv2.error 直接把主循环打死；锁定的模板坐标也随之作废，一并清掉重标。
        if self._prev_gray is not None and self._prev_gray.shape != gray.shape:
            self._prev_gray = None
            self.clear_player_lock()
        if self._lock_patch is not None:
            # 库里有掩膜条目（金色名字标签）才去算掩膜：HSV/阈值一趟约 1~2ms，
            # 没必要给纯灰度锁定白付这个钱。
            # 掩膜走 text_mask：锁定过就用建库时采到的颜色（见 set_player_lock），
            # 没采到才按亮度兜底。写死的金色阈值在本机客户端上一个像素都命不中。
            gold = (self.text_mask(bgr).astype(np.float32)
                    if any(e.get("mask") for e in self._lock_bank) else None)
            # 2026-10-27：**先认名字标签**。实测同一局里标签每帧都稳在 0.94
            # 上下（`_bot_run_1026_2310.log`），而 48x48 的身体模板在这种 40~50 人
            # 挤在一层地板上的活动图里 **4~10 秒必掉**：那块像素里塞着相邻好几只老鼠，
            # 它们一动模板就对不上了。日志里「重挂成功 → 4.3s 后跟丢 → 再重挂」
            # 一直循环，就是这个原因，送洞数自然永远是 0。
            # 标签是文字，同字体同字号，人群挡它也只是短暂几帧；位置 = 标签左上角
            # + 固定偏移，直接就是身体框。所以标签在手时根本不需要身体模板。
            p = self._track_by_label(bgr, gold)
            if p is not None:
                self._lock_lost = 0
                self._lock_lost_since = 0.0
                return p
            p = self.track_locked(gray, gold=gold, bgr=bgr)
            if p is not None:
                return p
            print(f"[LOCK] 跟丢（连续失配超过 {LOCK_LOST_GRACE_S:.1f}s 都是 "
                  f"score<{LOCK_STRONG}，最后一次 {getattr(self, '_last_score', -1):.2f}）"
                  f" 本次锁定豁免过 {self._nolabel_keep} 帧无标签高分"
                  f"（当前这轮 {self._nolabel_run}/{NOLABEL_KEEP_MAX}，"
                  f"打满说明锁是靠「窗里没金子」硬撑的）")
            self.clear_player_lock()          # 跟丢了，交回颜色/运动兜底

        h, w = bgr.shape[:2]
        center_x, center_y = w // 2, h // 2
        sc = self._scale(bgr, 1)

        # Track by continuity: prefer the candidate nearest the last known position,
        # so we stay locked on OUR mouse instead of jumping to other players' mice.
        anchor_x, anchor_y = center_x, center_y
        if self._last_player is not None:
            anchor_x, anchor_y = self._last_player[0], self._last_player[1]

        player = self._detect_player_color(bgr, anchor_x, anchor_y)
        if player is None and self._prev_gray is not None:
            player = self._detect_player_motion(gray, anchor_x, anchor_y)

        # Temporal smoothing
        if player is not None:
            px, py, pfeet, bbox = player
            if self._last_player is not None:
                lx, ly, _, _ = self._last_player
                dist = np.sqrt((px - lx) ** 2 + (py - ly) ** 2)
                if dist > 80 * sc:
                    self._player_lost_frames += 1
                    if self._player_lost_frames < 3:
                        return self._last_player
                    # Accept new position after timeout
            self._last_player = player
            self._player_lost_frames = 0
            return player

        self._player_lost_frames += 1
        if self._player_lost_frames <= 5 and self._last_player is not None:
            return self._last_player
        return None

    def _in_play_area(self, x, y, h):
        """排除顶部游戏状态栏和底部 UI 带（聊天框/道具栏/玩家列表）。

        这些区域里的图标也是肤色斑块，会被 HSV 过滤当成玩家，
        导致 detect_player 锁死在 UI 上返回恒定坐标。
        """
        return h * self.cfg["play_area_top"] < y < h * self.cfg["play_area_bottom"]

    def player_candidates(self, bgr):
        """所有通过颜色/形状/可玩区过滤的玩家候选 [(cx, cy, feet_y, bbox), ...]。

        多人房间里会有多只老鼠（也可能混入 UI 图标），冷启动时无法凭这一份
        列表区分哪只是本地玩家，需要配合输入相关性（见 Bot._calibrate_lock）。
        """
        h = bgr.shape[0]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo = np.array(self.cfg["player_hsv_lower"], dtype=np.uint8)
        hi = np.array(self.cfg["player_hsv_upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 玩家身体面积范围（比芝士大）
            if area < 120 or area > 8000:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)
            if aspect < 0.4 or aspect > 2.5:
                continue
            extent = area / (cw * ch) if cw * ch > 0 else 0
            if extent < 0.35:
                continue
            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if not self._in_play_area(cx, cy, h):
                continue
            out.append((cx, cy, y + ch, (x, y, cw, ch)))
        return out

    def _detect_player_color(self, bgr, anchor_x, anchor_y):
        best = None
        best_dist = float("inf")
        for cand in self.player_candidates(bgr):
            cx, cy = cand[0], cand[1]
            dist = np.sqrt((cx - anchor_x) ** 2 + (cy - anchor_y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = cand
        return best

    def _detect_player_motion(self, gray, anchor_x, anchor_y):
        """Fallback: detect moving blob near center via frame differencing."""
        h = gray.shape[0]
        diff = cv2.absdiff(gray, self._prev_gray)
        _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, self._k7)
        diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, self._k5)

        contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_dist = float("inf")
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80 or area > 5000:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if not self._in_play_area(cx, cy, h):
                    continue
                dist = np.sqrt((cx - anchor_x) ** 2 + (cy - anchor_y) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    best = (cx, cy, y + ch, (x, y, cw, ch))
        return best

    def store_frame(self, gray):
        """Store frame for next motion detection."""
        self._prev_gray = gray.copy()

    # ====== Cheese-on-back Detection ======
    # 2026-10-06 重写。老口径（框内最大黄块面积 >= max(60, w*h*0.25)）在 58x58 的身体框上
    # 门槛是 **841px**，而**真·背芝士的黄掩膜只有 119px**（见下），差 7 倍 ——
    # 也就是说这个函数**从来没可能对机器人自己的背芝士报 True**。
    # 这就是「走得再近也 0 次送洞」的直接原因：`cb` 恒 0 → state 永远进不了 1。
    CB_MIN_AREA = 60      # 真·背芝士 119px（reference），留一半余量收下更小的（被鼠身挡掉半个）
    CB_MAX_AREA = 2600    # 2026-10-26：万圣节的霉芝士**整块驮在背上**（约 55x30 = 1000+ px，
                          # 见 _zoom_pump.png），原来 600 的上限会把它整个判掉，而南瓜灯
                          # 只有 100~180 反而全能过 —— 等于判据挑反了。放宽到 2600 之后
                          # 「不是真的驮着」交给第五道闸门（跟不跟手）去否决。
                          # 旧说明：同一帧全画面最大黄块才 409px（远处平台上的地图芝士）；
                          # 地图背景/地形是几千~几十万的大块，这一刀正好把「站在黄地板上」切掉
    CB_OVL_MIN = 0.35     # 黄块要有 35% 以上落在身体框（外扩 10px）里，才算「贴着我」
    CB_BOX_PAD = 10
    CB_TOP_SLACK = 4      # 质心可以比身体框上沿高一点点，但**不能飘在上面**：
                          # 实测 corpus/1006_121006/121510_near_target_01x 那几张，鼠标头顶
                          # 悬着一颗**黄花皮球**（地图装饰，该图满地图都是），它的黄条纹
                          # 连通块 149px、下缘刚好探进外扩框里（ovl=0.90）——三条闸门全过。
                          # 而它质心 y=398 在身体框上沿 405 **之上**；
                          # 已确认的 3 个真阳性质心全落在框内（+65 / +13 / +62），
                          # 所以「质心不得高于上沿(容差 4px)」能切掉皮球、不误杀真芝士。
    CB_UP_EXTRA = 20      # （保留给区域搜索用）
    # 2026-10-26 21:45 改：原来按「**每帧**挪 ≥20px 才判」，实测跑动时单帧只挪 5~10px
    # （30~60fps），这条**永远不成立** → 判据形同虚设，假阳照旧。改成按**累计位移**
    # 设检查点：从上次立桩算，身体累计挪够 CB_FOLLOW_MOVE 才量一次「黄块跟没跟手」，
    # 量完把桩挪到当前位置。而且**从「只否决」升级成「必须自证」**：没攒够证据一律
    # 返回 False —— 站在南瓜旁边不动时，装饰永远攒不到证据。
    CB_FOLLOW_MOVE = 20   # 累计位移到这么多，就量一次「跟没跟手」
    CB_FOLLOW_TOL = 16    # 黄块质心相对身体框的偏移允许抖这么多 px（转身/跳跃动画）
    CB_FOLLOW_NEED = 1    # 要连续攒到几次「跟得住」才算真背着（1 次 ≈ 走 20px）
    # 2026-10-27 加的**相对基线**闸门（见 has_cheese_on_back 末尾）。
    # 活动期每只老鼠都穿同一套「圣诞帽 + 绿披风」，披风实测 H44.2/S169.7/V160.8、
    # 身体框内约 119px —— 和霉芝士（H44.3/S165.6/V166.6、整块 200~375px）
    # **颜色完全一样**，只是面积小一截。前五道闸门全拦不住它，因为披风本来就长在
    # 身上、跟着走、质心也在框里。于是 cb 在 0/1 之间乱跳：机器人「以为背了芝士」
    # 往洞口跑，跑两步又判掉 —— 用户看到的「在洞口徘徊」有一半是这个。
    CB_AREA_DELTA = 80    # 比「本底最小那一块」大出这么多才算多了一块芝士
                          # （实测披风本底 60~63px；霉芝士整块 200~375px，留足余量）

    def has_cheese_on_back(self, bgr, player_bbox):
        """机器人身上（背上/头顶）是不是驮着一块芝士。

        判据（2026-10-06 重写）：在全帧黄掩膜上取**连通块**，挑那块「落在身体框
        （外扩 CB_BOX_PAD）里最多」的，然后四道闸门：
          ① 全帧面积在 [CB_MIN_AREA, CB_MAX_AREA] —— 真·背芝士实测 119px；
             地图地形/背景是几千~几十万的大块，一刀切掉（老口径死在没这条上限）
          ② 图像素里 ≥CB_OVL_MIN 的比例要落在身体框内 —— 「贴着我才算」
          ③ 不贴画面边（贴边的黄块是铺开的地形，detect_cheeses 也是这么毙的）
          ④ 质心落在身体框内，且**不得高于身体框上沿**（容差 CB_TOP_SLACK）——
             头顶飘着的黄皮球/装饰就是这么切掉的

        为什么改：老口径是「身体框 ROI 内最大黄块面积 >= max(60, w*h*0.25)」，
        58x58 的框上门槛 = 841px，比真芝士（119px）**大 7 倍**，结构上不可能触发，
        于是 cb 恒 0 → state 永远进不了 1 → 一次洞都送不了。真阳性样本：
        corpus/1005_224725/225105_bg_004（Hebbebeee 背驮芝士）与
        corpus/1006_175302（机器人自己头顶顶着芝士、那场跑了 84 分钟 0 送洞）。
        """
        h, w = bgr.shape[:2]
        mask = self._cheese_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            self._cb_rel = None        # 画面上压根没有黄块 → 记忆作废，下次捡到重新记
            return False

        x, y, bw, bh = player_bbox
        p = self.CB_BOX_PAD
        x1 = max(0, x - p)
        y1 = max(0, y - p)
        x2 = min(w, x + bw + p)
        y2 = min(h, y + bh + p)
        if x2 <= x1 or y2 <= y1:
            return False

        sub = lab[y1:y2, x1:x2]
        vals, counts = np.unique(sub[sub > 0], return_counts=True)
        if len(vals) == 0:
            self._cb_rel = None        # 身体框里没有黄 → 记忆作废
            return False

        best_i, best_ovl = -1, 0.0
        for v, c in zip(vals, counts):
            a = int(st[v, cv2.CC_STAT_AREA])
            if a < self.CB_MIN_AREA or a > self.CB_MAX_AREA:
                continue
            ovl = float(c) / float(a)
            if ovl > best_ovl:
                best_i, best_ovl = int(v), ovl
        if best_i < 0 or best_ovl < self.CB_OVL_MIN:
            self._cb_rel = None        # 没有够格的黄块贴着我 → 记忆作废
            return False

        # 相对基线：身体框里那块绿到底是「披风本底」还是「多了块芝士」
        _area = int(st[best_i, cv2.CC_STAT_AREA])
        if getattr(self, "_cb_area_min", None) is None or _area < self._cb_area_min:
            self._cb_area_min = _area
        if _area < self._cb_area_min + self.CB_AREA_DELTA:
            self._cb_rel = None
            return False

        bx = int(st[best_i, cv2.CC_STAT_LEFT])
        by = int(st[best_i, cv2.CC_STAT_TOP])
        bw2 = int(st[best_i, cv2.CC_STAT_WIDTH])
        bh2 = int(st[best_i, cv2.CC_STAT_HEIGHT])
        if bx <= 2 or by <= 2 or bx + bw2 >= w - 2 or by + bh2 >= h - 2:
            self._cb_rel = None
            return False

        mx = float(cen[best_i][0])
        my = float(cen[best_i][1])
        if not (x - p <= mx <= x + bw + p
                and y - self.CB_TOP_SLACK <= my <= y + bh + p):
            self._cb_rel = None
            return False

        # 第五道闸门：**跟不跟手**（2026-10-26）。
        # 万圣节活动把芝士换成了南瓜灯，地图上又点了一堆同色南瓜。站在南瓜旁边时
        # 上面四道闸门全能通过，cb 于是假性打满 → 判「背起来了」→ 跑向洞口 →
        # 走开两步「芝士」就没了 → 记一次掉芝士。实测 21:37~21:41：背起 21 次、
        # 送洞 2 次（9%），绝大多数是南瓜。
        # 真芝士**跟着身体走**：身体走一段，它相对身体的偏移不变；地图装饰钉在世界
        # 上不动，身体一走偏移就跟着变。所以这里立「检查点」并**要求自证**：
        #   ① 从上次检查点算，身体累计挪够 CB_FOLLOW_MOVE 才量一次；
        #   ② 量的时候偏移还在容差内 → 记一次「跟得住」；
        #   ③ 攒够 CB_FOLLOW_NEED 次之前一律返回 False（不是「默认通过、只否决」）。
        # 站位不动时量不了，那就维持现状、不新增证据也不推翻 —— 真假都先按没有。
        rel = (mx - x, my - y)
        st = self._cb_rel
        if st is None:
            self._cb_rel = (rel[0], rel[1], x, y, 0, 0)
            return False
        prx, pry, bpx, bpy, nf, nm = st
        moved = max(abs(x - bpx), abs(y - bpy))
        if moved < self.CB_FOLLOW_MOVE:
            return nf >= self.CB_FOLLOW_NEED     # 还没挪够，先按已有证据说话
        if max(abs(rel[0] - prx), abs(rel[1] - pry)) <= self.CB_FOLLOW_TOL:
            nf, nm = nf + 1, 0
        else:
            nm += 1
            if nm >= 2:                          # 连着两次对不上 → 不是它的
                self._cb_rel = None
                return False
        self._cb_rel = (rel[0], rel[1], x, y, nf, nm)
        return nf >= self.CB_FOLLOW_NEED

    # ====== Ground/Platform Detection ======
    def build_terrain_mask(self, bgr):
        """Build a binary mask of terrain-colored pixels. Cached: rebuilds every 3 calls."""
        self._terrain_frame += 1
        if self._terrain_cache is not None and self._terrain_frame % 3 != 0:
            return self._terrain_cache
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask_brown = cv2.inRange(hsv, (5, 30, 30), (28, 255, 220))     # 木质/棕色平台
        mask_green = cv2.inRange(hsv, (30, 50, 30), (80, 255, 220))    # 草地/绿色地形
        mask_gray = cv2.inRange(hsv, (0, 0, 40), (180, 40, 200))       # 灰色石头
        mask_bright = cv2.inRange(hsv, (0, 0, 170), (180, 30, 255))    # 冰/雪等亮色地形

        combined = cv2.bitwise_or(mask_brown, mask_green)
        combined = cv2.bitwise_or(combined, mask_gray)
        combined = cv2.bitwise_or(combined, mask_bright)

        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._k10)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self._k5)
        self._terrain_cache = combined
        return combined

    def has_ground_at(self, gray, terrain_mask, x, y):
        """Check if a screen position has ground."""
        h, w = gray.shape
        # Check terrain mask first (fast)
        if 0 <= y < h and 0 <= x < w and terrain_mask[y, x] > 0:
            return True
        # Fall back to texture/edge heuristics
        s = 15
        y1, y2 = max(0, y - s), min(h, y + s)
        x1, x2 = max(0, x - s), min(w, x + s)
        if x2 - x1 < 10 or y2 - y1 < 6:
            return False
        patch = gray[y1:y2, x1:x2].astype(np.float32)
        if np.std(patch) > 15:
            return True
        edges = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        if np.mean(np.abs(edges)) > 8:
            return True
        return False

    def find_ground_below(self, bgr, gray, px, py_feet):
        """Find ground surface below player feet. Returns (ground_y, left_edge, right_edge) or None."""
        h, w = gray.shape
        terrain_mask = self.build_terrain_mask(bgr)

        # 从脚底向下扫描地面（150px 范围，步长 3；范围=角色可能的下落缓冲距离）
        ground_y = None
        for y in range(py_feet, min(py_feet + 150, h - 1), 3):
            if self.has_ground_at(gray, terrain_mask, px, y):
                ground_y = y
                break

        if ground_y is None:
            return None

        # 向左找平台左边缘（350px 范围，步长 7）
        left_edge = px
        for x in range(px, max(px - 350, 0), -7):
            if not self.has_ground_at(gray, terrain_mask, x, ground_y):
                left_edge = x
                break

        # 向右找平台右边缘（350px 范围，步长 7）
        right_edge = px
        for x in range(px, min(px + 350, w), 7):
            if not self.has_ground_at(gray, terrain_mask, x, ground_y):
                right_edge = x
                break

        # 平台太窄（<40px）视为不可站立，返回 None
        if right_edge - left_edge < 40:
            return None
        return (ground_y, left_edge, right_edge)

    # ====== Debug Drawing ======
    def draw_debug(self, bgr, player, cheeses, holes, ground_info, state, falling, stuck):
        """Draw detection overlays on frame."""
        d = bgr.copy()
        h, w = d.shape[:2]
        cx = w // 2

        # Player bounding box and marker
        if player:
            px, py, pfeet, bbox = player
            bx, by, bw, bh = bbox
            cv2.rectangle(d, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.drawMarker(d, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            cv2.line(d, (px, py), (cx, py), (0, 200, 0), 1)

        # Cheese markers
        for cx_c, cy_c in cheeses:
            cv2.circle(d, (cx_c, cy_c), 10, (0, 255, 255), 2)
            cv2.drawMarker(d, (cx_c, cy_c), (0, 255, 255), cv2.MARKER_DIAMOND, 8, 1)

        # Hole markers
        for hx, hy in holes:
            cv2.circle(d, (hx, hy), 12, (0, 140, 255), 2)
            cv2.drawMarker(d, (hx, hy), (0, 140, 255), cv2.MARKER_TILTED_CROSS, 8, 1)

        # Ground line
        if ground_info:
            gy, gl, gr = ground_info
            cv2.line(d, (gl, gy), (gr, gy), (255, 200, 0), 3)

        # Status
        mode_text = "HOLE" if state == 1 else "CHEESE"
        color = (255, 150, 0) if state == 1 else (0, 255, 200)
        cv2.putText(d, mode_text, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if player:
            cv2.putText(d, f"({px},{py})", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Warnings
        y_off = 44
        if falling:
            cv2.putText(d, "FALL!", (6, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_off += 18
        if stuck:
            cv2.putText(d, "STUCK", (6, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
            y_off += 18
        if state == 1 and player:
            cv2.putText(d, "HAS_CHEESE", (6, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 230, 230), 1)

        return d
