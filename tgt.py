# -*- coding: utf-8 -*-
"""目标选择的纯函数（跨帧迟滞）。改动一的核心。

**为什么单独一个模块**：`import bot` 不是无副作用的——`bot.py:186` 在模块级就
`Listener(...)` 装了全局键盘钩子，`bot.py:22` 还劫持了 `builtins.print`。
离线单测 import 它等于往系统里装钩子。所以这里**零 import**（不引 cv2 / numpy / time），
`now` 由调用方传进来，于是可以逐帧喂合成序列，把「两块芝士等距时会不会来回翻」
这种问题在不开游戏的情况下测出来。

## 病根

改前bot.py:1666 是 `tgt = min(reach, key=最近距离)` —— **无记忆的纯函数**。
两块芝士距离接近时必然逐帧翻转，净位移趋近 0，永远走不到任何一块跟前。
2026-10-05 实测：相邻采样目标跳变中位 **104px**，而玩家自身位移中位只有 46px。

## 两个必须守住的实现要点（写错过就白改）

1. **身份匹配一律在完整 `cands` 上做，绝不在「同平台子集」上做。**
   `ground_info` 一丢，子集就从「同平台」突变成全体（见 `_same_platform`），
   匹配基准跟着变 —— 这是**第二个跳变源**，会把迟滞整个绕过去。
   子集只用来「挑新目标」，不用来「认旧目标」。
2. **不要用背景位移做位置预测。** `bot.py:556` 存的是 `max(abs(dx),abs(dy))`，
   符号丢了；而且它量的是「相对锚点」的位移，锚点一到 `STUCK_MOVE_TOL=12` 就重置，
   是个上限 12 的锯齿信号。这里改用**有界半径预算**：匹配半径随 `miss` 逐帧放大，
   封顶 `TGT_MATCH_R_MAX_PX`。

## 回退

`on=False` 时输出与改前的裸 `min` **逐帧一致**（`_test_tgt.py` 里有断言）。
`TGT_HYST=0` 就是这个模式。
"""

# —— 可调参数 ——
# 全部按 `scale` 缩放后再用（scale 见 bot.py：max(1, min(w,h)/600)）。
DEFAULTS = {
    "SAME_PLATFORM_DY": 60,     # 沿用 bot.py 原值：判「目标和我站在同一层」的高度容差
    "TGT_MATCH_R_PX": 30,       # 认「还是同一块芝士」的初始半径（~1.5 个芝士宽）
    "TGT_MATCH_GROW_PX": 8,     # 跟丢后每帧把半径放大这么多（芝士会被别的玩家推走）
    "TGT_MATCH_R_MAX_PX": 90,   # 半径上限：再放大就成「随便认一个」了
    "TGT_MIN_COMMIT": 15,       # 锁定后至少守这么多帧才允许被抢（防刚锁定就抖掉）
    "TGT_SWITCH_MARGIN_PX": 60, # 挑战者要比现任近这么多 px 才有资格谈「换」
    "TGT_SWITCH_CONFIRM": 5,    # 且要连续这么多帧都赢 —— 防两块等距芝士来回翻
    "TGT_MISS_CLEAR": 10,       # 连续跟丢这么多帧 → 放弃芝士目标
    "TGT_MISS_CLEAR_HOLE": 20,  # 洞放宽：洞比芝士稳，丢了多半是遮挡/一时检测不到
    "TGT_REACH_GIVEUP": 60,     # 守这么多帧以后才开始结账「走不到」
    "TGT_STALL_S": 8.0,         # 「走不到」的结账周期：每这么久算一次有没有真的靠近
    "TGT_STALL_RATIO": 0.95,    # 一个周期内距离至少要缩到这个比例，否则算停滞
    "TGT_NEAR_KEEP_PX": 40,     # 已经这么近了就不判「走不到」（都贴上了，是拾取/状态机的事）
    "TGT_COOLDOWN_S": 3.0,      # 跟丢后的兜底超时（秒）
}


def new_state():
    """跨帧状态。由调用方（bot.py）持有，本模块不存任何全局量。"""
    return {
        "kind": None,      # 当前锁定的目标类型：0=芝士 1=洞；None=没锁
        "x": 0.0, "y": 0.0,
        "age": 0,          # 这个目标守了多少帧
        "miss": 0,         # 连续跟丢多少帧
        "last_seen": 0.0,  # 上次成功匹配到的时刻
        "since": 0.0,      # 本次锁定建立的时刻
        "start_d": 0.0,    # 刚锁上时的距离²（判「压根没靠近过」的基准）
        "best_d": 0.0,     # 刚锁上时的距离²（= start_d，留一份给「走不到」的门槛用）
        "chk_t": 0.0,      # 上次结账的时刻（判停滞，见 TGT_STALL_S）
        "chk_d": 0.0,      # 结账基准距离²：一个周期后要缩到 TGT_STALL_RATIO² 以内
        "chal_n": 0,       # 挑战者连续赢了几个帧
        "_chkey": None,    # 挑战者的整数化坐标，用来判「还是同一个挑战者吗」
    }


def _dist2(ax, ay, bx, by):
    return (ax - bx) ** 2 + (ay - by) ** 2


def _nearest(cands, x, y):
    return min(cands, key=lambda c: _dist2(c[0], c[1], x, y))


def _same_platform(cands, ground_info, px, py, C):
    """「同平台优先」的子集，找不到同平台的就回退全体。

    完全照抄 bot.py 原逻辑（gl<=x<=gr 且 |dy|<=SAME_PLATFORM_DY）。
    **这个子集只用来挑新目标**；认旧目标要在完整 cands 上做，见模块 docstring。
    """
    if not ground_info:
        return cands
    _, gl, gr = ground_info
    same = [c for c in cands
            if gl <= c[0] <= gr and abs(c[1] - py) <= C["SAME_PLATFORM_DY"]]
    return same if same else cands


def _nearest_excluding(cands, x, y, ex, ey, er):
    """离 (x,y) 最近、且不在 (ex,ey) 半径 er 以内的候选；都没有就 None。

    「走不到就换一个」必须排除当前这块 —— 否则重新挑出来的还是它自己，
    等于原地打转（2026-10-05 单测 [5] 就是这么挂的）。
    """
    er2 = er * er
    best, bd = None, None
    for c in cands:
        if _dist2(c[0], c[1], ex, ey) <= er2:
            continue
        d = _dist2(c[0], c[1], x, y)
        if bd is None or d < bd:
            best, bd = c, d
    return best


def _match(cands, x, y, r):
    """在完整 cands 里找离 (x,y) 半径 r 以内最近的那个；没有就返回 None。"""
    r2 = r * r
    best, bd = None, r2
    for c in cands:
        d = _dist2(c[0], c[1], x, y)
        if d <= bd:
            best, bd = c, d
    return best


def _lock(st, kind, t, now, best_d, log, why):
    st["kind"] = kind
    st["x"], st["y"] = float(t[0]), float(t[1])
    st["age"] = 1
    st["miss"] = 0
    st["since"] = now
    st["last_seen"] = now
    st["start_d"] = best_d
    st["best_d"] = best_d
    st["chk_t"] = now
    st["chk_d"] = best_d
    st["chal_n"] = 0
    st["_chkey"] = None
    if log:
        log("lock", why, t[0], t[1])


def pick_target(cands, kind, ground_info, px, py, scale, st, now, C=None,
                on=True, log=None):
    """挑这一帧要走向哪个目标。返回 (x, y) 或 None。

    cands      : [(x, y), ...]，这一帧的芝士或洞（调用方已做完自身过滤）
    kind       : 0=芝士 1=洞，用来判「目标类型变了没」
    ground_info: find_ground_below 的返回值 (_, gl, gr)，或 None
    scale      : 屏幕缩放；所有半径都乘它
    st         : new_state() 的产物，调用方持有
    now        : 单调秒（time.time() 即可；本模块不 import time）
    on         : False → 退化成改前的裸 min，用于一键回退 / A/B
    log        : 可选 log(action, why, x, y) 回调
    """
    C = C or DEFAULTS

    # 没有候选：清空锁定
    if not cands:
        if st["kind"] is not None and log:
            log("clear", "no_cands", 0, 0)
        st.update(kind=None, age=0, miss=0, chal_n=0, _chkey=None)
        return None

    # 目标类型变了（芝士 <-> 洞）：立刻重挑，不适用迟滞
    if st["kind"] != kind:
        t = _nearest(_same_platform(cands, ground_info, px, py, C), px, py)
        _lock(st, kind, t, now, _dist2(t[0], t[1], px, py), log, "kind")
        return (st["x"], st["y"])

    # 回退模式：与改前的裸 min 逐帧一致（同平台优先 + 最近）
    if not on:
        t = _nearest(_same_platform(cands, ground_info, px, py, C), px, py)
        _lock(st, kind, t, now, _dist2(t[0], t[1], px, py), log, "nohyst")
        return (st["x"], st["y"])

    cur_d2 = _dist2(st["x"], st["y"], px, py)

    # 挑战者只从「可达子集」里挑；但下面判「是不是同一块」要用满半径比
    reach = _same_platform(cands, ground_info, px, py, C)
    ch = _nearest(reach, px, py)
    ch_d2 = _dist2(ch[0], ch[1], px, py)

    # (a) 该不该换目标？挑战者要**明显更近**，还要连续确认若干帧
    same_as_cur = (_dist2(ch[0], ch[1], st["x"], st["y"])
                   <= (C["TGT_MATCH_R_MAX_PX"] * scale) ** 2)
    margin2 = (C["TGT_SWITCH_MARGIN_PX"] * scale) ** 2
    if (not same_as_cur) and st["age"] >= C["TGT_MIN_COMMIT"] \
            and ch_d2 + margin2 < cur_d2:
        k = (int(ch[0]), int(ch[1]))
        st["chal_n"] = st["chal_n"] + 1 if st["_chkey"] == k else 1
        st["_chkey"] = k
        if st["chal_n"] >= C["TGT_SWITCH_CONFIRM"]:
            _lock(st, kind, ch, now, ch_d2, log, "switch")
            return (st["x"], st["y"])
    else:
        st["_chkey"], st["chal_n"] = None, 0

    # (b) 沿用一个旧目标：在**完整 cands** 上按半径预算匹配
    r = min(C["TGT_MATCH_R_PX"] + C["TGT_MATCH_GROW_PX"] * min(st["miss"], 20),
            C["TGT_MATCH_R_MAX_PX"]) * scale
    m = _match(cands, st["x"], st["y"], r)
    if m is not None:
        st["x"], st["y"] = float(m[0]), float(m[1])
        st["miss"] = 0
        st["age"] += 1
        st["last_seen"] = now
        if cur_d2 < st["best_d"]:
            st["best_d"] = cur_d2
        # 「守了很久还是走不到」：守够 GIVEUP 帧 + 起始距离够远之后，**每 TGT_STALL_S 秒
        # 结一次账** —— 这段时间里距离有没有缩到 TGT_STALL_RATIO？没有就判它够不着，换一块。
        # 排除它自己，否则重新挑出来的还是同一块。
        #
        # ⚠ 2026-10-07 深夜改：原来那句「从没靠近过」= `best_d > start_d * 0.8`，
        # **被「原地跳」骗过去了**。实测（_bot_run_1008_0045_goldgate.log，122 秒那次锁）：
        # 芝士在玩家**上方 239px** 的高台上，够不着；bot 只会原地往上跳，跳起来的那一瞬
        # 距离从 239 缩到 ~203（0.85 倍）→ 0.8 这道门判成「有进展」→ 一直不放弃，
        # 于是 122 秒全在跳，STUCK 每 3 秒报一次（#25..#46+），一块芝士都没吃到。
        # 也不能用「最近距离多久没刷新」（best_t 那版试过，写在草稿里就否了）：跳跃顶点
        # 抖动一下就能把 best_d 推进 1px，照样一次一次续命。**必须按比例结账**：
        # 真走路是连续靠近（26px/s，8 秒 208px），原地跳的净进展 ≈ 0，两者一眼分得开。
        # 起始距离那道门（start_d > 半径上限²）保留：已经贴着目标（几 px）时不该判死。
        if (st["age"] > C["TGT_REACH_GIVEUP"]
                # 2026-10-26：「起始距离近」这条豁免**不能给另一层的目标**。它本来是为了
                # 「已经贴上了就别判死」，可 2D 距离近 ≠ 够得着：芝士在脚下一层 110px 处时
                # 2D 距离刚好压线（90px 半径×scale=107），豁免一开就永久免死。
                # 实测 19:59：target (817,253)、人 (820,143)，45 秒原地左右晃，一次没换目标。
                # 另一层的目标一律走「结账」：8 秒内距离没缩 5% 就换一块。
                and (st["start_d"] > (C["TGT_MATCH_R_MAX_PX"] * scale) ** 2
                     or abs(st["y"] - py) > C["SAME_PLATFORM_DY"] * scale)
                and cur_d2 > (C["TGT_NEAR_KEEP_PX"] * scale) ** 2
                and now - st["chk_t"] >= C["TGT_STALL_S"]):
            if cur_d2 < st["chk_d"] * C["TGT_STALL_RATIO"] ** 2:
                st["chk_t"], st["chk_d"] = now, cur_d2      # 有进展，重新起算
            else:
                alt = _nearest_excluding(reach, px, py, st["x"], st["y"],
                                         C["TGT_MATCH_R_MAX_PX"] * scale)
                st["chk_t"], st["chk_d"] = now, cur_d2
                if alt is not None:
                    _lock(st, kind, alt, now, _dist2(alt[0], alt[1], px, py),
                          log, "unreachable")
        return (st["x"], st["y"])

    # (c) 跟丢了：先保持承诺（仍返回旧位置往前走），攒够 miss 才真的丢
    st["miss"] += 1
    st["age"] += 1
    if now - st["last_seen"] > C["TGT_COOLDOWN_S"]:
        _lock(st, kind, ch, now, ch_d2, log, "cooldown")
        return (st["x"], st["y"])
    limit = C["TGT_MISS_CLEAR_HOLE"] if kind == 1 else C["TGT_MISS_CLEAR"]
    if st["miss"] > limit:
        _lock(st, kind, ch, now, ch_d2, log, "lost")
    return (st["x"], st["y"])
