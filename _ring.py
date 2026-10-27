# -*- coding: utf-8 -*-
"""帧环形缓冲：把「出事前后」的画面留下来，供改完离线复跑。

为什么需要它：原来的 `_dump_frame` 每 `DBG_DUMP_S` 秒把**同一个文件**覆盖写一次
（bot.py:1314），跑完只剩最后一帧——「目标跳变那一刻画面长什么样」根本拿不到。
而离线调 `detect_cheeses` / `has_cheese_on_back` 的阈值，前提是手里有一批**真帧**。

用法（bot.py 侧）：

    self._ring = Ring()                 # __init__ 里建一次
    self._ring.push(bgr, meta)          # 每帧调，meta 是 dict
    self._ring.event("cb_up")           # 触发：把此前 N 帧 + 其后 N 帧落盘

三个设计要点：

1. **必须 .copy()**。主循环里 `bgr` 是 dxcam 的复用缓冲，存引用等于什么都没存。
2. **默认关闭**（`DBG_RING` 不设 = 0）。平时的跑法一行都不多干，采集时才置 1。
3. 落盘分两半：触发时先写「之前 N 帧」，再把「之后 N 帧」挂起（`_post`），
   逐帧补。这样一次事件拿到 2N 帧的上下文，而不用一直全量存盘。
   补后一半期间**新事件会被丢掉**（返回 False），因为缓冲已经被清空重建，
   此时的前文不完整；丢掉的那次事件，其帧本来就在这次的前后文里。
"""

import os
import time
from collections import deque

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))


def _truthy(v):
    return str(v).strip().lower() not in ("0", "", "no", "false", "off")


class Ring:
    def __init__(self, n=None, out_dir=None, enabled=None):
        if n is None:
            n = int(os.environ.get("DBG_RING_N", "15"))
        self.n = max(1, n)
        if enabled is None:
            enabled = _truthy(os.environ.get("DBG_RING", "0"))
        self.on = bool(enabled)

        self.buf = deque(maxlen=self.n)     # [(bgr, meta), ...]
        self.out = out_dir or os.path.join(_HERE, "corpus", time.strftime("%m%d_%H%M%S"))
        self._post = None                   # {"tag":str,"idx":int,"left":int}
        self._seq = 0
        self.n_frames = 0
        self.n_events = 0
        self._csv = None

        if self.on:
            os.makedirs(self.out, exist_ok=True)
            self._csv = open(os.path.join(self.out, "labels.csv"), "w",
                             encoding="utf-8", newline="")
            # bbox 是给 has_cheese_on_back(bgr, bbox) 离线复跑用的 —— 只存人中心点的话，
            # 那批帧根本没法重跑「背芝士」那条路径（2026-10-05 加列时才发现）。
            self._csv.write("file,event,idx,fn,t,state,cb,px,py,bbox,score,"
                            "keys,fall,stuck,bg,note\n")

    # ---------- 每帧 ----------
    def push(self, bgr, meta):
        if not self.on or bgr is None:
            return
        self.n_frames += 1
        if self._post is not None:          # 正在补「事件之后」的帧
            self._write(bgr, meta, self._post["tag"], self._post["idx"])
            self._post["idx"] += 1
            self._post["left"] -= 1
            if self._post["left"] <= 0:
                self._post = None
        self.buf.append((bgr.copy(), dict(meta or {})))

    # ---------- 触发 ----------
    def event(self, tag, note=""):
        """把此前 N 帧落盘，并安排其后 N 帧。正在补后文时返回 False（丢弃该次）。"""
        if not self.on:
            return False
        if self._post is not None:
            return False
        self.n_events += 1
        nb = len(self.buf)
        for i, (b, m) in enumerate(self.buf):
            self._write(b, m, tag, i - nb, note=note)
        self.buf.clear()
        self._post = {"tag": tag, "idx": 0, "left": self.n}
        return True

    def sample(self, bgr, meta, tag="bg"):
        """只落当前这一帧（周期背景样本用，不动缓冲、不挂后文）。"""
        if not self.on or bgr is None:
            return
        self._write(bgr, meta, tag, 0)

    # ---------- 落盘 ----------
    def _write(self, bgr, meta, tag, idx, note=""):
        self._seq += 1
        name = "%s_%s_%03d_%+d.png" % (time.strftime("%H%M%S"), tag, self._seq, idx)
        try:
            cv2.imwrite(os.path.join(self.out, name), bgr)
        except Exception:
            return
        m = meta or {}
        keys = m.get("keys", "")
        if isinstance(keys, (list, tuple, set)):
            keys = "|".join(str(k) for k in sorted(keys))
        self._csv.write("%s,%s,%d,%s,%.3f,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
            name, tag, idx, m.get("fn", ""), time.time(), m.get("state", ""),
            m.get("cb", ""), m.get("px", ""), m.get("py", ""), m.get("bbox", ""),
            m.get("score", ""),
            keys, m.get("fall", ""), m.get("stuck", ""), m.get("bg", ""),
            str(note).replace(",", ";")))
        self._csv.flush()

    def close(self):
        if self._csv:
            try:
                self._csv.close()
            except Exception:
                pass
            self._csv = None

    def status(self):
        if not self.on:
            return "ring=off"
        return ("ring=on n=%d frames=%d events=%d -> %s"
                % (self.n, self.n_frames, self.n_events, self.out))
