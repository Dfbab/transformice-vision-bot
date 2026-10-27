"""内存读取：附加 Transformice 进程，扫描堆内存定位玩家/芝士/洞。可选增强，attach 失败则退化为纯视觉模式。"""
import struct, ctypes, time

# 玩家结构体字段偏移（相对结构体基址）
PLAYER_X_OFFSET = 32        # X 坐标（float）
PLAYER_Y_OFFSET = 36        # Y 坐标（float）
PLAYER_TYPE_OFFSET = 0      # 对象类型 ID（1/2/26 表示玩家）

# Windows MEMORY_BASIC_INFORMATION 结构，供 VirtualQueryEx 查询内存区域属性
class MBI(ctypes.Structure):
    _fields_ = [("base", ctypes.c_ulonglong), ("alloc", ctypes.c_ulonglong),
                ("prot", ctypes.c_uint32), ("_a", ctypes.c_uint32),
                ("size", ctypes.c_ulonglong), ("state", ctypes.c_uint32),
                ("rprot", ctypes.c_uint32), ("type", ctypes.c_uint32), ("_b", ctypes.c_uint32)]

class Mem:
    def __init__(self):
        self.pm = None
        self.player_base = None

    def attach(self):
        try:
            import pymem
            self.pm = pymem.Pymem("Transformice")
            return True
        except: return False

    def find_all_players(self, max_n=100):
        """Find ALL valid player candidates. Returns [(addr, x, y, type), ...]."""
        candidates = []
        h = self.pm.process_handle
        k32 = ctypes.windll.kernel32
        # 在堆区间 0x0E000000~0x50000000 内逐块扫描（游戏对象通常分配在这个范围）
        a = 0x0E000000
        while a < 0x50000000:
            m = MBI()
            if k32.VirtualQueryEx(ctypes.c_void_p(h), ctypes.c_void_p(a), ctypes.byref(m), ctypes.sizeof(m)) == 0: break
            # state=0x1000(MEM_COMMIT) 且可读，块大小在合理区间才扫
            if m.state == 0x1000 and m.rprot in {2,4,8,0x40,0x80} and 0x10000 < m.size < 16000000:
                try:
                    data = self.pm.read_bytes(m.base, min(m.size, 4*1024*1024))
                    # 每 4 字节看一个候选：类型 ID 必须是玩家类型(1/2/26)
                    for off in range(0, len(data)-48, 4):
                        if data[off] not in (1,2,26): continue
                        try:
                            tid = struct.unpack_from('<I', data, off)[0]
                            if tid not in (1,2,26): continue
                            ptr = struct.unpack_from('<I', data, off+4)[0]
                            if not (0x0E000000 < ptr < 0x50000000): continue
                            x = struct.unpack_from('<f', data, off+32)[0]
                            y = struct.unpack_from('<f', data, off+36)[0]
                            # 坐标必须在合理地图范围内，过滤垃圾数据
                            if not (10 < x < 10000 and 10 < y < 5000): continue
                            candidates.append((m.base+off, x, y, tid))
                            if len(candidates) >= max_n: break
                        except: continue
                except: pass
            a = m.base + m.size
        return candidates

    def locate_player(self):
        """**不要再用这个**：它取候选列表的第一个，而实测那个结构体坐标永远不动
        （diag_world.py：按 right/left/up，world Δ 全是 +0.00），拿它当锚点算出来的
        目标位置全是错的。改用 scan_player_addrs() + 按键前后 read_many() 比较，
        让「谁跟着按键动就是谁」那套绑定（bot.py 的 _bind_mem_player）。保留只为兼容旧调用。"""
        candidates = self.find_all_players(30)
        if not candidates: return False
        self.player_base = candidates[0][0]
        return True

    def scan_player_addrs(self, max_n=220):
        """扫一遍堆，收集所有「看起来像玩家」的结构体地址（type 1/2/26 且坐标在合理
        范围）。这一趟实测 2~3 秒，很贵，所以只在需要重新绑定时调用；之后每帧用
        read_many() 重读这些地址（几十个地址 <1ms）。"""
        return [a for a, _x, _y, _t in self.find_all_players(max_n)]

    def read_many(self, addrs):
        """重读一批已知地址，返回 {addr: (x, y, type)}；读不到/越界的直接丢弃。

        原来每帧调用 cheeses()/holes() 各要全堆扫一遍（实测各 1.1s，一帧就废了），
        现在改成扫描一次拿地址、之后只重读。"""
        out = {}
        if not self.pm:
            return out
        for a in addrs:
            try:
                raw = self.pm.read_bytes(a, 48)
            except Exception:
                continue
            x = struct.unpack_from('<f', raw, PLAYER_X_OFFSET)[0]
            y = struct.unpack_from('<f', raw, PLAYER_Y_OFFSET)[0]
            t = struct.unpack_from('<I', raw, PLAYER_TYPE_OFFSET)[0]
            if 10 < x < 10000 and 10 < y < 5000:
                out[a] = (x, y, t)
        return out

    def scan_object_addrs(self, types, max_n=60):
        """一次扫堆，按类型 ID 收集对象地址，返回 {type: [addr, ...]}。

        芝士(30/3)和洞(94/68/7)原本各扫一遍、每帧都扫，加起来两秒多；合并成一次
        扫描后每帧只重读地址。传进来的 types 里每个类型最多收 max_n 个。"""
        res = {t: [] for t in types}
        if not self.pm:
            return res
        h = self.pm.process_handle
        k32 = ctypes.windll.kernel32
        a = 0x0E000000
        while a < 0x50000000:
            m = MBI()
            if k32.VirtualQueryEx(ctypes.c_void_p(h), ctypes.c_void_p(a), ctypes.byref(m), ctypes.sizeof(m)) == 0:
                break
            if m.state == 0x1000 and m.rprot in {2, 4, 8, 0x40, 0x80} and m.size >= 0x1000:
                try:
                    data = self.pm.read_bytes(m.base, min(m.size, 2 * 1024 * 1024))
                    for off in range(0, len(data) - 48, 16):
                        t = struct.unpack_from('<I', data, off)[0]
                        if t in res and len(res[t]) < max_n:
                            x = struct.unpack_from('<f', data, off + 32)[0]
                            y = struct.unpack_from('<f', data, off + 36)[0]
                            if 10 < x < 10000 and 10 < y < 5000:
                                res[t].append(m.base + off)
                except Exception:
                    pass
            a = m.base + m.size
        return res

    def read_player(self):
        if not self.pm or not self.player_base: return None
        try:
            raw = self.pm.read_bytes(self.player_base, 48)
            x = struct.unpack_from('<f', raw, PLAYER_X_OFFSET)[0]
            y = struct.unpack_from('<f', raw, PLAYER_Y_OFFSET)[0]
            tid = struct.unpack_from('<I', raw, PLAYER_TYPE_OFFSET)[0]
            if 10 < x < 10000 and 10 < y < 5000 and tid in (1,2,26):
                return (x, y, tid)
        except: pass

    def read_objects(self, target_type, max_n=50):
        """Scan heap for objects of given type. Returns [(x,y), ...]."""
        if not self.pm: return []
        res = []
        h = self.pm.process_handle
        k32 = ctypes.windll.kernel32
        # 同 find_all_players，但按给定类型 ID 找对象（芝士/洞/平台），步长 16（对象对齐更粗）
        a = 0x0E000000
        while a < 0x50000000 and len(res) < max_n:
            m = MBI()
            if k32.VirtualQueryEx(ctypes.c_void_p(h), ctypes.c_void_p(a), ctypes.byref(m), ctypes.sizeof(m)) == 0: break
            if m.state == 0x1000 and m.rprot in {2,4,8,0x40,0x80} and m.size >= 0x1000:
                try:
                    data = self.pm.read_bytes(m.base, min(m.size, 2*1024*1024))
                    for off in range(0, len(data)-48, 16):
                        try:
                            if struct.unpack_from('<I', data, off)[0] == target_type:
                                x = struct.unpack_from('<f', data, off+32)[0]
                                y = struct.unpack_from('<f', data, off+36)[0]
                                if 10 < x < 10000 and 10 < y < 5000:
                                    res.append((x, y))
                                    if len(res) >= max_n: break
                        except: continue
                except: pass
            a = m.base + m.size
        return res

    def cheeses(self):
        # 芝士对象类型 ID：优先 30，备选 3；返回数量在 1~5 之间才算有效
        for tid in [30, 3]:
            res = self.read_objects(tid, 10)
            if 1 <= len(res) <= 5: return res
        return self.read_objects(30, 10)

    def holes(self):
        # 洞口对象类型 ID：优先 94，备选 68/7
        for tid in [94, 68, 7]:
            res = self.read_objects(tid, 10)
            if 1 <= len(res) <= 5: return res
        return self.read_objects(94, 10)

    def platforms(self):
        """Read platform positions. Types 1,4,5,6 are ground/platform objects."""
        # 平台/地面对象类型 ID：1/4/5/6
        res = []
        for tid in [1, 4, 5, 6]:
            res.extend(self.read_objects(tid, 20))
        return res
