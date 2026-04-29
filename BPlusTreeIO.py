"""
B+ Tree mit simuliertem Buffer-Pool (Disk-I/O).

Erweiterung der reinen In-Memory-Variante aus BPlusTree.py:
  * Knoten werden nicht mehr direkt verlinkt, sondern über page_ids.
  * Jeder Knotenzugriff läuft über den BufferManager.
  * Der BufferManager hält genau `capacity` Seiten im RAM;
    überzählige Seiten werden nach LRU auf "Platte" ausgelagert
    ("Platte" = weiteres dict, keine echte Serialisierung):
    Wir wollen nur die Anzahl der nötigen Plattenzugriffe messen, 
    weder die Kosten der Serialisierung noch die Performance der
    (Platten-) Caches oder gar der Massenspicher selber. Das schätzen wir
    lieber ab.

Zwei Stellschrauben für den Benchmark:
  * page_size         → steuert Fanout und Baumhöhe
  * buffer_capacity   → steuert die Hit-Rate im Puffer

Gezählt werden:
  * page_accesses     logische Seitenzugriffe (entspricht dem
                      node_accesses der In-Memory-Variante)
  * disk_reads        Buffer-Misses (kostet I/O)
  * disk_writes       dirty-evictions (nur modifizierte Seiten
                      verursachen einen Write-Back)

Kein Pinning: während einer Operation wird die "aktuelle" Seite durch
das move_to_end in get() automatisch ans jüngste LRU-Ende verschoben
und damit nicht evictet. Für buffer_capacity >= ~8 ist das ausreichend;
für Produktivsysteme müsste man echtes Pinning implementieren.

Autor: Claude.ai
"""

from bisect import bisect_left, bisect_right
from collections import OrderedDict


class Node:
    __slots__ = ("is_leaf", "keys", "children", "values", "next_pid")

    def __init__(self, is_leaf: bool):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []     # internal: list of page_ids
        self.values = []       # leaf: list of values
        self.next_pid = None   # leaf: page_id des nächsten Blattes


# =====================================================================
#   BUFFER MANAGER
# =====================================================================
class BufferManager:
    """
    Einfacher LRU-Buffer-Pool.

    "Platte" ist lediglich ein weiteres dict; uns geht es nur um die
    Bewegung von Seiten zwischen Puffer und "Disk", nicht um echte
    Serialisierung. Write-Back erfolgt nur bei dirty-evictions.
    """

    def __init__(self, capacity: int):
        assert capacity >= 4, "capacity sollte >= tree height sein (praktisch >=8)"
        self.capacity = capacity
        self.buffer = OrderedDict()   # pid -> Node (LRU-geordnet, ältester zuerst)
        self.disk = {}                 # pid -> Node (ausgelagerte Seiten)
        self.dirty = set()             # pids mit geänderter Buffer-Kopie
        self._next_id = 0
        # --- Zähler -----------------------------------------------
        self.page_accesses = 0
        self.disk_reads = 0
        self.disk_writes = 0

    def new_page(self, node) -> int:
        """Neue Seite anlegen. Landet sofort im Buffer und gilt als dirty."""
        pid = self._next_id
        self._next_id += 1
        self._ensure_space() # Buffer evicten, falls voll
        self.buffer[pid] = node
        self.dirty.add(pid)
        return pid

    def get(self, pid) -> "Node":
        """Seite holen. Bei Miss: von Platte laden (zählt als disk_read)."""
        self.page_accesses += 1
        if pid in self.buffer:
            self.buffer.move_to_end(pid)   # LRU: frisch benutzt
            return self.buffer[pid]
        # Miss --> von Platte holen
        self.disk_reads += 1
        node = self.disk.pop(pid)
        self._ensure_space()
        self.buffer[pid] = node
        return node

    def mark_dirty(self, pid):
        """Änderung auf Seite pid merken (→ wird beim Evict geschrieben)."""
        self.dirty.add(pid)

    def _ensure_space(self):
        """Wenn Buffer voll: LRU-Seite auslagern (dirty --> Write-Back)."""
        if len(self.buffer) < self.capacity:
            return
        victim_pid, victim = self.buffer.popitem(last=False)
        if victim_pid in self.dirty:
            self.disk_writes += 1
            self.dirty.discard(victim_pid)
        self.disk[victim_pid] = victim

    def reset_stats(self):
        self.page_accesses = 0
        self.disk_reads = 0
        self.disk_writes = 0

    def total_pages(self) -> int:
        """Wie viele Seiten existieren insgesamt (Buffer + Disk)?"""
        return self._next_id


# =====================================================================
#   B+ TREE
# =====================================================================
class BPlusTree:
    def __init__(self, page_size: int = 64, buffer_capacity: int = 1024):
        assert page_size >= 3
        self.page_size = page_size
        self.bm = BufferManager(buffer_capacity)
        self.root_pid = self.bm.new_page(Node(is_leaf=True))

    # ----------------------------------------------------------------
    #   SEARCH
    # ----------------------------------------------------------------
    def _find_leaf(self, key):
        """Abstieg bis zum Blatt. Gibt (pid, node) zurück."""
        pid = self.root_pid
        node = self.bm.get(pid)
        while not node.is_leaf:
            i = bisect_right(node.keys, key)
            pid = node.children[i]
            node = self.bm.get(pid)
        return pid, node

    def search(self, key):
        _, leaf = self._find_leaf(key)
        i = bisect_left(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            return leaf.values[i]
        return None

    def range_search(self, lo, hi):
        result = []
        _, node = self._find_leaf(lo)
        while True:
            for k, v in zip(node.keys, node.values):
                if k < lo:
                    continue
                if k > hi:
                    return result
                result.append((k, v))
            if node.next_pid is None:
                return result
            node = self.bm.get(node.next_pid)

    # ----------------------------------------------------------------
    #   INSERT
    # ----------------------------------------------------------------
    def insert(self, key, value):
        # Abstieg, Pfad aus (parent_pid, child_idx) merken.
        path = []
        pid = self.root_pid
        node = self.bm.get(pid)
        while not node.is_leaf:
            i = bisect_right(node.keys, key)
            path.append((pid, i))
            pid = node.children[i]
            node = self.bm.get(pid)

        # Blatt-Insert (Duplikat → überschreiben).
        i = bisect_left(node.keys, key)
        if i < len(node.keys) and node.keys[i] == key:
            node.values[i] = value
            self.bm.mark_dirty(pid)
            return
        node.keys.insert(i, key)
        node.values.insert(i, value)
        self.bm.mark_dirty(pid)

        # Overflow? Dann nach oben propagieren.
        if len(node.keys) <= self.page_size:
            return

        sep, new_pid = self._split(pid, node)
        while path:
            parent_pid, child_idx = path.pop()
            parent = self.bm.get(parent_pid)
            parent.keys.insert(child_idx, sep)
            parent.children.insert(child_idx + 1, new_pid)
            self.bm.mark_dirty(parent_pid)
            if len(parent.keys) <= self.page_size:
                return
            sep, new_pid = self._split(parent_pid, parent)

        # Split der Wurzel → neue Wurzel anlegen.
        new_root = Node(is_leaf=False)
        new_root.keys = [sep]
        new_root.children = [self.root_pid, new_pid]
        self.root_pid = self.bm.new_page(new_root)

    def _split(self, pid, node):
        mid = len(node.keys) // 2
        right = Node(is_leaf=node.is_leaf)
        if node.is_leaf:
            # COPY-UP
            right.keys     = node.keys[mid:]
            right.values   = node.values[mid:]
            right.next_pid = node.next_pid
            node.keys      = node.keys[:mid]
            node.values    = node.values[:mid]
            sep = right.keys[0]
            right_pid = self.bm.new_page(right)
            node.next_pid = right_pid
        else:
            # PUSH-UP
            sep = node.keys[mid]
            right.keys     = node.keys[mid + 1:]
            right.children = node.children[mid + 1:]
            node.keys      = node.keys[:mid]
            node.children  = node.children[:mid + 1]
            right_pid = self.bm.new_page(right)
        self.bm.mark_dirty(pid)
        return sep, right_pid

    # ----------------------------------------------------------------
    #   Introspection
    # ----------------------------------------------------------------
    def height(self) -> int:
        pid = self.root_pid
        node = self.bm.get(pid)
        h = 1
        while not node.is_leaf:
            pid = node.children[0]
            node = self.bm.get(pid)
            h += 1
        return h


# =====================================================================
#   BENCHMARK: Effekt der Buffer-Größe
# =====================================================================
if __name__ == "__main__":
    import random, time

    random.seed(42)
    N = 100_000
    PAGE_SIZE = 64
    QUERIES = 10_000

    data = [(random.randint(0, 10_000_000), i) for i in range(N)]
    query_keys = [k for k, _ in random.sample(data, QUERIES)]

    # Erst einmal mit "unbegrenztem" Buffer bauen, um die Gesamt-Seitenzahl
    # und die Anzahl interner Knoten zu ermitteln.
    ref = BPlusTree(page_size=PAGE_SIZE, buffer_capacity=10**9)
    for k, v in data:
        ref.insert(k, v)
    total_pages = ref.bm.total_pages()
    height = ref.height()

    # Interne Knoten zählen (alles außer den Blättern).
    internal = 0
    stack = [ref.root_pid]
    while stack:
        n = ref.bm.get(stack.pop())
        if not n.is_leaf:
            internal += 1
            stack.extend(n.children)

    print(f"Baum: N={N:,}, page_size={PAGE_SIZE}, height={height}, "
          f"pages={total_pages:,} (davon {internal} intern, "
          f"{total_pages - internal} Blätter)\n")

    # Zwei Workloads: random vs. sortiert (zeigt Effekt von Zugriffs-Lokalität)
    sorted_keys = sorted(query_keys)

    def run(cap, keys):
        t = BPlusTree(page_size=PAGE_SIZE, buffer_capacity=cap)
        for k, v in data:
            t.insert(k, v)
        t.bm.reset_stats()
        for k in keys:
            t.search(k)
        return t.bm.disk_reads / len(keys), t.bm.page_accesses / len(keys)

    hdr = (f"{'capacity':>9} {'% pages':>8} {'pacc/q':>7} "
           f"{'reads/q RND':>12} {'reads/q SEQ':>12}")
    print(hdr)
    print("-" * len(hdr))

    caps = (4, 16, 64, internal + 20, 256, 1024, total_pages + 10)
    for cap in caps:
        r_rnd, pacc = run(cap, query_keys)
        r_seq, _    = run(cap, sorted_keys)
        print(f"{cap:>9} {100*cap/total_pages:>7.1f}% {pacc:>7.2f} "
              f"{r_rnd:>12.2f} {r_seq:>12.2f}")

    print("\nLesehilfe:")
    print("  pacc/q   = logische Seitenzugriffe pro Query (= Baumhöhe, konstant)")
    print("  reads/q  = tatsächliche Disk-Reads pro Query (hängt vom Buffer + Workload ab)")
    print()
    print("  RND (uniform zufällige Keys):")
    print("    * Leaves werden fast nie wiederverwendet → ~1 read pro Query,")
    print("      sobald der Buffer deutlich größer als die Zahl interner Knoten ist.")
    print("    * Erst wenn der gesamte Baum reinpasst, geht reads/q → 0.")
    print("  SEQ (sortierte Keys):")
    print("    * Jedes Blatt hält ~page_size aufeinanderfolgende Keys → ein Read")
    print("      bedient ~page_size Queries → reads/q → 1/page_size auch mit")
    print("      kleinem Buffer. Das ist der klassische B+-Range-Scan-Vorteil.")
