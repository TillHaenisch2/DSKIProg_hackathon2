"""
B+ Tree — didaktische Referenzimplementierung für Performance-Benchmarks.

Zweck: Studierende sollen die Seitengröße (page_size) variieren und beobachten,
wie sich Baumhöhe, Knotenzahl und v.a. die Anzahl der Seitenzugriffe pro
Operation verhalten.

Designprinzipien:
  * Minimalistisch, nicht produktionsreif (kein Locking, kein Delete,
    keine Bulkload-Optimierung, keine Persistenz).
  * Eine Node-Klasse mit is_leaf-Flag → kompakter Code, leicht lesbar.
  * Primäre Benchmark-Metrik: node_accesses (≈ page fetches in echten DBs).
  * Leaves sind als verkettete Liste verknüpft → effiziente Range-Scans.

Autor: Claude.ai
"""

from bisect import bisect_left, bisect_right


class Node:
    """Eine Seite im Baum. Leaf- und Internal-Node teilen sich die Klasse."""
    __slots__ = ("is_leaf", "keys", "children", "values", "next")

    def __init__(self, is_leaf: bool):
        self.is_leaf = is_leaf
        self.keys = []            # sortierte Keys
        self.children = []        # internal: len(keys)+1 Kind-Knoten
        self.values = []          # leaf:     len(keys) Values
        self.next = None          # leaf:     Zeiger auf nächstes Blatt


class BPlusTree:
    """
    Minimaler B+ Baum.

    Parameters
    ----------
    page_size : int
        Maximale Anzahl Keys pro Knoten.  Overflow (und damit Split) tritt
        ein, sobald ein Knoten page_size+1 Keys hätte.  In realen DBs so
        gewählt, dass ein Knoten in eine Disk-Page passt (typ. 4-16 KiB).
    """

    def __init__(self, page_size: int = 64):
        assert page_size >= 3, "page_size muss >= 3 sein"
        self.page_size = page_size
        self.root = Node(is_leaf=True)
        # --- Benchmark-Zähler ---------------------------------------
        self.node_accesses = 0    # +1 bei jedem besuchten Knoten
        self.splits = 0           # Anzahl Node-Splits insgesamt

    # ================================================================
    #   SEARCH
    # ================================================================
    def _find_leaf(self, key) -> Node:
        """Abstieg von der Wurzel bis zum Blatt, das key enthalten müsste."""
        node = self.root
        while not node.is_leaf:
            self.node_accesses += 1
            i = bisect_right(node.keys, key)       # welchem Kind folgen?
            node = node.children[i]
        self.node_accesses += 1
        return node

    def search(self, key) -> object:
        """Point-Query. Gibt den Wert zurück oder None."""
        leaf = self._find_leaf(key)
        i = bisect_left(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            return leaf.values[i]
        return None

    def range_search(self, lo, hi) -> list:
        """Range-Query [lo, hi].  Nutzt die verkettete Blatt-Liste."""
        result = []
        leaf = self._find_leaf(lo)
        while leaf is not None:
            for k, v in zip(leaf.keys, leaf.values):
                if k < lo:
                    continue
                if k > hi:
                    return result
                result.append((k, v))
            leaf = leaf.next
            if leaf is not None:
                self.node_accesses += 1
        return result

    # ================================================================
    #   INSERT
    # ================================================================
    def insert(self, key, value):
        """Einfügen oder Überschreiben (bei gleichem Key)."""
        # Abstieg, Pfad merken (damit wir Splits nach oben propagieren können).
        path = []
        node = self.root
        while not node.is_leaf:
            self.node_accesses += 1
            i = bisect_right(node.keys, key)
            path.append((node, i))
            node = node.children[i]
        self.node_accesses += 1

        # Einfügen auf Blatt-Ebene (bei Duplikat: Wert überschreiben).
        i = bisect_left(node.keys, key)
        if i < len(node.keys) and node.keys[i] == key:
            node.values[i] = value
            return
        node.keys.insert(i, key)
        node.values.insert(i, value)

        # Nach oben durchreichen, solange ein Knoten überläuft.
        if len(node.keys) <= self.page_size:
            return

        new_key, new_node = self._split(node)
        while path:
            parent, child_idx = path.pop()
            parent.keys.insert(child_idx, new_key)
            parent.children.insert(child_idx + 1, new_node)
            if len(parent.keys) <= self.page_size:
                return
            new_key, new_node = self._split(parent)

        # Split der Wurzel → Baum wächst um eine Ebene nach oben.
        new_root = Node(is_leaf=False)
        new_root.keys = [new_key]
        new_root.children = [self.root, new_node]
        self.root = new_root

    def _split(self, node) -> tuple:
        """Teilt node in zwei Hälften. Liefert (Separator-Key, Rechte-Hälfte)."""
        self.splits += 1
        mid = len(node.keys) // 2
        right = Node(is_leaf=node.is_leaf)

        if node.is_leaf:
            # COPY-UP: der erste Key des rechten Blattes dient als Separator,
            # bleibt aber physisch im Blatt (typische B+-Eigenschaft).
            right.keys   = node.keys[mid:]
            right.values = node.values[mid:]
            node.keys    = node.keys[:mid]
            node.values  = node.values[:mid]
            # Blatt-Liste aktuell halten
            right.next = node.next
            node.next  = right
            sep = right.keys[0]
        else:
            # PUSH-UP: der mittlere Key wandert in den Elternknoten.
            sep = node.keys[mid]
            right.keys     = node.keys[mid + 1:]
            right.children = node.children[mid + 1:]
            node.keys      = node.keys[:mid]
            node.children  = node.children[:mid + 1]

        return sep, right

    # ================================================================
    #   INTROSPECTION  (für Benchmarks & Visualisierung)
    # ================================================================
    def height(self) -> int:
        h, n = 1, self.root
        while not n.is_leaf:
            n = n.children[0]
            h += 1
        return h

    def count_nodes(self) -> int:
        total, stack = 0, [self.root]
        while stack:
            n = stack.pop()
            total += 1
            if not n.is_leaf:
                stack.extend(n.children)
        return total

    def reset_stats(self):
        self.node_accesses = 0
        self.splits = 0

    def pretty(self) -> str:
        """Kleiner Pretty-Printer für Debug/Demo (nur für kleine Bäume!)."""
        lines = []
        def walk(node, depth):
            prefix = "  " * depth
            tag = "LEAF" if node.is_leaf else "INT "
            lines.append(f"{prefix}{tag} {node.keys}")
            if not node.is_leaf:
                for c in node.children:
                    walk(c, depth + 1)
        walk(self.root, 0)
        return "\n".join(lines)


# ====================================================================
#   Mini-Benchmark: Effekt der Seitengröße
# ====================================================================
if __name__ == "__main__":
    import random, time

    random.seed(42)
    N = 100_000
    data = [(random.randint(0, 10_000_000), i) for i in range(N)]
    queries = [k for k, _ in random.sample(data, 10_000)]

    hdr = f"{'page_size':>10} {'height':>7} {'nodes':>8} {'splits':>8} " \
          f"{'build[s]':>9} {'lookup[s]':>10} {'avg accesses':>14}"
    print(hdr)
    print("-" * len(hdr))

    for page_size in (4, 8, 16, 32, 64, 128, 256, 512):
        tree = BPlusTree(page_size=page_size)

        t0 = time.perf_counter()
        for k, v in data:
            tree.insert(k, v)
        t_build = time.perf_counter() - t0
        n_splits = tree.splits

        tree.reset_stats()
        t0 = time.perf_counter()
        for k in queries:
            tree.search(k)
        t_lookup = time.perf_counter() - t0
        avg_acc = tree.node_accesses / len(queries)

        print(f"{page_size:>10} {tree.height():>7} {tree.count_nodes():>8} "
              f"{n_splits:>8} {t_build:>9.3f} {t_lookup:>10.3f} "
              f"{avg_acc:>14.2f}")
