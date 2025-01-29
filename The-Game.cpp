/**
 *    author:  tourist
 *    created: 26.01.2025 07:05:24
**/
#include <bits/stdc++.h>

using namespace std;

#ifdef LOCAL
#include "algo/debug.h"
#else
#define debug(...) 42
#endif

template <typename T>
class graph {
 public:
  struct edge {
    int from;
    int to;
    T cost;
  };

  vector<edge> edges;
  vector<vector<int>> g;
  int n;

  graph(int _n) : n(_n) {
    g.resize(n);
  }

  virtual int add(int from, int to, T cost) = 0;
};

template <typename T>
class forest : public graph<T> {
 public:
  using graph<T>::edges;
  using graph<T>::g;
  using graph<T>::n;

  forest(int _n) : graph<T>(_n) {
  }

  int add(int from, int to, T cost = 1) {
    assert(0 <= from && from < n && 0 <= to && to < n);
    int id = (int) edges.size();
    assert(id < n - 1);
    g[from].push_back(id);
    g[to].push_back(id);
    edges.push_back({from, to, cost});
    return id;
  }
};

template <typename T>
class dfs_forest : public forest<T> {
 public:
  using forest<T>::edges;
  using forest<T>::g;
  using forest<T>::n;

  vector<int> pv;
  vector<int> pe;
  vector<int> order;
  vector<int> pos;
  vector<int> end;
  vector<int> sz;
  vector<int> root;
  vector<int> depth;
  vector<T> dist;

  dfs_forest(int _n) : forest<T>(_n) {
  }

  void init() {
    pv = vector<int>(n, -1);
    pe = vector<int>(n, -1);
    order.clear();
    pos = vector<int>(n, -1);
    end = vector<int>(n, -1);
    sz = vector<int>(n, 0);
    root = vector<int>(n, -1);
    depth = vector<int>(n, -1);
    dist = vector<T>(n);
  }

  void clear() {
    pv.clear();
    pe.clear();
    order.clear();
    pos.clear();
    end.clear();
    sz.clear();
    root.clear();
    depth.clear();
    dist.clear();
  }

 private:
  void do_dfs(int v) {
    pos[v] = (int) order.size();
    order.push_back(v);
    sz[v] = 1;
    for (int id : g[v]) {
      if (id == pe[v]) {
        continue;
      }
      auto &e = edges[id];
      int to = e.from ^ e.to ^ v;
      depth[to] = depth[v] + 1;
      dist[to] = dist[v] + e.cost;
      pv[to] = v;
      pe[to] = id;
      root[to] = (root[v] != -1 ? root[v] : to);
      do_dfs(to);
      sz[v] += sz[to];
    }
    end[v] = (int) order.size() - 1;
  }

  void do_dfs_from(int v) {
    depth[v] = 0;
    dist[v] = T{};
    root[v] = v;
    pv[v] = pe[v] = -1;
    do_dfs(v);
  }

 public:
  void dfs(int v, bool clear_order = true) {
    if (pv.empty()) {
      init();
    } else {
      if (clear_order) {
        order.clear();
      }
    }
    do_dfs_from(v);
  }

  void dfs_all() {
    init();
    for (int v = 0; v < n; v++) {
      if (depth[v] == -1) {
        do_dfs_from(v);
      }
    }
    assert((int) order.size() == n);
  }
};

template <typename T>
class hld_forest : public dfs_forest<T> {
 public:
  using dfs_forest<T>::edges;
  using dfs_forest<T>::g;
  using dfs_forest<T>::n;
  using dfs_forest<T>::pv;
  using dfs_forest<T>::sz;
  using dfs_forest<T>::root;
  using dfs_forest<T>::pos;
  using dfs_forest<T>::end;
  using dfs_forest<T>::order;
  using dfs_forest<T>::depth;
  using dfs_forest<T>::dfs;
  using dfs_forest<T>::dfs_all;

  vector<int> head;
  vector<int> visited;

  hld_forest(int _n) : dfs_forest<T>(_n) {
    visited.resize(n);
  }

  void build_hld(const vector<int> &vs) {
    for (int tries = 0; tries < 2; tries++) {
      if (vs.empty()) {
        dfs_all();
      } else {
        order.clear();
        for (int v : vs) {
          dfs(v, false);
        }
        assert((int) order.size() == n);
      }
      if (tries == 1) {
        break;
      }
      for (int i = 0; i < n; i++) {
        if (g[i].empty()) {
          continue;
        }
        int best = -1, bid = 0;
        for (int j = 0; j < (int) g[i].size(); j++) {
          int id = g[i][j];
          int v = edges[id].from ^ edges[id].to ^ i;
          if (pv[v] != i) {
            continue;
          }
          if (sz[v] > best) {
            best = sz[v];
            bid = j;
          }
        }
        swap(g[i][0], g[i][bid]);
      }
    }
    head.resize(n);
    for (int i = 0; i < n; i++) {
      head[i] = i;
    }
    for (int i = 0; i < n - 1; i++) {
      int x = order[i];
      int y = order[i + 1];
      if (pv[y] == x) {
        head[y] = head[x];
      }
    }
  }

  void build_hld(int v) {
    build_hld(vector<int>(1, v));
  }

  void build_hld_all() {
    build_hld(vector<int>());
  }

  bool apply_on_path(int x, int y, bool with_lca, function<void(int,int,bool)> f) {
    // f(x, y, up): up -- whether this part of the path goes up
    assert(!head.empty());
    int z = lca(x, y);
    if (z == -1) {
      return false;
    }
    {
      int v = x;
      while (v != z) {
        if (depth[head[v]] <= depth[z]) {
          f(pos[z] + 1, pos[v], true);
          break;
        }
        f(pos[head[v]], pos[v], true);
        v = pv[head[v]];
      }
    }
    if (with_lca) {
      f(pos[z], pos[z], false);
    }
    {
      int v = y;
      int cnt_visited = 0;
      while (v != z) {
        if (depth[head[v]] <= depth[z]) {
          f(pos[z] + 1, pos[v], false);
          break;
        }
        visited[cnt_visited++] = v;
        v = pv[head[v]];
      }
      for (int at = cnt_visited - 1; at >= 0; at--) {
        v = visited[at];
        f(pos[head[v]], pos[v], false);
      }
    }
    return true;
  }

  inline bool anc(int x, int y) {
    return (pos[x] <= pos[y] && end[y] <= end[x]);
  }

  inline int go_up(int x, int up) {
    int target = depth[x] - up;
    if (target < 0) {
      return -1;
    }
    while (depth[head[x]] > target) {
      x = pv[head[x]];
    }
    return order[pos[x] - depth[x] + target];
  }

  inline int lca(int x, int y) {
    if (root[x] != root[y]) {
      return -1;
    }
    while (head[x] != head[y]) {
      if (depth[head[x]] > depth[head[y]]) {
        x = pv[head[x]];
      } else {
        y = pv[head[y]];
      }
    }
    return depth[x] < depth[y] ? x : y;
  }
};

namespace seg_tree {

// Floor of log_2(a); index of highest 1-bit
inline int floor_log_2(int a) {
  return a ? bit_width(unsigned(a)) - 1 : -1;
}

struct point {
  int a;
  point() : a(0) {}
  explicit point(int a_) : a(a_) { assert(a >= -1); }

  explicit operator bool () { return bool(a); }

  // This is useful so you can directly do array indices
  /* implicit */ operator int() const { return a; }

  point c(bool z) const {
    return point((a << 1) | z);
  }

  point operator [] (bool z) const {
    return c(z);
  }

  point p() const {
    return point(a >> 1);
  }

  friend std::ostream& operator << (std::ostream& o, const point& p) { return o << int(p); }

  template <typename F> void for_each(F f) const {
    for (int v = a; v > 0; v >>= 1) {
      f(point(v));
    }
  }

  template <typename F> void for_parents_down(F f) const {
    // strictly greater than 0
    for (int L = floor_log_2(a); L > 0; L--) {
      f(point(a >> L));
    }
  }

  template <typename F> void for_parents_up(F f) const {
    for (int v = a >> 1; v > 0; v >>= 1) {
      f(point(v));
    }
  }

  point& operator ++ () { ++a; return *this; }
  point operator ++ (int) { return point(a++); }
  point& operator -- () { --a; return *this; }
  point operator -- (int) { return point(a--); }
};

struct range {
  int a, b;
  range() : a(1), b(1) {}
  range(int a_, int b_) : a(a_), b(b_) {
    assert(1 <= a && a <= b && b <= 2 * a);
  }
  explicit range(std::array<int, 2> r) : range(r[0], r[1]) {}

  explicit operator std::array<int, 2>() const {
    return {a, b};
  }

  const int& operator[] (bool z) const {
    return z ? b : a;
  }

  friend std::ostream& operator << (std::ostream& o, const range& r) { return o << "[" << r.a << ".." << r.b << ")"; }

  // Iterate over the range from outside-in.
  //   Calls f(point a)
  template <typename F> void for_each(F f) const {
    for (int x = a, y = b; x < y; x >>= 1, y >>= 1) {
      if (x & 1) f(point(x++));
      if (y & 1) f(point(--y));
    }
  }

  // Iterate over the range from outside-in.
  //   Calls f(point a, bool is_right)
  template <typename F> void for_each_with_side(F f) const {
    for (int x = a, y = b; x < y; x >>= 1, y >>= 1) {
      if (x & 1) f(point(x++), false);
      if (y & 1) f(point(--y), true);
    }
  }

  // Iterate over the range from left to right.
  //    Calls f(point)
  template <typename F> void for_each_l_to_r(F f) const {
    int anc_depth = floor_log_2((a - 1) ^ b);
    int anc_msk = (1 << anc_depth) - 1;
    for (int v = (-a) & anc_msk; v; v &= v - 1) {
      int i = countr_zero(unsigned(v));
      f(point(((a - 1) >> i) + 1));
    }
    for (int v = b & anc_msk; v; ) {
      int i = floor_log_2(v);
      f(point((b >> i) - 1));
      v ^= (1 << i);
    }
  }

  // Iterate over the range from right to left.
  //    Calls f(point)
  template <typename F> void for_each_r_to_l(F f) const {
    int anc_depth = floor_log_2((a - 1) ^ b);
    int anc_msk = (1 << anc_depth) - 1;
    for (int v = b & anc_msk; v; v &= v - 1) {
      int i = countr_zero(unsigned(v));
      f(point((b >> i) - 1));
    }
    for (int v = (-a) & anc_msk; v; ) {
      int i = floor_log_2(v);
      f(point(((a - 1) >> i) + 1));
      v ^= (1 << i);
    }
  }

  template <typename F> void for_parents_down(F f) const {
    int x = a, y = b;
    if ((x ^ y) > x) { x <<= 1, std::swap(x, y); }
    int dx = countr_zero(unsigned(x));
    int dy = countr_zero(unsigned(y));
    int anc_depth = floor_log_2((x - 1) ^ y);
    for (int i = floor_log_2(x); i > dx; i--) {
      f(point(x >> i));
    }
    for (int i = anc_depth; i > dy; i--) {
      f(point(y >> i));
    }
  }

  template <typename F> void for_parents_up(F f) const {
    int x = a, y = b;
    if ((x ^ y) > x) { x <<= 1, std::swap(x, y); }
    int dx = countr_zero(unsigned(x));
    int dy = countr_zero(unsigned(y));
    int anc_depth = floor_log_2((x - 1) ^ y);
    for (int i = dx + 1; i <= anc_depth; i++) {
      f(point(x >> i));
    }
    for (int v = y >> (dy + 1); v; v >>= 1) {
      f(point(v));
    }
  }
};

struct in_order_layout {
  // Alias them in for convenience
  using point = seg_tree::point;
  using range = seg_tree::range;

  int n, s;
  in_order_layout() : n(0), s(0) {}
  in_order_layout(int n_) : n(n_), s(n ? bit_ceil(unsigned(n)) : 0) {}

  point get_point(int a) const {
    assert(0 <= a && a < n);
    a += s;
    return point(a >= 2 * n ? a - n : a);
  }

  range get_range(int a, int b) const {
    assert(0 <= a && a <= b && b <= n);
    if (n == 0) return range();
    a += s, b += s;
    return range((a >= 2 * n ? 2 * (a - n) : a), (b >= 2 * n ? 2 * (b - n) : b));
  }

  range get_range(std::array<int, 2> p) const {
    return get_range(p[0], p[1]);
  }

  int get_leaf_index(point pt) const {
    int a = int(pt);
    assert(n <= a && a < 2 * n);
    return (a < s ? a + n : a) - s;
  }

  std::array<int, 2> get_node_bounds(point pt) const {
    int a = int(pt);
    assert(1 <= a && a < 2 * n);
    int l = countl_zero(unsigned(a)) - countl_zero(unsigned(2 * n - 1));
    int x = a << l, y = (a + 1) << l;
    assert(s <= x && x < y && y <= 2 * s);
    return {(x >= 2 * n ? (x >> 1) + n : x) - s, (y >= 2 * n ? (y >> 1) + n : y) - s};
  }

  int get_node_split(point pt) const {
    int a = int(pt);
    assert(1 <= a && a < n);
    int l = (unsigned(2 * a + 1)) - countl_zero(unsigned(2 * n - 1));
    int x = (2 * a + 1) << l;
    assert(s <= x && x < 2 * s);
    return (x >= 2 * n ? (x >> 1) + n : x) - s;
  }

  int get_node_size(point pt) const {
    auto bounds = get_node_bounds(pt);
    return bounds[1] - bounds[0];
  }
};

struct circular_layout {
  // Alias them in for convenience
  using point = seg_tree::point;
  using range = seg_tree::range;

  int n;
  circular_layout() : n(0) {}
  circular_layout(int n_) : n(n_) {}

  point get_point(int a) const {
    assert(0 <= a && a < n);
    return point(n + a);
  }

  range get_range(int a, int b) const {
    assert(0 <= a && a <= b && b <= n);
    if (n == 0) return range();
    return range(n + a, n + b);
  }

  range get_range(std::array<int, 2> p) const {
    return get_range(p[0], p[1]);
  }

  int get_leaf_index(point pt) const {
    int a = int(pt);
    assert(n <= a && a < 2 * n);
    return a - n;
  }

  // Returns {x,y} so that 0 <= x < n and 1 <= y <= n
  // If the point is non-wrapping, then 0 <= x < y <= n
  std::array<int, 2> get_node_bounds(point pt) const {
    int a = int(pt);
    assert(1 <= a && a < 2 * n);
    int l = countl_zero(unsigned(a)) - countl_zero(unsigned(2 * n - 1));
    int s = bit_ceil(unsigned(n));
    int x = a << l, y = (a + 1) << l;
    assert(s <= x && x < y && y <= 2 * s);
    return {(x >= 2 * n ? x >> 1 : x) - n, (y > 2 * n ? y >> 1 : y) - n};
  }

  // Returns the split point of the node, such that 1 <= s <= n.
  int get_node_split(point pt) const {
    int a = int(pt);
    assert(1 <= a && a < n);
    return get_node_bounds(pt.c(0))[1];
  }

  int get_node_size(point pt) const {
    auto bounds = get_node_bounds(pt);
    int r = bounds[1] - bounds[0];
    return r > 0 ? r : r + n;
  }
};

} // namespace seg_tree

template <typename Info>
class SimpleSegmentTree {
 public:
  int n;
  vector<Info> infos;
  seg_tree::in_order_layout layout;

  void UpdateNode(seg_tree::point a) {
    infos[a] = infos[a.c(0)].Unite(infos[a.c(1)]);
  }

  SimpleSegmentTree(int n_) : SimpleSegmentTree(vector<Info>(n_)) {}

  SimpleSegmentTree(const vector<Info>& a) : n(int(a.size())) {
    assert(n > 0);
    infos.resize(2 * n);
    layout = seg_tree::in_order_layout(n);
    for (int i = 0; i < n; i++) {
      infos[layout.get_point(i)] = a[i];
    }
    for (int i = n - 1; i >= 1; i--) {
      infos[i] = infos[2 * i].Unite(infos[2 * i + 1]);
    }
  }

  void Set(int p, const Info& v) {
    auto pt = layout.get_point(p);
    infos[pt] = v;
    pt.for_parents_up([&](seg_tree::point a) {
      UpdateNode(a);
    });
  }

  Info Query(int l, int r) {
    auto rng = layout.get_range(l, r);
    Info res;
    rng.for_each_l_to_r([&](seg_tree::point a) {
      res = res.Unite(infos[a]);
    });
    return res;
  }

  Info Get(int p) {
    auto pt = layout.get_point(p);
    return infos[pt];
  }

  template<typename F>
  int MaxRight(int l, F f) {
    auto rng = layout.get_range(l, n);
    int res = n;
    Info sum;
    rng.for_each_l_to_r([&](seg_tree::point a) {
      if (res != n) {
        return;
      }
      auto new_sum = sum.Unite(infos[a]);
      if (f(new_sum)) {
        sum = new_sum;
        return;
      }
      while (a < n) {
        new_sum = sum.Unite(infos[a.c(0)]);
        if (f(new_sum)) {
          sum = new_sum;
          a = a.c(1);
        } else {
          a = a.c(0);
        }
      }
      res = layout.get_node_bounds(a)[0];
    });
    return res;
  }

  template<typename F>
  int MinLeft(int r, F f) {
    auto rng = layout.get_range(0, r);
    int res = 0;
    Info sum;
    rng.for_each_r_to_l([&](seg_tree::point a) {
      if (res != 0) {
        return;
      }
      auto new_sum = infos[a].Unite(sum);
      if (f(new_sum)) {
        sum = new_sum;
        return;
      }
      while (a < n) {
        new_sum = infos[a.c(1)].Unite(sum);
        if (f(new_sum)) {
          sum = new_sum;
          a = a.c(0);
        } else {
          a = a.c(1);
        }
      }
      res = layout.get_node_bounds(a)[1];
    });
    return res;
  }
};

const int inf = int(1e9);

struct Info {
  int mx = -1;
  int l = inf;
  int r = -1;
  int cnt = 0;

  Info Unite(const Info& b) const {
    Info res;
    res.mx = max(mx, b.mx);
    res.l = min(res.mx == mx ? l : inf, res.mx == b.mx ? b.l : inf);
    res.r = max(res.mx == mx ? r : -1, res.mx == b.mx ? b.r : -1);
    res.cnt = (res.mx == mx ? cnt : 0) + (res.mx == b.mx ? b.cnt : 0);
    return res;
  }

  static Info GetDefault([[maybe_unused]] int l, [[maybe_unused]] int r) {
    return Info();
  }
};

template <typename T>
class FenwickTree {
 public:
  vector<T> fenw;
  int n;
  int pw;

  FenwickTree() : n(0) {}
  FenwickTree(int n_) : n(n_) {
    fenw.resize(n);
    pw = bit_floor(unsigned(n));
  }

  void Modify(int x, T v) {
    assert(0 <= x && x <= n);
    while (x < n) {
      fenw[x] += v;
      x |= x + 1;
    }
  }

  T Query(int x) {
    assert(0 <= x && x <= n);
    T v{};
    while (x > 0) {
      v += fenw[x - 1];
      x &= x - 1;
    }
    return v;
  }

  // Returns the length of the longest prefix with sum <= c
  int MaxPrefix(T c) {
    T v{};
    int at = 0;
    for (int len = pw; len > 0; len >>= 1) {
      if (at + len <= n) {
        auto nv = v;
        nv += fenw[at + len - 1];
        if (!(c < nv)) {
          v = nv;
          at += len;
        }
      }
    }
    assert(0 <= at && at <= n);
    return at;
  }
};

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  int tt;
  cin >> tt;
  while (tt--) {
    int n;
    cin >> n;
    vector<int> w(n);
    for (int i = 0; i < n; i++) {
      cin >> w[i];
      --w[i];
    }
    hld_forest<int> g(n);
    for (int i = 0; i < n - 1; i++) {
      int x, y;
      cin >> x >> y;
      --x; --y;
      g.add(x, y);
    }
    g.build_hld(0);
    vector<vector<int>> at(n);
    for (int i = 0; i < n; i++) {
      at[w[i]].push_back(i);
    }
    vector<bool> bad(n, false);
    SimpleSegmentTree<Info> st(n);
    FenwickTree<int> fenw(n);
    int l = n + 1;
    int r = -1;
    int cnt = 0;
    for (int it = n - 1; it >= 0; it--) {
      for (int v : at[it]) {
        if (l < g.pos[v] || r > g.end[v]) {
          bad[v] = true;
          continue;
        }
        auto got = st.Query(0, g.pos[v]).Unite(st.Query(g.end[v] + 1, n));
        if (got.mx == -1) {
          bad[v] = true;
          continue;
        }
        int u = g.lca(g.order[got.l], g.order[got.r]);
        int ins = fenw.Query(g.end[u] + 1) - fenw.Query(g.pos[u] + 1);
        {
          int ll = max(g.pos[v], g.pos[u] + 1);
          int rr = min(g.end[v], g.end[u]);
          if (ll <= rr) {
            ins -= fenw.Query(rr + 1) - fenw.Query(ll);
          }
        }
        int total = ins;
        g.apply_on_path(0, u, true, [&](int i, int j, bool) {
          total += fenw.Query(j + 1) - fenw.Query(i);
        });
        int sub = fenw.Query(g.end[v] + 1) - fenw.Query(g.pos[v]);
        if (total != cnt - sub) {
          bad[v] = true;
          continue;
        }
        ins += int(w[u] == got.mx);
        if (ins != got.cnt) {
          bad[v] = true;
          continue;
        }
      }
      for (int v : at[it]) {
        int sum = 0;
        g.apply_on_path(0, v, true, [&](int i, int j, bool) {
          sum += fenw.Query(j + 1) - fenw.Query(i);
        });
        if (sum > 0) {
          l = min(l, g.pos[v]);
          r = max(r, g.end[v]);
        }
      }
      for (int v : at[it]) {
        fenw.Modify(g.pos[v], +1);
        st.Set(g.pos[v], {it, g.pos[v], g.pos[v], 1});
        cnt += 1;
      }
    }
    vector<int> ans;
    for (int i = 1; i < n; i++) {
      if (!bad[i]) {
        ans.push_back(i);
      }
    }
    cout << ans.size();
    for (int v : ans) {
      cout << " " << v + 1;
    }
    cout << '\n';
  }
  return 0;
}
