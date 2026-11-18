use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

const NONE: usize = usize::MAX;

/// Errors returned by the dense linear assignment auction solver.
#[derive(Debug, Clone, PartialEq)]
pub enum SolverError {
    /// Input matrix dimensions, values, or solver configuration parameters are invalid.
    InvalidInput(String),
    /// Gated complete assignment has no feasible bipartite matching of size `min(n_rows, n_cols)`.
    Infeasible(String),
    /// Solver exceeded `max_iterations` or stalled due to IEEE-754 `f64` ULP precision loss.
    Convergence {
        message: String,
        iterations: usize,
        final_eps: f64,
    },
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::Infeasible(msg) => write!(f, "infeasible problem: {msg}"),
            Self::Convergence {
                message,
                iterations,
                final_eps,
            } => write!(
                f,
                "solver failed to converge after {iterations} iterations (eps={final_eps:.3e}): {message}"
            ),
        }
    }
}

impl Error for SolverError {}

/// Configuration parameters for [`solve_dense`].
#[derive(Debug, Clone, PartialEq)]
pub struct SolveConfig {
    /// If `true`, maximize total cost/score; otherwise minimize total cost.
    pub maximize: bool,
    /// Optional strict upper (minimization) or lower (maximization) edge threshold.
    pub cost_limit: Option<f64>,
    /// Final target epsilon for scaling; defaults to `max(1e-12, scale * 1e-9)`.
    pub eps_min: Option<f64>,
    /// Multiplicative epsilon reduction factor between scaling rounds (`> 1.0`).
    pub eps_factor: f64,
    /// Optional cap on total bid steps across all scaling rounds.
    pub max_iterations: Option<usize>,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            maximize: false,
            cost_limit: None,
            eps_min: None,
            eps_factor: 5.0,
            max_iterations: None,
        }
    }
}

/// Result and diagnostic telemetry returned by [`solve_dense`].
#[derive(Debug, Clone, PartialEq)]
pub struct SolveOutput {
    /// Matched row indices in strictly ascending order.
    pub row_ind: Vec<usize>,
    /// Matched column indices corresponding 1-to-1 with `row_ind`.
    pub col_ind: Vec<usize>,
    /// Exact sum of the original `costs` values over matched pairs.
    pub cost: f64,
    /// Total forward and reverse bids executed across all scaling rounds.
    pub iterations: usize,
    /// Number of epsilon-scaling rounds completed.
    pub scaling_rounds: usize,
    /// Epsilon value of the final completed scaling round.
    pub final_eps: f64,
    /// Final row dual variables (`pi`) aligned with original rows.
    pub row_prices: Vec<f64>,
    /// Final column dual variables (`p`) aligned with original columns.
    pub col_prices: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AssignmentMode {
    Complete,
    Partial(f64),
    Empty,
}

/// Compact row-major CSR representation for matrices with forbidden (`inf`) or sparse partial entries.
struct CsrMatrix {
    offsets: Vec<usize>,
    indices: Vec<usize>,
    vals: Vec<f64>,
}

impl CsrMatrix {
    /// Builds a transposed CSR matrix (column-major adjacency for reverse bidding) in `O(nnz + n)`.
    fn transpose(&self, m: usize, n: usize) -> Self {
        let mut counts = vec![0usize; n + 1];
        for &col in &self.indices {
            counts[col + 1] += 1;
        }
        for c in 0..n {
            counts[c + 1] += counts[c];
        }
        let nnz = self.indices.len();
        let mut t_indices = vec![0usize; nnz];
        let mut t_vals = vec![0.0f64; nnz];
        let mut cursor = counts[..n].to_vec();

        for row in 0..m {
            for pos in self.offsets[row]..self.offsets[row + 1] {
                let col = self.indices[pos];
                let dst = cursor[col];
                t_indices[dst] = row;
                t_vals[dst] = self.vals[pos];
                cursor[col] = dst + 1;
            }
        }

        Self {
            offsets: counts,
            indices: t_indices,
            vals: t_vals,
        }
    }
}

enum BenefitStorage {
    Dense(Vec<f64>),
    Sparse(CsrMatrix),
}

struct NormalizedBenefits {
    storage: BenefitStorage,
    m: usize,
    n: usize,
    transposed: bool,
    b_shift: f64,
    scale: f64,
    half_spread: f64,
    is_constant: bool,
}

struct AuctionState<'a> {
    storage: &'a BenefitStorage,
    sparse_cols: Option<CsrMatrix>,
    m: usize,
    n: usize,
    pi: Vec<f64>,
    p: Vec<f64>,
    row_to_col: Vec<usize>,
    col_to_row: Vec<usize>,
    unassigned_rows: VecDeque<usize>,
    unassigned_cols: VecDeque<usize>,
    matched_count: usize,
    iterations: usize,
    max_iter: usize,
}

/// Validates shape, hyperparameters, and numerical constraints on `costs`.
fn validate_inputs(
    costs: &[f64],
    n_rows: usize,
    n_cols: usize,
    config: &SolveConfig,
) -> Result<(), SolverError> {
    let expected_len = n_rows
        .checked_mul(n_cols)
        .ok_or_else(|| SolverError::InvalidInput("matrix dimensions overflow usize".to_string()))?;
    if costs.len() != expected_len {
        return Err(SolverError::InvalidInput(format!(
            "costs length ({}) does not match n_rows * n_cols ({n_rows} * {n_cols} = {expected_len})",
            costs.len()
        )));
    }

    if let Some(eps_min) = config.eps_min {
        if !eps_min.is_finite() || eps_min <= 0.0 {
            return Err(SolverError::InvalidInput(format!(
                "eps_min must be finite and > 0.0, got {eps_min}"
            )));
        }
    }

    if !config.eps_factor.is_finite() || config.eps_factor <= 1.0 {
        return Err(SolverError::InvalidInput(format!(
            "eps_factor must be finite and > 1.0, got {}",
            config.eps_factor
        )));
    }

    if config.max_iterations == Some(0) {
        return Err(SolverError::InvalidInput(
            "max_iterations must be > 0".to_string(),
        ));
    }

    if let Some(limit) = config.cost_limit {
        if limit.is_nan() {
            return Err(SolverError::InvalidInput(
                "cost_limit must not be NaN".to_string(),
            ));
        }
    }

    for (idx, &val) in costs.iter().enumerate() {
        if val.is_nan() {
            return Err(SolverError::InvalidInput(format!(
                "cost matrix contains NaN at flat index {idx}"
            )));
        }
        if !config.maximize && val == f64::NEG_INFINITY {
            return Err(SolverError::InvalidInput(format!(
                "cost matrix contains -inf in minimization mode at flat index {idx}"
            )));
        }
        if config.maximize && val == f64::INFINITY {
            return Err(SolverError::InvalidInput(format!(
                "cost matrix contains +inf in maximization mode at flat index {idx}"
            )));
        }
    }

    Ok(())
}

/// Returns `true` iff an `m x n` (`m <= n`) CSR bipartite adjacency admits a matching of size `m`.
fn has_complete_bipartite_matching_csr(csr: &CsrMatrix, m: usize, n: usize) -> bool {
    if m == 0 {
        return true;
    }
    if m > n {
        return false;
    }

    // Quick necessary checks: every row must have an edge, and at least m distinct columns must be reachable.
    let mut col_reachable = vec![false; n];
    let mut reachable_cols = 0usize;
    for i in 0..m {
        let cols = &csr.indices[csr.offsets[i]..csr.offsets[i + 1]];
        if cols.is_empty() {
            return false;
        }
        for &j in cols {
            if !col_reachable[j] {
                col_reachable[j] = true;
                reachable_cols += 1;
            }
        }
    }
    if reachable_cols < m {
        return false;
    }

    let mut pair_u = vec![NONE; m];
    let mut pair_v = vec![NONE; n];
    let mut matched = 0usize;

    // Greedy initial matching to shrink the number of Hopcroft-Karp BFS phases.
    for (u, pair_slot) in pair_u.iter_mut().enumerate() {
        for &v in &csr.indices[csr.offsets[u]..csr.offsets[u + 1]] {
            if pair_v[v] == NONE {
                *pair_slot = v;
                pair_v[v] = u;
                matched += 1;
                break;
            }
        }
    }
    if matched == m {
        return true;
    }

    let mut dist = vec![usize::MAX; m];
    let mut cur = vec![0usize; m];
    let mut queue = VecDeque::with_capacity(m);
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(m);

    loop {
        queue.clear();
        let mut dist_nil = usize::MAX;
        for u in 0..m {
            if pair_u[u] == NONE {
                dist[u] = 0;
                queue.push_back(u);
            } else {
                dist[u] = usize::MAX;
            }
        }

        while let Some(u) = queue.pop_front() {
            if dist[u] >= dist_nil {
                continue;
            }
            let next_dist = dist[u] + 1;
            for &v in &csr.indices[csr.offsets[u]..csr.offsets[u + 1]] {
                let next_u = pair_v[v];
                if next_u == NONE {
                    dist_nil = next_dist;
                } else if dist[next_u] == usize::MAX {
                    dist[next_u] = next_dist;
                    queue.push_back(next_u);
                }
            }
        }

        if dist_nil == usize::MAX {
            return matched == m;
        }

        cur.fill(0);
        for start_u in 0..m {
            if pair_u[start_u] != NONE {
                continue;
            }
            stack.clear();
            stack.push((start_u, NONE));
            let mut augmented = false;

            while let Some(&(u, _)) = stack.last() {
                let row_start = csr.offsets[u];
                let row_len = csr.offsets[u + 1] - row_start;
                let mut pushed_or_found = false;

                while cur[u] < row_len {
                    let v = csr.indices[row_start + cur[u]];
                    cur[u] += 1;
                    let next_u = pair_v[v];
                    if next_u == NONE {
                        if dist[u] + 1 == dist_nil {
                            if let Some(top) = stack.last_mut() {
                                top.1 = v;
                            }
                            for &(su, sv) in &stack {
                                pair_u[su] = sv;
                                pair_v[sv] = su;
                            }
                            augmented = true;
                            pushed_or_found = true;
                            break;
                        }
                    } else if dist[next_u] == dist[u] + 1 {
                        if let Some(top) = stack.last_mut() {
                            top.1 = v;
                        }
                        stack.push((next_u, NONE));
                        pushed_or_found = true;
                        break;
                    }
                }

                if augmented {
                    matched += 1;
                    break;
                }
                if !pushed_or_found {
                    dist[u] = usize::MAX;
                    stack.pop();
                }
            }
        }

        if matched == m {
            return true;
        }
    }
}

#[cfg(test)]
fn has_complete_bipartite_matching(feasible: &[bool], m: usize, n: usize) -> bool {
    if m == 0 {
        return true;
    }
    if m > n || feasible.len() != m * n {
        return false;
    }
    let mut offsets = Vec::with_capacity(m + 1);
    let mut indices = Vec::new();
    offsets.push(0);
    for i in 0..m {
        for j in 0..n {
            if feasible[i * n + j] {
                indices.push(j);
            }
        }
        offsets.push(indices.len());
    }
    has_complete_bipartite_matching_csr(
        &CsrMatrix {
            offsets,
            indices,
            vals: Vec::new(),
        },
        m,
        n,
    )
}

/// Converts input costs into an `m <= n` benefit representation (dense slice when all finite, CSR when `inf`-gated).
fn build_complete_benefits(
    costs: &[f64],
    n_rows: usize,
    n_cols: usize,
    maximize: bool,
) -> NormalizedBenefits {
    let transposed = n_rows > n_cols;
    let (m, n) = if transposed {
        (n_cols, n_rows)
    } else {
        (n_rows, n_cols)
    };

    let mut has_forbidden = false;
    let mut b_min = f64::INFINITY;
    let mut b_max = f64::NEG_INFINITY;
    let mut max_abs_b = 0.0f64;
    let mut nnz = 0usize;

    for &val in costs {
        let forbidden = if maximize {
            val == f64::NEG_INFINITY
        } else {
            val == f64::INFINITY
        };
        if forbidden {
            has_forbidden = true;
        } else {
            let b = if maximize { val } else { -val };
            b_min = b_min.min(b);
            b_max = b_max.max(b);
            max_abs_b = max_abs_b.max(b.abs());
            nnz += 1;
        }
    }

    let b_shift = if b_min > 0.0 {
        b_min
    } else if b_max < 0.0 {
        b_max
    } else {
        0.0
    };
    let shift = if b_shift != 0.0 && b_shift.is_finite() {
        b_shift
    } else {
        0.0
    };

    let storage = if !has_forbidden {
        let mut benefits = Vec::with_capacity(m * n);
        if !transposed {
            if maximize {
                if shift == 0.0 {
                    benefits.extend_from_slice(costs);
                } else {
                    benefits.extend(costs.iter().map(|&v| v - shift));
                }
            } else if shift == 0.0 {
                benefits.extend(costs.iter().map(|&v| -v));
            } else {
                benefits.extend(costs.iter().map(|&v| -v - shift));
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    let v = costs[j * n_cols + i];
                    benefits.push((if maximize { v } else { -v }) - shift);
                }
            }
        }
        BenefitStorage::Dense(benefits)
    } else {
        let mut offsets = Vec::with_capacity(m + 1);
        let mut indices = Vec::with_capacity(nnz);
        let mut vals = Vec::with_capacity(nnz);
        offsets.push(0);
        for i in 0..m {
            for j in 0..n {
                let raw = if transposed {
                    costs[j * n_cols + i]
                } else {
                    costs[i * n_cols + j]
                };
                let forbidden = if maximize {
                    raw == f64::NEG_INFINITY
                } else {
                    raw == f64::INFINITY
                };
                if !forbidden {
                    indices.push(j);
                    vals.push((if maximize { raw } else { -raw }) - shift);
                }
            }
            offsets.push(indices.len());
        }
        BenefitStorage::Sparse(CsrMatrix {
            offsets,
            indices,
            vals,
        })
    };

    let spread = b_max - b_min;
    let is_constant = spread == 0.0;
    let half_spread = if b_max.is_finite() && b_min.is_finite() {
        (b_max * 0.5) - (b_min * 0.5)
    } else {
        0.0
    };
    let scale = if is_constant {
        if max_abs_b > 0.0 { max_abs_b } else { 1.0 }
    } else if spread.is_finite() {
        spread
    } else {
        f64::MAX
    };

    NormalizedBenefits {
        storage,
        m,
        n,
        transposed,
        b_shift,
        scale,
        half_spread,
        is_constant,
    }
}

/// Converts input costs into an `m <= n` CSR net-benefit matrix for `cost_limit` partial mode.
/// Returns `Ok(None)` when no eligible (`net > 0.0`) entry exists.
fn build_partial_benefits(
    costs: &[f64],
    n_rows: usize,
    n_cols: usize,
    maximize: bool,
    limit: f64,
) -> Result<Option<NormalizedBenefits>, SolverError> {
    let transposed = n_rows > n_cols;
    let (m, n) = if transposed {
        (n_cols, n_rows)
    } else {
        (n_rows, n_cols)
    };

    let mut offsets = Vec::with_capacity(m + 1);
    let mut indices = Vec::new();
    let mut vals = Vec::new();
    let mut b_max = 0.0_f64;
    offsets.push(0);

    for i in 0..m {
        for j in 0..n {
            let val = if transposed {
                costs[j * n_cols + i]
            } else {
                costs[i * n_cols + j]
            };
            if !val.is_finite() {
                continue;
            }
            let eligible = if maximize { val > limit } else { val < limit };
            if !eligible {
                continue;
            }
            let net = if maximize { val - limit } else { limit - val };
            if !net.is_finite() {
                return Err(SolverError::Convergence {
                    message: "net benefit overflows f64 in partial mode".to_string(),
                    iterations: 0,
                    final_eps: 0.0,
                });
            }
            indices.push(j);
            vals.push(net);
            b_max = b_max.max(net);
        }
        offsets.push(indices.len());
    }

    if indices.is_empty() {
        return Ok(None);
    }

    Ok(Some(NormalizedBenefits {
        storage: BenefitStorage::Sparse(CsrMatrix {
            offsets,
            indices,
            vals,
        }),
        m,
        n,
        transposed,
        b_shift: 0.0,
        scale: b_max,
        half_spread: b_max * 0.5,
        is_constant: false,
    }))
}

/// Returns the maximum `row[k] - p[k]` over a slice using 4 independent accumulators (`vmaxpd`-friendly).
#[inline]
fn max_diff_slice(row: &[f64], p: &[f64]) -> f64 {
    debug_assert_eq!(row.len(), p.len());
    let len = row.len();
    let mut m0 = f64::NEG_INFINITY;
    let mut m1 = f64::NEG_INFINITY;
    let mut m2 = f64::NEG_INFINITY;
    let mut m3 = f64::NEG_INFINITY;

    let mut k = 0usize;
    while k + 4 <= len {
        unsafe {
            let d0 = *row.get_unchecked(k) - *p.get_unchecked(k);
            let d1 = *row.get_unchecked(k + 1) - *p.get_unchecked(k + 1);
            let d2 = *row.get_unchecked(k + 2) - *p.get_unchecked(k + 2);
            let d3 = *row.get_unchecked(k + 3) - *p.get_unchecked(k + 3);
            if d0 > m0 {
                m0 = d0;
            }
            if d1 > m1 {
                m1 = d1;
            }
            if d2 > m2 {
                m2 = d2;
            }
            if d3 > m3 {
                m3 = d3;
            }
        }
        k += 4;
    }
    let mut best = m0.max(m1).max(m2.max(m3));
    while k < len {
        let d = unsafe { *row.get_unchecked(k) - *p.get_unchecked(k) };
        if d > best {
            best = d;
        }
        k += 1;
    }
    best
}

/// Finds `(best_idx, best_val, second_best_val)` over a dense finite slice of `(b, p)` pairs (`len >= 2`).
#[inline]
fn best_two_dense(row: &[f64], p: &[f64]) -> (usize, f64, f64) {
    debug_assert_eq!(row.len(), p.len());
    debug_assert!(row.len() >= 2);
    let len = row.len();

    if len < 16 {
        let v0 = row[0] - p[0];
        let v1 = row[1] - p[1];
        let (mut best_idx, mut best_val, mut second_val) = if v1 > v0 {
            (1usize, v1, v0)
        } else {
            (0usize, v0, v1)
        };
        for idx in 2..len {
            let val = unsafe { *row.get_unchecked(idx) - *p.get_unchecked(idx) };
            if val > best_val {
                second_val = best_val;
                best_val = val;
                best_idx = idx;
            } else if val > second_val {
                second_val = val;
            }
        }
        return (best_idx, best_val, second_val);
    }

    // Pass 1: 4-way unrolled argmax breaks loop-carried dependency for instruction-level parallelism.
    let mut m0 = f64::NEG_INFINITY;
    let mut m1 = f64::NEG_INFINITY;
    let mut m2 = f64::NEG_INFINITY;
    let mut m3 = f64::NEG_INFINITY;
    let mut i0 = 0usize;
    let mut i1 = 1usize;
    let mut i2 = 2usize;
    let mut i3 = 3usize;

    let mut k = 0usize;
    while k + 4 <= len {
        unsafe {
            let d0 = *row.get_unchecked(k) - *p.get_unchecked(k);
            let d1 = *row.get_unchecked(k + 1) - *p.get_unchecked(k + 1);
            let d2 = *row.get_unchecked(k + 2) - *p.get_unchecked(k + 2);
            let d3 = *row.get_unchecked(k + 3) - *p.get_unchecked(k + 3);
            if d0 > m0 {
                m0 = d0;
                i0 = k;
            }
            if d1 > m1 {
                m1 = d1;
                i1 = k + 1;
            }
            if d2 > m2 {
                m2 = d2;
                i2 = k + 2;
            }
            if d3 > m3 {
                m3 = d3;
                i3 = k + 3;
            }
        }
        k += 4;
    }

    let (mut best_idx, mut best_val) = if m1 > m0 { (i1, m1) } else { (i0, m0) };
    if m2 > best_val {
        best_val = m2;
        best_idx = i2;
    }
    if m3 > best_val {
        best_val = m3;
        best_idx = i3;
    }
    while k < len {
        let d = unsafe { *row.get_unchecked(k) - *p.get_unchecked(k) };
        if d > best_val {
            best_val = d;
            best_idx = k;
        }
        k += 1;
    }

    // Pass 2: branchless 4-lane max excluding best_idx.
    let left_max = max_diff_slice(&row[..best_idx], &p[..best_idx]);
    let right_max = max_diff_slice(&row[best_idx + 1..], &p[best_idx + 1..]);
    (best_idx, best_val, left_max.max(right_max))
}

/// Finds `(best_idx, best_val, second_best_val, feasible_count)` over a sparse CSR adjacency slice.
#[inline]
fn best_two_csr(indices: &[usize], vals: &[f64], duals: &[f64]) -> (usize, f64, f64, usize) {
    let len = indices.len();
    if len == 0 {
        return (NONE, f64::NEG_INFINITY, f64::NEG_INFINITY, 0);
    }
    let idx0 = unsafe { *indices.get_unchecked(0) };
    let v0 = unsafe { *vals.get_unchecked(0) - *duals.get_unchecked(idx0) };
    if len == 1 {
        return (idx0, v0, f64::NEG_INFINITY, 1);
    }
    let idx1 = unsafe { *indices.get_unchecked(1) };
    let v1 = unsafe { *vals.get_unchecked(1) - *duals.get_unchecked(idx1) };
    let (mut best_idx, mut best_val, mut second_val) = if v1 > v0 {
        (idx1, v1, v0)
    } else {
        (idx0, v0, v1)
    };
    for k in 2..len {
        let idx = unsafe { *indices.get_unchecked(k) };
        let val = unsafe { *vals.get_unchecked(k) - *duals.get_unchecked(idx) };
        if val > best_val {
            second_val = best_val;
            best_val = val;
            best_idx = idx;
        } else if val > second_val {
            second_val = val;
        }
    }
    (best_idx, best_val, second_val, len)
}

/// Checks and increments the global bid iteration counter, returning `Convergence` if exhausted.
#[inline]
fn step_iteration(state: &mut AuctionState<'_>, eps: f64) -> Result<(), SolverError> {
    if state.iterations >= state.max_iter {
        return Err(SolverError::Convergence {
            message: format!("exceeded max_iterations ({})", state.max_iter),
            iterations: state.iterations,
            final_eps: eps,
        });
    }
    state.iterations += 1;
    Ok(())
}

/// Ensures `state.pi[i]` satisfies `eps`-CS for unmatched rows before entering reverse bidding.
fn sync_unmatched_row_duals(state: &mut AuctionState<'_>) {
    match state.storage {
        BenefitStorage::Dense(benefits) => {
            for i in 0..state.m {
                if state.row_to_col[i] != NONE {
                    continue;
                }
                let row_off = i * state.n;
                let mut best_val = f64::NEG_INFINITY;
                for (&b, &pj) in benefits[row_off..row_off + state.n]
                    .iter()
                    .zip(state.p.iter())
                {
                    best_val = best_val.max(b - pj);
                }
                state.pi[i] = best_val;
            }
        }
        BenefitStorage::Sparse(csr) => {
            for i in 0..state.m {
                if state.row_to_col[i] != NONE {
                    continue;
                }
                let start = csr.offsets[i];
                let end = csr.offsets[i + 1];
                let mut best_val = f64::NEG_INFINITY;
                for k in start..end {
                    let j = csr.indices[k];
                    best_val = best_val.max(csr.vals[k] - state.p[j]);
                }
                state.pi[i] = best_val;
            }
        }
    }
}

/// Lazily builds `sparse_cols` from sparse row storage when reverse bidding runs.
fn ensure_sparse_cols(state: &mut AuctionState<'_>) {
    if state.sparse_cols.is_some() {
        return;
    }
    if let BenefitStorage::Sparse(csr) = state.storage {
        state.sparse_cols = Some(csr.transpose(state.m, state.n));
    }
}

/// Executes one forward bid for unassigned row `row`. Returns `true` if `matched_count` increased.
fn forward_bid(
    state: &mut AuctionState<'_>,
    row: usize,
    eps: f64,
    partial: bool,
) -> Result<bool, SolverError> {
    step_iteration(state, eps)?;

    let (j1, v1, v2, feasible_count, b_best) = match state.storage {
        BenefitStorage::Dense(benefits) => {
            let row_off = row * state.n;
            let row_slice = &benefits[row_off..row_off + state.n];
            if state.n >= 2 {
                let (idx, b1, b2) = best_two_dense(row_slice, &state.p);
                (idx, b1, b2, state.n, row_slice[idx])
            } else {
                (
                    0,
                    row_slice[0] - state.p[0],
                    f64::NEG_INFINITY,
                    1,
                    row_slice[0],
                )
            }
        }
        BenefitStorage::Sparse(csr) => {
            let start = csr.offsets[row];
            let end = csr.offsets[row + 1];
            let cols = &csr.indices[start..end];
            let vals = &csr.vals[start..end];
            let (idx, b1, b2, count) = best_two_csr(cols, vals, &state.p);
            let b_val = if idx == NONE {
                f64::NEG_INFINITY
            } else {
                b1 + state.p[idx]
            };
            (idx, b1, b2, count, b_val)
        }
    };

    if partial {
        if j1 == NONE || v1 <= 0.0 {
            state.pi[row] = 0.0;
            return Ok(false);
        }
    } else if j1 == NONE {
        return Err(SolverError::Infeasible(format!(
            "row {row} has no feasible columns"
        )));
    }

    let p_old = state.p[j1];
    let (p_new, pi_new) = if partial {
        let pi_new = v2.max(0.0) - eps;
        (b_best - pi_new, pi_new)
    } else if feasible_count == 1 {
        let p_new = p_old + eps;
        (p_new, b_best - p_new)
    } else {
        let pi_new = v2 - eps;
        (b_best - pi_new, pi_new)
    };

    if p_new <= p_old || !p_new.is_finite() || !pi_new.is_finite() {
        return Err(SolverError::Convergence {
            message: "f64 ULP stagnation in forward bid".to_string(),
            iterations: state.iterations,
            final_eps: eps,
        });
    }

    state.p[j1] = p_new;
    state.pi[row] = pi_new;

    let prev_row = state.col_to_row[j1];
    state.row_to_col[row] = j1;
    state.col_to_row[j1] = row;

    if prev_row == NONE {
        state.matched_count += 1;
        Ok(true)
    } else {
        state.row_to_col[prev_row] = NONE;
        state.unassigned_rows.push_back(prev_row);
        Ok(false)
    }
}

/// Best `(i1, u1, u2, feasible_count, b_best)` for column `col` during reverse bidding.
#[inline]
fn best_two_for_col(state: &AuctionState<'_>, col: usize) -> (usize, f64, f64, usize, f64) {
    match (&state.storage, &state.sparse_cols) {
        (BenefitStorage::Dense(benefits), _) => {
            let mut best_idx = NONE;
            let mut best_val = f64::NEG_INFINITY;
            let mut second_val = f64::NEG_INFINITY;
            for i in 0..state.m {
                let b = benefits[i * state.n + col];
                let val = b - state.pi[i];
                if best_idx == NONE || val > best_val {
                    second_val = best_val;
                    best_val = val;
                    best_idx = i;
                } else if i == 1 || val > second_val {
                    second_val = val;
                }
            }
            let b_val = if best_idx == NONE {
                f64::NEG_INFINITY
            } else {
                benefits[best_idx * state.n + col]
            };
            (best_idx, best_val, second_val, state.m, b_val)
        }
        (_, Some(csr_cols)) => {
            let start = csr_cols.offsets[col];
            let end = csr_cols.offsets[col + 1];
            let rows = &csr_cols.indices[start..end];
            let vals = &csr_cols.vals[start..end];
            let (idx, u1, u2, count) = best_two_csr(rows, vals, &state.pi);
            let b_val = if idx == NONE {
                f64::NEG_INFINITY
            } else {
                u1 + state.pi[idx]
            };
            (idx, u1, u2, count, b_val)
        }
        (BenefitStorage::Sparse(_), None) => unreachable!("sparse_cols must be initialized"),
    }
}

/// Executes one symmetric reverse bid for unassigned column `col` (`m == n`). Returns `true` if `matched_count` increased.
fn reverse_bid_symmetric(
    state: &mut AuctionState<'_>,
    col: usize,
    eps: f64,
) -> Result<bool, SolverError> {
    step_iteration(state, eps)?;

    let (i1, _u1, u2, feasible_count, b_val) = best_two_for_col(state, col);
    if i1 == NONE {
        return Err(SolverError::Infeasible(format!(
            "column {col} has no feasible rows"
        )));
    }

    let pi_old = state.pi[i1];
    let (pi_new, p_new) = if feasible_count == 1 {
        let pi_new = pi_old + eps;
        (pi_new, b_val - pi_new)
    } else {
        let p_new = u2 - eps;
        (b_val - p_new, p_new)
    };

    if pi_new <= pi_old || !pi_new.is_finite() || !p_new.is_finite() {
        return Err(SolverError::Convergence {
            message: "f64 ULP stagnation in reverse bid".to_string(),
            iterations: state.iterations,
            final_eps: eps,
        });
    }

    state.pi[i1] = pi_new;
    state.p[col] = p_new;

    let prev_col = state.row_to_col[i1];
    state.col_to_row[col] = i1;
    state.row_to_col[i1] = col;

    if prev_col == NONE {
        state.matched_count += 1;
        Ok(true)
    } else {
        state.col_to_row[prev_col] = NONE;
        state.unassigned_cols.push_back(prev_col);
        Ok(false)
    }
}

/// Runs one scaling round for square matrices (`m == n`), preferring fast forward passes and
/// switching to reverse only when forward stalls without increasing cardinality.
fn run_square_round(state: &mut AuctionState<'_>, eps: f64) -> Result<(), SolverError> {
    let mut forward = true;
    let mut pi_synced = false;
    let stall_limit = (state.m * 4).max(64);
    let mut bids_since_progress = 0usize;

    while state.matched_count < state.m {
        if forward {
            let mut did_bid = false;
            while let Some(row) = state.unassigned_rows.pop_front() {
                if state.row_to_col[row] != NONE {
                    continue;
                }
                did_bid = true;
                if forward_bid(state, row, eps, false)? {
                    bids_since_progress = 0;
                } else {
                    bids_since_progress += 1;
                    if bids_since_progress >= stall_limit {
                        if !pi_synced {
                            sync_unmatched_row_duals(state);
                            ensure_sparse_cols(state);
                            pi_synced = true;
                        }
                        forward = false;
                        bids_since_progress = 0;
                    }
                }
                break;
            }
            if !did_bid {
                return Err(SolverError::Infeasible(
                    "no unassigned rows remaining before reaching complete matching".to_string(),
                ));
            }
        } else {
            let mut did_bid = false;
            while let Some(col) = state.unassigned_cols.pop_front() {
                if state.col_to_row[col] != NONE {
                    continue;
                }
                did_bid = true;
                if reverse_bid_symmetric(state, col, eps)? {
                    bids_since_progress = 0;
                    forward = true;
                } else {
                    bids_since_progress += 1;
                    if bids_since_progress >= stall_limit {
                        forward = true;
                        bids_since_progress = 0;
                    }
                }
                break;
            }
            if !did_bid {
                return Err(SolverError::Infeasible(
                    "no unassigned columns remaining before reaching complete matching".to_string(),
                ));
            }
        }
    }
    Ok(())
}

/// Computes `lambda = min_{k assigned} p[k]`.
#[inline]
fn min_assigned_price(row_to_col: &[usize], p: &[f64]) -> f64 {
    let mut lambda = f64::INFINITY;
    for &c in row_to_col {
        if p[c] < lambda {
            lambda = p[c];
        }
    }
    lambda
}

/// Lowers unassigned columns above `floor` (`lambda` in complete `m < n`, `0.0` in partial mode).
fn run_reverse_cleanup(
    state: &mut AuctionState<'_>,
    eps: f64,
    partial: bool,
) -> Result<(), SolverError> {
    ensure_sparse_cols(state);
    let mut floor = if partial {
        0.0
    } else {
        min_assigned_price(&state.row_to_col, &state.p)
    };
    let mut rev_queue = VecDeque::new();
    for j in 0..state.n {
        if state.col_to_row[j] == NONE && state.p[j] > floor {
            rev_queue.push_back(j);
        }
    }

    while let Some(col) = rev_queue.pop_front() {
        if state.col_to_row[col] != NONE || state.p[col] <= floor {
            continue;
        }
        step_iteration(state, eps)?;

        let (i1, u1, u2, _feasible_count, b_val) = best_two_for_col(state, col);
        let should_drop = if partial {
            i1 == NONE || u1 <= 0.0
        } else {
            i1 == NONE || u1 - eps <= floor
        };
        if should_drop {
            state.p[col] = floor;
            continue;
        }

        let p_new = (u2 - eps).max(floor);
        let pi_old = state.pi[i1];
        let pi_new = b_val - p_new;
        if pi_new <= pi_old || !pi_new.is_finite() || !p_new.is_finite() {
            return Err(SolverError::Convergence {
                message: "f64 ULP stagnation in asymmetric reverse bid".to_string(),
                iterations: state.iterations,
                final_eps: eps,
            });
        }

        state.pi[i1] = pi_new;
        state.p[col] = p_new;

        let prev_col = state.row_to_col[i1];
        state.row_to_col[i1] = col;
        state.col_to_row[col] = i1;
        if prev_col == NONE {
            state.matched_count += 1;
        } else {
            state.col_to_row[prev_col] = NONE;
            if !partial {
                floor = min_assigned_price(&state.row_to_col, &state.p);
            }
            if state.p[prev_col] > floor {
                rev_queue.push_back(prev_col);
            }
        }
    }

    Ok(())
}

/// Runs one scaling round for wide rectangular matrices (`m < n`): forward assignment then asymmetric reverse cleanup.
fn run_rectangular_round(state: &mut AuctionState<'_>, eps: f64) -> Result<(), SolverError> {
    while state.matched_count < state.m {
        let Some(row) = state.unassigned_rows.pop_front() else {
            return Err(SolverError::Infeasible(
                "no unassigned rows remaining before reaching complete matching".to_string(),
            ));
        };
        if state.row_to_col[row] != NONE {
            continue;
        }
        forward_bid(state, row, eps, false)?;
    }

    // Re-verify after cleanup that no unassigned column remains above min_{k assigned} p[k].
    loop {
        let lambda = min_assigned_price(&state.row_to_col, &state.p);
        let has_unassigned_above = state
            .col_to_row
            .iter()
            .enumerate()
            .any(|(j, &row)| row == NONE && state.p[j] > lambda);
        if !has_unassigned_above {
            break;
        }
        run_reverse_cleanup(state, eps, false)?;
    }
    Ok(())
}

/// Runs one scaling round for partial assignment (`cost_limit` finite): forward pass with unassigned option at `0.0`, then reverse cleanup down to `0.0`.
fn run_partial_round(state: &mut AuctionState<'_>, eps: f64) -> Result<(), SolverError> {
    while let Some(row) = state.unassigned_rows.pop_front() {
        if state.row_to_col[row] != NONE {
            continue;
        }
        forward_bid(state, row, eps, true)?;
    }
    let has_unassigned_pos = state
        .col_to_row
        .iter()
        .enumerate()
        .any(|(j, &row)| row == NONE && state.p[j] > 0.0);
    if has_unassigned_pos {
        run_reverse_cleanup(state, eps, true)?;
    }
    Ok(())
}

/// Initializes a scaling round by clearing assignments and anchoring `min(p) = 0` in complete mode.
fn init_scaling_round(state: &mut AuctionState<'_>, is_first_round: bool, partial: bool) {
    if !partial && !is_first_round {
        let (p_min, p_max) = state
            .p
            .iter()
            .copied()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(v), mx.max(v))
            });
        if p_min != 0.0 && p_min.is_finite() && (p_max - p_min).is_finite() {
            for pj in &mut state.p {
                *pj -= p_min;
            }
        }
    }

    state.row_to_col.fill(NONE);
    state.col_to_row.fill(NONE);
    state.matched_count = 0;
    state.unassigned_rows.clear();
    state.unassigned_rows.extend(0..state.m);
    if !partial && state.m == state.n {
        state.unassigned_cols.clear();
        state.unassigned_cols.extend(0..state.n);
    }
}

fn classify_mode(cost_limit: Option<f64>, maximize: bool) -> AssignmentMode {
    match cost_limit {
        None => AssignmentMode::Complete,
        Some(limit) if !maximize && limit == f64::INFINITY => AssignmentMode::Complete,
        Some(limit) if maximize && limit == f64::NEG_INFINITY => AssignmentMode::Complete,
        Some(limit) if !maximize && limit == f64::NEG_INFINITY => AssignmentMode::Empty,
        Some(limit) if maximize && limit == f64::INFINITY => AssignmentMode::Empty,
        Some(limit) => AssignmentMode::Partial(limit),
    }
}

fn empty_output(n_rows: usize, n_cols: usize) -> SolveOutput {
    SolveOutput {
        row_ind: Vec::new(),
        col_ind: Vec::new(),
        cost: 0.0,
        iterations: 0,
        scaling_rounds: 0,
        final_eps: 0.0,
        row_prices: vec![0.0; n_rows],
        col_prices: vec![0.0; n_cols],
    }
}

/// Solves a dense 2D linear assignment problem in row-major layout.
pub fn solve_dense(
    costs: &[f64],
    n_rows: usize,
    n_cols: usize,
    config: &SolveConfig,
) -> Result<SolveOutput, SolverError> {
    solve_dense_with_mode(costs, n_rows, n_cols, config, true)
}

/// Fast path returning only `(row_ind, col_ind)` without computing final `cost` or returning dual price vectors.
pub fn assign_dense(
    costs: &[f64],
    n_rows: usize,
    n_cols: usize,
    config: &SolveConfig,
) -> Result<(Vec<usize>, Vec<usize>), SolverError> {
    let out = solve_dense_with_mode(costs, n_rows, n_cols, config, false)?;
    Ok((out.row_ind, out.col_ind))
}

fn solve_dense_with_mode(
    costs: &[f64],
    n_rows: usize,
    n_cols: usize,
    config: &SolveConfig,
    include_diagnostics: bool,
) -> Result<SolveOutput, SolverError> {
    validate_inputs(costs, n_rows, n_cols, config)?;

    let mode = classify_mode(config.cost_limit, config.maximize);
    if n_rows == 0 || n_cols == 0 || mode == AssignmentMode::Empty {
        return Ok(empty_output(n_rows, n_cols));
    }

    let (norm, partial) = match mode {
        AssignmentMode::Complete => (
            build_complete_benefits(costs, n_rows, n_cols, config.maximize),
            false,
        ),
        AssignmentMode::Partial(limit) => {
            match build_partial_benefits(costs, n_rows, n_cols, config.maximize, limit)? {
                Some(norm) => (norm, true),
                None => return Ok(empty_output(n_rows, n_cols)),
            }
        }
        AssignmentMode::Empty => unreachable!(),
    };

    if !partial {
        if let BenefitStorage::Sparse(csr) = &norm.storage {
            if !has_complete_bipartite_matching_csr(csr, norm.m, norm.n) {
                return Err(SolverError::Infeasible(
                    "gated cost matrix has no valid complete assignment".to_string(),
                ));
            }
        }
    }

    let eps_min = config
        .eps_min
        .unwrap_or_else(|| (norm.scale * 1e-9).max(1e-12));
    let mut eps = if norm.is_constant {
        eps_min
    } else {
        norm.half_spread.max(eps_min)
    };

    let max_iter = config.max_iterations.unwrap_or_else(|| {
        100_000usize.max(500usize.saturating_mul(norm.m).saturating_mul(norm.n))
    });

    let mut state = AuctionState {
        storage: &norm.storage,
        sparse_cols: None,
        m: norm.m,
        n: norm.n,
        pi: vec![0.0; norm.m],
        p: vec![0.0; norm.n],
        row_to_col: vec![NONE; norm.m],
        col_to_row: vec![NONE; norm.n],
        unassigned_rows: VecDeque::with_capacity(norm.m),
        unassigned_cols: if !partial && norm.m == norm.n {
            VecDeque::with_capacity(norm.n)
        } else {
            VecDeque::new()
        },
        matched_count: 0,
        iterations: 0,
        max_iter,
    };

    let mut scaling_rounds = 0usize;
    loop {
        scaling_rounds += 1;
        init_scaling_round(&mut state, scaling_rounds == 1, partial);
        if partial {
            run_partial_round(&mut state, eps)?;
        } else if state.m == state.n {
            run_square_round(&mut state, eps)?;
        } else {
            run_rectangular_round(&mut state, eps)?;
        }

        if eps <= eps_min {
            break;
        }
        let next_eps = (eps / config.eps_factor).max(eps_min);
        eps = if next_eps >= eps { eps_min } else { next_eps };
    }

    if include_diagnostics && norm.b_shift != 0.0 && norm.b_shift.is_finite() {
        for r_price in &mut state.pi {
            *r_price += norm.b_shift;
        }
    }

    let (row_ind, col_ind, row_prices, col_prices) = if norm.transposed {
        let mut pairs: Vec<(usize, usize)> = (0..norm.m)
            .filter_map(|c| {
                let r = state.row_to_col[c];
                (r != NONE).then_some((r, c))
            })
            .collect();
        pairs.sort_unstable_by_key(|&(r, _)| r);
        let (rows, cols): (Vec<usize>, Vec<usize>) = pairs.into_iter().unzip();
        if include_diagnostics {
            (rows, cols, state.p, state.pi)
        } else {
            (rows, cols, Vec::new(), Vec::new())
        }
    } else {
        let mut rows = Vec::with_capacity(state.matched_count);
        let mut cols = Vec::with_capacity(state.matched_count);
        for (r, &c) in state.row_to_col.iter().enumerate() {
            if c != NONE {
                rows.push(r);
                cols.push(c);
            }
        }
        if include_diagnostics {
            (rows, cols, state.pi, state.p)
        } else {
            (rows, cols, Vec::new(), Vec::new())
        }
    };

    let cost: f64 = if include_diagnostics {
        row_ind
            .iter()
            .zip(&col_ind)
            .map(|(&r, &c)| costs[r * n_cols + c])
            .sum()
    } else {
        0.0
    };

    Ok(SolveOutput {
        row_ind,
        col_ind,
        cost,
        iterations: state.iterations,
        scaling_rounds,
        final_eps: eps,
        row_prices,
        col_prices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SolveConfig {
        SolveConfig {
            maximize: false,
            cost_limit: None,
            eps_min: None,
            eps_factor: 5.0,
            max_iterations: None,
        }
    }

    fn assert_valid_matching(row_ind: &[usize], col_ind: &[usize], n_rows: usize, n_cols: usize) {
        let expected_len = n_rows.min(n_cols);
        assert_eq!(row_ind.len(), expected_len);
        assert_eq!(col_ind.len(), expected_len);

        let mut seen_cols = vec![false; n_cols];
        for i in 0..expected_len {
            assert!(row_ind[i] < n_rows);
            assert!(col_ind[i] < n_cols);
            if i > 0 {
                assert!(
                    row_ind[i - 1] < row_ind[i],
                    "row_ind must be strictly sorted"
                );
            }
            assert!(
                !seen_cols[col_ind[i]],
                "col_ind must contain unique indices"
            );
            seen_cols[col_ind[i]] = true;
        }
    }

    fn assert_unassigned_price_floor(assigned_indices: &[usize], prices: &[f64], label: &str) {
        let mut assigned = vec![false; prices.len()];
        let mut min_assigned = f64::INFINITY;
        for &idx in assigned_indices {
            assigned[idx] = true;
            min_assigned = min_assigned.min(prices[idx]);
        }
        let max_unassigned = prices
            .iter()
            .enumerate()
            .filter(|&(idx, _)| !assigned[idx])
            .map(|(_, &p)| p)
            .fold(f64::NEG_INFINITY, f64::max);
        let p_scale = prices.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
        let tol = 1e-12 * p_scale;
        assert!(
            max_unassigned <= min_assigned + tol,
            "asymmetric {label} price invariant violated: max_unassigned={max_unassigned} > min_assigned={min_assigned}"
        );
    }

    fn assert_eps_cs(
        costs: &[f64],
        n_rows: usize,
        n_cols: usize,
        maximize: bool,
        out: &SolveOutput,
    ) {
        let scale = out
            .row_prices
            .iter()
            .chain(out.col_prices.iter())
            .map(|v| v.abs())
            .fold(1.0_f64, f64::max);
        let tol = 1e-8 * scale;
        for r in 0..n_rows {
            for c in 0..n_cols {
                let raw = costs[r * n_cols + c];
                let b = if maximize { raw } else { -raw };
                if b.is_finite() {
                    assert!(
                        out.row_prices[r] + out.col_prices[c] >= b - out.final_eps - tol,
                        "eps-CS violated at ({r}, {c}): {} + {} < {} - {}",
                        out.row_prices[r],
                        out.col_prices[c],
                        b,
                        out.final_eps
                    );
                }
            }
        }
        for (&r, &c) in out.row_ind.iter().zip(&out.col_ind) {
            let raw = costs[r * n_cols + c];
            let b = if maximize { raw } else { -raw };
            assert!(b.is_finite(), "matched pair ({r}, {c}) has non-finite cost");
            assert!(
                (out.row_prices[r] + out.col_prices[c] - b).abs() <= tol,
                "matched CS violated at ({r}, {c}): {} + {} != {b}",
                out.row_prices[r],
                out.col_prices[c],
            );
        }
        if n_rows < n_cols {
            assert_unassigned_price_floor(&out.col_ind, &out.col_prices, "col");
        } else if n_rows > n_cols {
            assert_unassigned_price_floor(&out.row_ind, &out.row_prices, "row");
        }
    }

    #[test]
    fn test_empty_matrices() {
        let configs = [
            default_config(),
            SolveConfig {
                cost_limit: Some(10.0),
                ..default_config()
            },
            SolveConfig {
                cost_limit: Some(f64::NEG_INFINITY),
                ..default_config()
            },
            SolveConfig {
                maximize: true,
                cost_limit: Some(-5.0),
                ..default_config()
            },
            SolveConfig {
                maximize: true,
                cost_limit: Some(f64::INFINITY),
                ..default_config()
            },
        ];
        for cfg in &configs {
            for (n_rows, n_cols) in [(0, 0), (0, 4), (4, 0)] {
                let out =
                    solve_dense(&[], n_rows, n_cols, cfg).expect("empty matrix should succeed");
                assert!(out.row_ind.is_empty());
                assert!(out.col_ind.is_empty());
                assert_eq!(out.cost, 0.0);
                assert_eq!(out.iterations, 0);
                assert_eq!(out.scaling_rounds, 0);
                assert_eq!(out.final_eps, 0.0);
                assert_eq!(out.row_prices, vec![0.0; n_rows]);
                assert_eq!(out.col_prices, vec![0.0; n_cols]);
            }
        }
    }

    #[test]
    fn test_invalid_inputs() {
        let cfg = default_config();

        // NaN in costs
        assert!(matches!(
            solve_dense(&[1.0, f64::NAN, 2.0, 3.0], 2, 2, &cfg),
            Err(SolverError::InvalidInput(_))
        ));

        // -inf when minimizing
        assert!(matches!(
            solve_dense(&[1.0, f64::NEG_INFINITY, 2.0, 3.0], 2, 2, &cfg),
            Err(SolverError::InvalidInput(_))
        ));

        // +inf when maximizing
        let max_cfg = SolveConfig {
            maximize: true,
            ..default_config()
        };
        assert!(matches!(
            solve_dense(&[1.0, f64::INFINITY, 2.0, 3.0], 2, 2, &max_cfg),
            Err(SolverError::InvalidInput(_))
        ));

        // Invalid eps_min values
        for bad_eps in [0.0, -1e-6, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let bad_cfg = SolveConfig {
                eps_min: Some(bad_eps),
                ..default_config()
            };
            assert!(matches!(
                solve_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2, &bad_cfg),
                Err(SolverError::InvalidInput(_))
            ));
        }

        // Invalid eps_factor values
        for bad_factor in [1.0, 0.5, -2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let bad_cfg = SolveConfig {
                eps_factor: bad_factor,
                ..default_config()
            };
            assert!(matches!(
                solve_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2, &bad_cfg),
                Err(SolverError::InvalidInput(_))
            ));
        }

        // max_iterations == Some(0)
        let zero_iter_cfg = SolveConfig {
            max_iterations: Some(0),
            ..default_config()
        };
        assert!(matches!(
            solve_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2, &zero_iter_cfg),
            Err(SolverError::InvalidInput(_))
        ));

        // Mismatched slice length
        assert!(matches!(
            solve_dense(&[1.0, 2.0, 3.0], 2, 2, &cfg),
            Err(SolverError::InvalidInput(_))
        ));

        // NaN cost_limit
        let nan_limit_cfg = SolveConfig {
            cost_limit: Some(f64::NAN),
            ..default_config()
        };
        assert!(matches!(
            solve_dense(&[1.0, 2.0, 3.0, 4.0], 2, 2, &nan_limit_cfg),
            Err(SolverError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_infeasible_hall_condition() {
        let inf = f64::INFINITY;
        let costs = [1.0, inf, inf, 2.0, inf, inf, 3.0, 4.0, 5.0];
        let cfg = default_config();
        assert!(matches!(
            solve_dense(&costs, 3, 3, &cfg),
            Err(SolverError::Infeasible(_))
        ));

        // Maximization with -inf Hall violation on transposed orientation (2x3)
        let ninf = f64::NEG_INFINITY;
        let max_costs = [1.0, ninf, ninf, 2.0, ninf, ninf];
        let max_cfg = SolveConfig {
            maximize: true,
            ..default_config()
        };
        assert!(matches!(
            solve_dense(&max_costs, 2, 3, &max_cfg),
            Err(SolverError::Infeasible(_))
        ));
    }

    #[test]
    fn test_square_and_rectangular_complete() {
        let cfg = default_config();

        // 3x3 classic matrix
        let classic = [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        let out = solve_dense(&classic, 3, 3, &cfg).expect("classic 3x3 should solve");
        assert_valid_matching(&out.row_ind, &out.col_ind, 3, 3);
        assert_eq!(out.row_ind, vec![0, 1, 2]);
        assert_eq!(out.col_ind, vec![1, 0, 2]);
        assert!((out.cost - 5.0).abs() < 1e-12);
        assert_eq!(out.row_prices.len(), 3);
        assert_eq!(out.col_prices.len(), 3);

        // Rectangular 2x4 (M < N)
        let rect_wide = [10.0, 1.0, 5.0, 3.0, 2.0, 9.0, 4.0, 8.0];
        let out_wide = solve_dense(&rect_wide, 2, 4, &cfg).expect("2x4 should solve");
        assert_valid_matching(&out_wide.row_ind, &out_wide.col_ind, 2, 4);
        assert_eq!(out_wide.row_ind, vec![0, 1]);
        assert_eq!(out_wide.col_ind, vec![1, 0]);
        assert!((out_wide.cost - 3.0).abs() < 1e-12);
        assert_eq!(out_wide.row_prices.len(), 2);
        assert_eq!(out_wide.col_prices.len(), 4);

        // Transposed 4x2 (M > N), non-prefix optimal rows [1, 3]
        let rect_tall = [10.0, 10.0, 1.0, 9.0, 10.0, 10.0, 9.0, 2.0];
        let out_tall = solve_dense(&rect_tall, 4, 2, &cfg).expect("4x2 should solve");
        assert_valid_matching(&out_tall.row_ind, &out_tall.col_ind, 4, 2);
        assert_eq!(out_tall.row_ind, vec![1, 3]);
        assert_eq!(out_tall.col_ind, vec![0, 1]);
        assert!((out_tall.cost - 3.0).abs() < 1e-12);
        assert_eq!(out_tall.row_prices.len(), 4);
        assert_eq!(out_tall.col_prices.len(), 2);

        // Negative costs with maximize=true
        let neg_costs = [-10.0, -1.0, -5.0, -2.0, -9.0, -4.0, -3.0, -6.0, -2.0];
        let max_cfg = SolveConfig {
            maximize: true,
            ..default_config()
        };
        let out_max = solve_dense(&neg_costs, 3, 3, &max_cfg).expect("maximization should solve");
        assert_valid_matching(&out_max.row_ind, &out_max.col_ind, 3, 3);
        assert_eq!(out_max.row_ind, vec![0, 1, 2]);
        assert_eq!(out_max.col_ind, vec![1, 0, 2]);
        assert!((out_max.cost - (-5.0)).abs() < 1e-12);

        // All-equal and all-zero 5x5 matrices finish in 1 scaling round
        let zeros = vec![0.0; 25];
        let out_zero = solve_dense(&zeros, 5, 5, &cfg).expect("5x5 zeros should solve");
        assert_valid_matching(&out_zero.row_ind, &out_zero.col_ind, 5, 5);
        assert_eq!(out_zero.scaling_rounds, 1);
        assert_eq!(out_zero.cost, 0.0);

        let const_val = 7.5;
        let ones = vec![const_val; 25];
        let out_const = solve_dense(&ones, 5, 5, &cfg).expect("5x5 constant should solve");
        assert_valid_matching(&out_const.row_ind, &out_const.col_ind, 5, 5);
        assert_eq!(out_const.scaling_rounds, 1);
        assert!((out_const.cost - 5.0 * const_val).abs() < 1e-12);

        // Single-feasible-column gated rows and large constant offset
        let inf = f64::INFINITY;
        let gated = [100.0, inf, inf, 0.0, 5.0, inf, inf, 0.0, 1.0];
        let out_gated = solve_dense(&gated, 3, 3, &cfg).expect("gated 3x3 should solve");
        assert_valid_matching(&out_gated.row_ind, &out_gated.col_ind, 3, 3);
        assert_eq!(out_gated.col_ind, vec![0, 1, 2]);
        assert!((out_gated.cost - 106.0).abs() < 1e-12);

        let offset = [1e12, 1e12 + 1.0, 1e12, 1e12 + 2.0];
        let out_offset = solve_dense(&offset, 2, 2, &cfg).expect("offset 2x2 should solve");
        assert_valid_matching(&out_offset.row_ind, &out_offset.col_ind, 2, 2);
        assert_eq!(out_offset.col_ind, vec![1, 0]);
        assert!((out_offset.cost - (2e12 + 1.0)).abs() < 1e-6);

        // Asymmetric rectangular matrix requiring Hopcroft-Karp augmenting path and reverse lambda cleanup
        let asym_max = [100.0, 90.0, 0.0, 0.0, 90.0, 89.0];
        let out_asym = solve_dense(&asym_max, 2, 3, &max_cfg).expect("2x3 asym max should solve");
        assert_valid_matching(&out_asym.row_ind, &out_asym.col_ind, 2, 3);
        assert_eq!(out_asym.row_ind, vec![0, 1]);
        assert_eq!(out_asym.col_ind, vec![0, 1]);
        assert!((out_asym.cost - 190.0).abs() < 1e-9);

        // Tall 4x2 matrix with two all-inf rows that is still feasible (requires augmenting path)
        let tall_gated = [1.0, 2.0, inf, inf, 10.0, inf, inf, inf];
        let out_tall_gated =
            solve_dense(&tall_gated, 4, 2, &cfg).expect("4x2 gated feasible should solve");
        assert_valid_matching(&out_tall_gated.row_ind, &out_tall_gated.col_ind, 4, 2);
        assert_eq!(out_tall_gated.row_ind, vec![0, 2]);
        assert_eq!(out_tall_gated.col_ind, vec![1, 0]);
        assert!((out_tall_gated.cost - 12.0).abs() < 1e-12);

        // Hopcroft-Karp augmenting path test where greedy matching fails
        let aug_adj = [true, true, false, true, false, false];
        assert!(has_complete_bipartite_matching(&aug_adj, 2, 3));
        let no_aug_adj = [true, false, false, true, false, false];
        assert!(!has_complete_bipartite_matching(&no_aug_adj, 2, 3));

        // 1x1 matrix
        let single = [42.0];
        let out_single = solve_dense(&single, 1, 1, &cfg).expect("1x1 should solve");
        assert_valid_matching(&out_single.row_ind, &out_single.col_ind, 1, 1);
        assert_eq!(out_single.cost, 42.0);
        assert_eps_cs(&single, 1, 1, false, &out_single);

        // Dual eps-CS invariant checks on square, wide, and tall orientations
        assert_eps_cs(&classic, 3, 3, false, &out);
        assert_eps_cs(&rect_wide, 2, 4, false, &out_wide);
        assert_eps_cs(&rect_tall, 4, 2, false, &out_tall);
        assert_eps_cs(&neg_costs, 3, 3, true, &out_max);
        assert_eps_cs(&asym_max, 2, 3, true, &out_asym);
        assert_eps_cs(&tall_gated, 4, 2, false, &out_tall_gated);

        // Unbounded cost_limit equivalents
        let inf_limit_cfg = SolveConfig {
            cost_limit: Some(f64::INFINITY),
            ..default_config()
        };
        let out_inf_limit =
            solve_dense(&classic, 3, 3, &inf_limit_cfg).expect("+inf min limit should solve");
        assert_eq!(out_inf_limit.col_ind, out.col_ind);

        let ninf_max_limit_cfg = SolveConfig {
            maximize: true,
            cost_limit: Some(f64::NEG_INFINITY),
            ..default_config()
        };
        let out_ninf_limit = solve_dense(&neg_costs, 3, 3, &ninf_max_limit_cfg)
            .expect("-inf max limit should solve");
        assert_eq!(out_ninf_limit.col_ind, out_max.col_ind);
    }

    fn brute_force_complete_optimal_cost(
        costs: &[f64],
        n_rows: usize,
        n_cols: usize,
        maximize: bool,
    ) -> Option<f64> {
        struct CompleteSearch<'a> {
            costs: &'a [f64],
            m: usize,
            n: usize,
            n_cols: usize,
            transposed: bool,
            maximize: bool,
            used: Vec<bool>,
            best: Option<f64>,
        }

        impl CompleteSearch<'_> {
            fn dfs(&mut self, i: usize, acc: f64) {
                if i == self.m {
                    self.best = Some(match self.best {
                        None => acc,
                        Some(cur) => {
                            if self.maximize {
                                cur.max(acc)
                            } else {
                                cur.min(acc)
                            }
                        }
                    });
                    return;
                }
                for j in 0..self.n {
                    if self.used[j] {
                        continue;
                    }
                    let (r, c) = if self.transposed { (j, i) } else { (i, j) };
                    let val = self.costs[r * self.n_cols + c];
                    if !val.is_finite() {
                        continue;
                    }
                    self.used[j] = true;
                    self.dfs(i + 1, acc + val);
                    self.used[j] = false;
                }
            }
        }

        let transposed = n_rows > n_cols;
        let (m, n) = if transposed {
            (n_cols, n_rows)
        } else {
            (n_rows, n_cols)
        };
        let mut search = CompleteSearch {
            costs,
            m,
            n,
            n_cols,
            transposed,
            maximize,
            used: vec![false; n],
            best: None,
        };
        search.dfs(0, 0.0);
        search.best
    }

    #[test]
    fn test_rectangular_reverse_cleanup_lambda_invariant() {
        let min_cfg = default_config();
        let max_cfg = SolveConfig {
            maximize: true,
            ..default_config()
        };
        let inf = f64::INFINITY;
        let ninf = f64::NEG_INFINITY;

        // Multi-round 2x3 and 3x5/3x6 matrices exercising reverse cleanup steals, clamping at floor, and drops.
        let wide_cases: &[(&[f64], usize, usize, bool, f64)] = &[
            (&[100.0, 90.0, 0.0, 0.0, 90.0, 89.0], 2, 3, true, 190.0),
            (
                &[
                    1.0, 2.0, 100.0, 100.0, 100.0, //
                    1.0, 100.0, 2.0, 100.0, 100.0, //
                    100.0, 100.0, 3.0, 4.0, 5.0,
                ],
                3,
                5,
                false,
                6.0,
            ),
            (
                &[
                    50.0, 49.0, 10.0, 0.0, inf, //
                    49.0, 48.0, inf, 12.0, 11.0, //
                    1.0, 2.0, 30.0, 29.0, 28.0,
                ],
                3,
                5,
                false,
                12.0,
            ),
            (
                &[
                    80.0, 79.0, ninf, 10.0, 5.0, 1.0, //
                    79.0, 77.0, 60.0, ninf, 20.0, 2.0, //
                    10.0, 12.0, 59.0, 58.0, 40.0, ninf,
                ],
                3,
                6,
                true,
                217.0,
            ),
        ];

        for &(costs, n_rows, n_cols, maximize, expected_opt) in wide_cases {
            let cfg = if maximize { &max_cfg } else { &min_cfg };
            let out = solve_dense(costs, n_rows, n_cols, cfg)
                .expect("rectangular complete case should solve");
            assert!(out.scaling_rounds > 1);
            assert_valid_matching(&out.row_ind, &out.col_ind, n_rows, n_cols);
            assert_eps_cs(costs, n_rows, n_cols, maximize, &out);
            assert_unassigned_price_floor(&out.col_ind, &out.col_prices, "col");
            let oracle = brute_force_complete_optimal_cost(costs, n_rows, n_cols, maximize)
                .expect("crafted rectangular case must be feasible");
            assert!((oracle - expected_opt).abs() < 1e-12);
            assert!(
                (out.cost - expected_opt).abs() <= (n_rows as f64) * out.final_eps + 1e-9,
                "wide optimality gap: got={}, expected={expected_opt}",
                out.cost
            );

            // Also verify transposed tall orientation (M > N).
            let mut transposed = vec![0.0; n_rows * n_cols];
            for r in 0..n_rows {
                for c in 0..n_cols {
                    transposed[c * n_rows + r] = costs[r * n_cols + c];
                }
            }
            let out_t = solve_dense(&transposed, n_cols, n_rows, cfg)
                .expect("transposed rectangular case should solve");
            assert_valid_matching(&out_t.row_ind, &out_t.col_ind, n_cols, n_rows);
            assert_eps_cs(&transposed, n_cols, n_rows, maximize, &out_t);
            assert_unassigned_price_floor(&out_t.row_ind, &out_t.row_prices, "row");
            assert!(
                (out_t.cost - expected_opt).abs() <= (n_rows as f64) * out_t.final_eps + 1e-9,
                "tall optimality gap: got={}, expected={expected_opt}",
                out_t.cost
            );
        }

        // Deterministic PRNG differential sweep over multi-round m < n and m > n matrices.
        let mut seed: u64 = 0x517c_c1b7_2722_0a95;
        let mut next_u64 = || -> u64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for case_idx in 0..120 {
            let m_small = ((next_u64() % 4) + 2) as usize;
            let n_large = m_small + ((next_u64() % 3) + 1) as usize;
            let (n_rows, n_cols) = if case_idx % 2 == 0 {
                (m_small, n_large)
            } else {
                (n_large, m_small)
            };
            let maximize = (case_idx / 2) % 2 == 1;
            let offset = if case_idx % 3 == 0 { 1000.0 } else { 0.0 };

            let mut costs = Vec::with_capacity(n_rows * n_cols);
            for _ in 0..(n_rows * n_cols) {
                if next_u64() % 6 == 0 {
                    costs.push(if maximize { ninf } else { inf });
                } else {
                    let v = ((next_u64() % 200) as f64) * 0.5 - 50.0 + offset;
                    costs.push(v);
                }
            }

            let oracle = brute_force_complete_optimal_cost(&costs, n_rows, n_cols, maximize);
            let cfg = if maximize { &max_cfg } else { &min_cfg };
            match (oracle, solve_dense(&costs, n_rows, n_cols, cfg)) {
                (None, Err(SolverError::Infeasible(_))) => {}
                (Some(best_cost), Ok(out)) => {
                    assert_valid_matching(&out.row_ind, &out.col_ind, n_rows, n_cols);
                    assert_eps_cs(&costs, n_rows, n_cols, maximize, &out);
                    let tol = 1e-8 * best_cost.abs().max(1.0);
                    assert!(
                        (out.cost - best_cost).abs() <= (m_small as f64) * out.final_eps + tol,
                        "case {case_idx} ({n_rows}x{n_cols}, max={maximize}): best={best_cost}, got={}",
                        out.cost
                    );
                }
                (oracle_res, solver_res) => {
                    panic!(
                        "case {case_idx} mismatch: oracle={oracle_res:?}, solver={solver_res:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_max_iterations_and_ulp_stall() {
        let non_trivial = [
            10.0, 1.0, 5.0, 3.0, 2.0, 9.0, 4.0, 8.0, 6.0, 3.0, 11.0, 2.0, 7.0, 8.0, 1.0, 9.0,
        ];
        let one_iter_cfg = SolveConfig {
            max_iterations: Some(1),
            ..default_config()
        };
        assert!(matches!(
            solve_dense(&non_trivial, 4, 4, &one_iter_cfg),
            Err(SolverError::Convergence { iterations: 1, .. })
        ));

        // Extreme dynamic range (0.0, 1e12) with eps_min below ULP(1e12)
        let extreme = [0.0, 1e12, 0.0, 1e12];
        let tiny_eps_cfg = SolveConfig {
            eps_min: Some(1e-20),
            ..default_config()
        };
        match solve_dense(&extreme, 2, 2, &tiny_eps_cfg) {
            Err(SolverError::Convergence {
                iterations,
                final_eps,
                ..
            }) => {
                assert!(iterations > 0);
                assert!(
                    iterations < 10_000,
                    "should halt promptly on ULP stall, got {iterations}"
                );
                assert!(final_eps > 0.0);
            }
            other => panic!("expected Convergence error on ULP stall, got {other:?}"),
        }
    }

    fn assert_partial_matching(
        out: &SolveOutput,
        costs: &[f64],
        n_rows: usize,
        n_cols: usize,
        maximize: bool,
        limit: f64,
    ) {
        assert_eq!(out.row_ind.len(), out.col_ind.len());
        assert!(out.row_ind.len() <= n_rows.min(n_cols));
        assert_eq!(out.row_prices.len(), n_rows);
        assert_eq!(out.col_prices.len(), n_cols);

        let mut seen_cols = vec![false; n_cols];
        let mut expected_cost = 0.0;
        let mut abs_sum = 0.0;
        for i in 0..out.row_ind.len() {
            let r = out.row_ind[i];
            let c = out.col_ind[i];
            assert!(r < n_rows);
            assert!(c < n_cols);
            if i > 0 {
                assert!(out.row_ind[i - 1] < r, "row_ind must be strictly sorted");
            }
            assert!(!seen_cols[c], "col_ind must contain unique indices");
            seen_cols[c] = true;

            let val = costs[r * n_cols + c];
            assert!(val.is_finite(), "matched cost must be finite, got {val}");
            if maximize {
                assert!(val > limit, "matched score {val} must be > limit {limit}");
            } else {
                assert!(val < limit, "matched cost {val} must be < limit {limit}");
            }
            expected_cost += val;
            abs_sum += val.abs();
        }
        let tol = 1e-12 * abs_sum.max(1.0);
        assert!(
            (out.cost - expected_cost).abs() <= tol,
            "cost mismatch: out.cost={}, expected={expected_cost}",
            out.cost
        );
    }

    fn assert_partial_eps_cs(
        costs: &[f64],
        n_rows: usize,
        n_cols: usize,
        maximize: bool,
        limit: f64,
        out: &SolveOutput,
    ) {
        let mut scale = 1.0_f64;
        for &val in costs {
            if val.is_finite() {
                let net = if maximize { val - limit } else { limit - val };
                if net > 0.0 {
                    scale = scale.max(net);
                }
            }
        }
        let tol = 1e-8 * scale;

        let mut row_matched = vec![false; n_rows];
        let mut col_matched = vec![false; n_cols];
        for (&r, &c) in out.row_ind.iter().zip(&out.col_ind) {
            row_matched[r] = true;
            col_matched[c] = true;
            let val = costs[r * n_cols + c];
            let net = if maximize { val - limit } else { limit - val };
            assert!(
                (out.row_prices[r] + out.col_prices[c] - net).abs() <= tol,
                "matched CS violated at ({r}, {c}): {} + {} != {net}",
                out.row_prices[r],
                out.col_prices[c]
            );
        }

        for (r, &pi_r) in out.row_prices.iter().enumerate() {
            if !row_matched[r] {
                assert_eq!(
                    pi_r, 0.0,
                    "unassigned row {r} price must be 0.0, got {pi_r}"
                );
            }
            assert!(
                pi_r >= -out.final_eps - tol,
                "row {r} price {pi_r} below -final_eps (-{})",
                out.final_eps
            );
        }
        for (c, &p_c) in out.col_prices.iter().enumerate() {
            if !col_matched[c] {
                assert_eq!(p_c, 0.0, "unassigned col {c} price must be 0.0, got {p_c}");
            }
            assert!(
                p_c >= -out.final_eps - tol,
                "col {c} price {p_c} below -final_eps (-{})",
                out.final_eps
            );
        }

        for r in 0..n_rows {
            for c in 0..n_cols {
                let val = costs[r * n_cols + c];
                if !val.is_finite() {
                    continue;
                }
                let net = if maximize { val - limit } else { limit - val };
                if net > 0.0 {
                    assert!(
                        out.row_prices[r] + out.col_prices[c] >= net - out.final_eps - tol,
                        "partial eps-CS violated at ({r}, {c}): {} + {} < {net} - {}",
                        out.row_prices[r],
                        out.col_prices[c],
                        out.final_eps
                    );
                }
            }
        }
    }

    fn brute_force_best_partial_benefit(
        costs: &[f64],
        n_rows: usize,
        n_cols: usize,
        maximize: bool,
        limit: f64,
    ) -> f64 {
        fn dfs(
            row: usize,
            costs: &[f64],
            n_rows: usize,
            n_cols: usize,
            maximize: bool,
            limit: f64,
            used_cols: &mut [bool],
        ) -> f64 {
            if row == n_rows {
                return 0.0;
            }
            let mut best = dfs(row + 1, costs, n_rows, n_cols, maximize, limit, used_cols);
            for col in 0..n_cols {
                if used_cols[col] {
                    continue;
                }
                let val = costs[row * n_cols + col];
                if !val.is_finite() {
                    continue;
                }
                let net = if maximize { val - limit } else { limit - val };
                if net > 0.0 {
                    used_cols[col] = true;
                    let candidate =
                        net + dfs(row + 1, costs, n_rows, n_cols, maximize, limit, used_cols);
                    if candidate > best {
                        best = candidate;
                    }
                    used_cols[col] = false;
                }
            }
            best
        }

        let mut used_cols = vec![false; n_cols];
        dfs(0, costs, n_rows, n_cols, maximize, limit, &mut used_cols)
    }

    fn net_benefit_sum(
        out: &SolveOutput,
        costs: &[f64],
        n_cols: usize,
        maximize: bool,
        limit: f64,
    ) -> f64 {
        out.row_ind
            .iter()
            .zip(&out.col_ind)
            .map(|(&r, &c)| {
                let val = costs[r * n_cols + c];
                if maximize { val - limit } else { limit - val }
            })
            .sum()
    }

    #[test]
    fn test_cost_limit_unbounded_vs_empty() {
        let costs = [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        let min_none = solve_dense(&costs, 3, 3, &default_config()).expect("min None should solve");
        let min_inf = solve_dense(
            &costs,
            3,
            3,
            &SolveConfig {
                cost_limit: Some(f64::INFINITY),
                ..default_config()
            },
        )
        .expect("min +inf limit should solve");
        assert_eq!(min_inf.row_ind, min_none.row_ind);
        assert_eq!(min_inf.col_ind, min_none.col_ind);
        assert_eq!(min_inf.cost, min_none.cost);

        let min_ninf = solve_dense(
            &costs,
            3,
            3,
            &SolveConfig {
                cost_limit: Some(f64::NEG_INFINITY),
                ..default_config()
            },
        )
        .expect("min -inf limit should return empty");
        assert!(min_ninf.row_ind.is_empty());
        assert!(min_ninf.col_ind.is_empty());
        assert_eq!(min_ninf.cost, 0.0);
        assert_eq!(min_ninf.iterations, 0);
        assert_eq!(min_ninf.scaling_rounds, 0);
        assert_eq!(min_ninf.final_eps, 0.0);
        assert_eq!(min_ninf.row_prices, vec![0.0; 3]);
        assert_eq!(min_ninf.col_prices, vec![0.0; 3]);

        let max_none = solve_dense(
            &costs,
            3,
            3,
            &SolveConfig {
                maximize: true,
                ..default_config()
            },
        )
        .expect("max None should solve");
        let max_ninf = solve_dense(
            &costs,
            3,
            3,
            &SolveConfig {
                maximize: true,
                cost_limit: Some(f64::NEG_INFINITY),
                ..default_config()
            },
        )
        .expect("max -inf limit should solve");
        assert_eq!(max_ninf.row_ind, max_none.row_ind);
        assert_eq!(max_ninf.col_ind, max_none.col_ind);
        assert_eq!(max_ninf.cost, max_none.cost);

        let max_inf = solve_dense(
            &costs,
            3,
            3,
            &SolveConfig {
                maximize: true,
                cost_limit: Some(f64::INFINITY),
                ..default_config()
            },
        )
        .expect("max +inf limit should return empty");
        assert!(max_inf.row_ind.is_empty());
        assert!(max_inf.col_ind.is_empty());
        assert_eq!(max_inf.cost, 0.0);
        assert_eq!(max_inf.iterations, 0);
        assert_eq!(max_inf.scaling_rounds, 0);
        assert_eq!(max_inf.final_eps, 0.0);
        assert_eq!(max_inf.row_prices, vec![0.0; 3]);
        assert_eq!(max_inf.col_prices, vec![0.0; 3]);
    }

    #[test]
    fn test_cost_limit_strict_boundary() {
        let costs = [5.0, 10.0, 10.0, 5.0];

        let exact_min = solve_dense(
            &costs,
            2,
            2,
            &SolveConfig {
                cost_limit: Some(5.0),
                ..default_config()
            },
        )
        .expect("exact boundary min should succeed with empty matching");
        assert!(exact_min.row_ind.is_empty());
        assert!(exact_min.col_ind.is_empty());
        assert_eq!(exact_min.cost, 0.0);
        assert_eq!(exact_min.row_prices, vec![0.0; 2]);
        assert_eq!(exact_min.col_prices, vec![0.0; 2]);

        let just_above_min = solve_dense(
            &costs,
            2,
            2,
            &SolveConfig {
                cost_limit: Some(5.01),
                ..default_config()
            },
        )
        .expect("just above boundary min should match");
        assert_partial_matching(&just_above_min, &costs, 2, 2, false, 5.01);
        assert_eq!(just_above_min.row_ind, vec![0, 1]);
        assert_eq!(just_above_min.col_ind, vec![0, 1]);
        assert!((just_above_min.cost - 10.0).abs() < 1e-12);
        assert_partial_eps_cs(&costs, 2, 2, false, 5.01, &just_above_min);

        let exact_max = solve_dense(
            &costs,
            2,
            2,
            &SolveConfig {
                maximize: true,
                cost_limit: Some(10.0),
                ..default_config()
            },
        )
        .expect("exact boundary max should succeed with empty matching");
        assert!(exact_max.row_ind.is_empty());
        assert!(exact_max.col_ind.is_empty());
        assert_eq!(exact_max.cost, 0.0);

        let just_below_max = solve_dense(
            &costs,
            2,
            2,
            &SolveConfig {
                maximize: true,
                cost_limit: Some(9.99),
                ..default_config()
            },
        )
        .expect("just below boundary max should match");
        assert_partial_matching(&just_below_max, &costs, 2, 2, true, 9.99);
        assert_eq!(just_below_max.row_ind, vec![0, 1]);
        assert_eq!(just_below_max.col_ind, vec![1, 0]);
        assert!((just_below_max.cost - 20.0).abs() < 1e-12);
        assert_partial_eps_cs(&costs, 2, 2, true, 9.99, &just_below_max);
    }

    #[test]
    fn test_cost_limit_partial_matching_beats_forced() {
        let inf = f64::INFINITY;
        // Track-fusion matrix: rows 0, 1 match cleanly (1.0, 2.0), row 2's only finite edge is 50.0.
        let fusion = [1.0, inf, inf, inf, 2.0, inf, inf, inf, 50.0];
        let out_min = solve_dense(
            &fusion,
            3,
            3,
            &SolveConfig {
                cost_limit: Some(10.0),
                ..default_config()
            },
        )
        .expect("partial gated fusion should solve");
        assert_partial_matching(&out_min, &fusion, 3, 3, false, 10.0);
        assert_eq!(out_min.row_ind, vec![0, 1]);
        assert_eq!(out_min.col_ind, vec![0, 1]);
        assert!((out_min.cost - 3.0).abs() < 1e-12);
        assert_partial_eps_cs(&fusion, 3, 3, false, 10.0, &out_min);

        // Maximization with negative-reward third track and cost_limit = 0.0
        let ninf = f64::NEG_INFINITY;
        let rewards = [10.0, ninf, ninf, ninf, 2.0, ninf, ninf, ninf, -5.0];
        let out_max = solve_dense(
            &rewards,
            3,
            3,
            &SolveConfig {
                maximize: true,
                cost_limit: Some(0.0),
                ..default_config()
            },
        )
        .expect("partial max fusion should skip negative-reward match");
        assert_partial_matching(&out_max, &rewards, 3, 3, true, 0.0);
        assert_eq!(out_max.row_ind, vec![0, 1]);
        assert_eq!(out_max.col_ind, vec![0, 1]);
        assert!((out_max.cost - 12.0).abs() < 1e-12);
        assert_partial_eps_cs(&rewards, 3, 3, true, 0.0, &out_max);
    }

    #[test]
    fn test_cost_limit_rectangular_and_inf() {
        let inf = f64::INFINITY;
        let ninf = f64::NEG_INFINITY;

        // 4x2 (tall) with inf entries and finite cost_limit
        let tall = [2.0, 9.0, 1.0, inf, 8.5, 4.0, inf, 12.0];
        let limit_tall = 9.0;
        let out_tall = solve_dense(
            &tall,
            4,
            2,
            &SolveConfig {
                cost_limit: Some(limit_tall),
                ..default_config()
            },
        )
        .expect("4x2 partial should solve");
        assert_partial_matching(&out_tall, &tall, 4, 2, false, limit_tall);
        assert_partial_eps_cs(&tall, 4, 2, false, limit_tall, &out_tall);
        let best_tall = brute_force_best_partial_benefit(&tall, 4, 2, false, limit_tall);
        let got_tall = net_benefit_sum(&out_tall, &tall, 2, false, limit_tall);
        assert!(
            (best_tall - got_tall).abs() <= 4.0 * out_tall.final_eps + 1e-9,
            "tall optimality gap: best={best_tall}, got={got_tall}"
        );

        // 2x4 (wide) with inf entries and finite cost_limit
        let wide = [2.0, 1.0, 8.5, inf, 9.0, inf, 4.0, 12.0];
        let limit_wide = 9.0;
        let out_wide = solve_dense(
            &wide,
            2,
            4,
            &SolveConfig {
                cost_limit: Some(limit_wide),
                ..default_config()
            },
        )
        .expect("2x4 partial should solve");
        assert_partial_matching(&out_wide, &wide, 2, 4, false, limit_wide);
        assert_partial_eps_cs(&wide, 2, 4, false, limit_wide, &out_wide);
        let best_wide = brute_force_best_partial_benefit(&wide, 2, 4, false, limit_wide);
        let got_wide = net_benefit_sum(&out_wide, &wide, 4, false, limit_wide);
        assert!(
            (best_wide - got_wide).abs() <= 2.0 * out_wide.final_eps + 1e-9,
            "wide optimality gap: best={best_wide}, got={got_wide}"
        );

        // Maximize on 4x2 with -inf and negative scores
        let tall_max = [8.0, 1.0, 9.0, ninf, 1.5, 6.0, ninf, -2.0];
        let out_tall_max = solve_dense(
            &tall_max,
            4,
            2,
            &SolveConfig {
                maximize: true,
                cost_limit: Some(1.0),
                ..default_config()
            },
        )
        .expect("4x2 max partial should solve");
        assert_partial_matching(&out_tall_max, &tall_max, 4, 2, true, 1.0);
        assert_partial_eps_cs(&tall_max, 4, 2, true, 1.0, &out_tall_max);
        let best_tall_max = brute_force_best_partial_benefit(&tall_max, 4, 2, true, 1.0);
        let got_tall_max = net_benefit_sum(&out_tall_max, &tall_max, 2, true, 1.0);
        assert!(
            (best_tall_max - got_tall_max).abs() <= 4.0 * out_tall_max.final_eps + 1e-9,
            "tall max optimality gap: best={best_tall_max}, got={got_tall_max}"
        );
    }

    #[test]
    fn test_cost_limit_tricky_scaling_and_bruteforce_suite() {
        let inf = f64::INFINITY;
        let ninf = f64::NEG_INFINITY;

        // 4x4 with a 3x2 overdemanded equal-cost subblock (must scale rapidly, not spin)
        let overdemanded = [
            1.0, 1.0, inf, inf, 1.0, 1.0, inf, inf, 1.0, 1.0, inf, inf, inf, inf, 1.0, inf,
        ];
        let out_od = solve_dense(
            &overdemanded,
            4,
            4,
            &SolveConfig {
                cost_limit: Some(5.0),
                ..default_config()
            },
        )
        .expect("overdemanded subblock should solve rapidly");
        assert_partial_matching(&out_od, &overdemanded, 4, 4, false, 5.0);
        assert_eq!(out_od.row_ind.len(), 3);
        assert!((out_od.cost - 3.0).abs() < 1e-12);
        assert!(out_od.iterations < 200);
        assert_partial_eps_cs(&overdemanded, 4, 4, false, 5.0, &out_od);

        // Matrix forcing reverse pass to rescue optimal 2-pair matching after round 1
        // B = [[5.0, 1.0], [5.0, 0.0]] for limit 6.0 -> costs [[1.0, 5.0], [1.0, 6.0]]
        let reverse_rescue = [1.0, 5.0, 1.0, 6.0];
        let out_rev = solve_dense(
            &reverse_rescue,
            2,
            2,
            &SolveConfig {
                cost_limit: Some(6.0),
                ..default_config()
            },
        )
        .expect("reverse rescue 2x2 should solve");
        assert_partial_matching(&out_rev, &reverse_rescue, 2, 2, false, 6.0);
        assert_eq!(out_rev.row_ind, vec![0, 1]);
        assert_eq!(out_rev.col_ind, vec![1, 0]);
        assert!((out_rev.cost - 6.0).abs() < 1e-12);
        assert_partial_eps_cs(&reverse_rescue, 2, 2, false, 6.0, &out_rev);

        // Deterministic PRNG differential suite against brute force over 1..=5 x 1..=5
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next_u64 = || -> u64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for case_idx in 0..80 {
            let n_rows = ((next_u64() % 5) + 1) as usize;
            let n_cols = ((next_u64() % 5) + 1) as usize;
            let maximize = case_idx % 2 == 1;
            let limit = ((next_u64() % 21) as f64) - 10.0;
            let mut costs = Vec::with_capacity(n_rows * n_cols);
            for _ in 0..(n_rows * n_cols) {
                let r = next_u64() % 10;
                if r < 3 {
                    costs.push(if maximize { ninf } else { inf });
                } else if r == 3 {
                    // Exact boundary entry
                    costs.push(limit);
                } else {
                    let v = ((next_u64() % 25) as f64) - 12.0;
                    costs.push(v);
                }
            }

            let out = solve_dense(
                &costs,
                n_rows,
                n_cols,
                &SolveConfig {
                    maximize,
                    cost_limit: Some(limit),
                    ..default_config()
                },
            )
            .expect("random small partial matrix should solve");
            assert_partial_matching(&out, &costs, n_rows, n_cols, maximize, limit);
            assert_partial_eps_cs(&costs, n_rows, n_cols, maximize, limit, &out);

            let best = brute_force_best_partial_benefit(&costs, n_rows, n_cols, maximize, limit);
            let got = net_benefit_sum(&out, &costs, n_cols, maximize, limit);
            assert!(
                (best - got).abs() <= (n_rows.min(n_cols) as f64) * out.final_eps + 1e-9,
                "case {case_idx} ({n_rows}x{n_cols}, max={maximize}, limit={limit}): best={best}, got={got}"
            );
        }

        // ULP stagnation in partial mode with huge net benefit and tiny eps_min
        let ulp_partial = [0.0, 1e12, 0.0, 1e12];
        let res = solve_dense(
            &ulp_partial,
            2,
            2,
            &SolveConfig {
                cost_limit: Some(2e12),
                eps_min: Some(1e-20),
                ..default_config()
            },
        );
        assert!(
            matches!(res, Err(SolverError::Convergence { .. })),
            "expected Convergence on partial ULP stall, got {res:?}"
        );

        // Larger sparse gated partial instances (15x15, 20x8, 8x20) checking dual eps-CS
        for (n_rows, n_cols, maximize, limit) in [
            (15, 15, false, 3.5),
            (20, 8, false, 0.0),
            (8, 20, true, -2.5),
        ] {
            let mut large = Vec::with_capacity(n_rows * n_cols);
            for _ in 0..(n_rows * n_cols) {
                if next_u64() % 4 == 0 {
                    large.push(if maximize { ninf } else { inf });
                } else {
                    let v = ((next_u64() % 400) as f64) * 0.05 - 10.0;
                    large.push(v);
                }
            }
            let out_large = solve_dense(
                &large,
                n_rows,
                n_cols,
                &SolveConfig {
                    maximize,
                    cost_limit: Some(limit),
                    ..default_config()
                },
            )
            .expect("larger sparse partial matrix should solve");
            assert_partial_matching(&out_large, &large, n_rows, n_cols, maximize, limit);
            assert_partial_eps_cs(&large, n_rows, n_cols, maximize, limit, &out_large);
        }

        // Unbounded-benefit infinity still rejected even with finite cost_limit
        assert!(matches!(
            solve_dense(
                &[1.0, ninf, 2.0, 3.0],
                2,
                2,
                &SolveConfig {
                    cost_limit: Some(10.0),
                    ..default_config()
                }
            ),
            Err(SolverError::InvalidInput(_))
        ));
        assert!(matches!(
            solve_dense(
                &[1.0, inf, 2.0, 3.0],
                2,
                2,
                &SolveConfig {
                    maximize: true,
                    cost_limit: Some(0.0),
                    ..default_config()
                }
            ),
            Err(SolverError::InvalidInput(_))
        ));
    }
}
