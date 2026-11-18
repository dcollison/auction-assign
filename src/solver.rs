#![cfg_attr(not(test), allow(dead_code))]

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

struct NormalizedBenefits {
    benefits: Vec<f64>,
    m: usize,
    n: usize,
    transposed: bool,
    has_forbidden: bool,
    b_shift: f64,
    scale: f64,
    half_spread: f64,
    is_constant: bool,
}

struct AuctionState<'a> {
    benefits: &'a [f64],
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

/// Returns `true` iff an `m x n` (`m <= n`) boolean bipartite adjacency matrix admits a matching of size `m`.
fn has_complete_bipartite_matching(feasible: &[bool], m: usize, n: usize) -> bool {
    if m == 0 {
        return true;
    }
    if m > n || feasible.len() != m * n {
        return false;
    }

    // Quick necessary checks: every row must have an edge, and at least m distinct columns must be reachable.
    let mut col_reachable = vec![false; n];
    let mut reachable_cols = 0usize;
    for i in 0..m {
        let row_off = i * n;
        let mut row_ok = false;
        for j in 0..n {
            if feasible[row_off + j] {
                row_ok = true;
                if !col_reachable[j] {
                    col_reachable[j] = true;
                    reachable_cols += 1;
                }
            }
        }
        if !row_ok {
            return false;
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
        let row_off = u * n;
        for (v, col_slot) in pair_v.iter_mut().enumerate() {
            if feasible[row_off + v] && *col_slot == NONE {
                *pair_slot = v;
                *col_slot = u;
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
        // BFS builds layered shortest-augmenting-path graph from unmatched rows.
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
            let row_off = u * n;
            let next_dist = dist[u] + 1;
            for v in 0..n {
                if !feasible[row_off + v] {
                    continue;
                }
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

        // Iterative Dinic-style DFS over layered graph finds vertex-disjoint augmenting paths.
        cur.fill(0);
        for start_u in 0..m {
            if pair_u[start_u] != NONE {
                continue;
            }
            stack.clear();
            stack.push((start_u, NONE));
            let mut augmented = false;

            while let Some(&(u, _)) = stack.last() {
                let row_off = u * n;
                let mut pushed_or_found = false;

                while cur[u] < n {
                    let v = cur[u];
                    cur[u] += 1;
                    if !feasible[row_off + v] {
                        continue;
                    }
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

/// Converts input costs into an `m <= n` row-major benefit matrix with global offset normalization.
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

    let mut benefits = vec![f64::NEG_INFINITY; m * n];
    let mut has_forbidden = false;
    let mut b_min = f64::INFINITY;
    let mut b_max = f64::NEG_INFINITY;
    let mut max_abs_b = 0.0f64;

    for r in 0..n_rows {
        for c in 0..n_cols {
            let val = costs[r * n_cols + c];
            let (i, j) = if transposed { (c, r) } else { (r, c) };
            let benefit = if maximize {
                if val == f64::NEG_INFINITY {
                    has_forbidden = true;
                    f64::NEG_INFINITY
                } else {
                    val
                }
            } else if val == f64::INFINITY {
                has_forbidden = true;
                f64::NEG_INFINITY
            } else {
                -val
            };

            benefits[i * n + j] = benefit;
            if benefit.is_finite() {
                b_min = b_min.min(benefit);
                b_max = b_max.max(benefit);
                max_abs_b = max_abs_b.max(benefit.abs());
            }
        }
    }

    // Shift out large uniform offsets while preserving 0 in-range matrices untouched.
    let b_shift = if b_min > 0.0 {
        b_min
    } else if b_max < 0.0 {
        b_max
    } else {
        0.0
    };
    if b_shift != 0.0 && b_shift.is_finite() {
        for b in &mut benefits {
            if b.is_finite() {
                *b -= b_shift;
            }
        }
    }

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
        benefits,
        m,
        n,
        transposed,
        has_forbidden,
        b_shift,
        scale,
        half_spread,
        is_constant,
    }
}

/// Finds `(best_idx, best_val, second_best_val, feasible_count)` over `(benefit, dual)` pairs.
#[inline]
fn best_two(items: impl Iterator<Item = (f64, f64)>) -> (usize, f64, f64, usize) {
    let mut best_idx = NONE;
    let mut best_val = f64::NEG_INFINITY;
    let mut second_val = f64::NEG_INFINITY;
    let mut feasible_count = 0usize;

    for (idx, (b, dual)) in items.enumerate() {
        if !b.is_finite() {
            continue;
        }
        feasible_count += 1;
        let val = b - dual;
        if best_idx == NONE || val > best_val {
            second_val = best_val;
            best_val = val;
            best_idx = idx;
        } else if feasible_count == 2 || val > second_val {
            second_val = val;
        }
    }

    (best_idx, best_val, second_val, feasible_count)
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

/// Executes one forward bid for unassigned row `row`. Returns `true` if `matched_count` increased.
fn forward_bid(state: &mut AuctionState<'_>, row: usize, eps: f64) -> Result<bool, SolverError> {
    step_iteration(state, eps)?;

    let row_off = row * state.n;
    let (j1, _v1, v2, feasible_count) = best_two(
        state.benefits[row_off..row_off + state.n]
            .iter()
            .copied()
            .zip(state.p.iter().copied()),
    );
    if j1 == NONE {
        return Err(SolverError::Infeasible(format!(
            "row {row} has no feasible columns"
        )));
    }

    let p_old = state.p[j1];
    let (p_new, pi_new) = if feasible_count == 1 {
        let p_new = p_old + eps;
        (p_new, state.benefits[row_off + j1] - p_new)
    } else {
        let pi_new = v2 - eps;
        (state.benefits[row_off + j1] - pi_new, pi_new)
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

/// Executes one symmetric reverse bid for unassigned column `col` (`m == n`). Returns `true` if `matched_count` increased.
fn reverse_bid_symmetric(
    state: &mut AuctionState<'_>,
    col: usize,
    eps: f64,
) -> Result<bool, SolverError> {
    step_iteration(state, eps)?;

    let (i1, _u1, u2, feasible_count) = best_two(
        state
            .pi
            .iter()
            .enumerate()
            .map(|(i, &pi_i)| (state.benefits[i * state.n + col], pi_i)),
    );
    if i1 == NONE {
        return Err(SolverError::Infeasible(format!(
            "column {col} has no feasible rows"
        )));
    }

    let pi_old = state.pi[i1];
    let b_val = state.benefits[i1 * state.n + col];
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

/// Runs one scaling round for square matrices (`m == n`), switching forward/reverse whenever cardinality increases.
fn run_square_round(state: &mut AuctionState<'_>, eps: f64) -> Result<(), SolverError> {
    let mut forward = true;
    while state.matched_count < state.m {
        if forward {
            let mut did_bid = false;
            while let Some(row) = state.unassigned_rows.pop_front() {
                if state.row_to_col[row] != NONE {
                    continue;
                }
                did_bid = true;
                if forward_bid(state, row, eps)? {
                    forward = false;
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
                    forward = true;
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
        forward_bid(state, row, eps)?;
    }

    let mut lambda = min_assigned_price(&state.row_to_col, &state.p);
    let mut rev_queue = VecDeque::new();
    for j in 0..state.n {
        if state.col_to_row[j] == NONE && state.p[j] > lambda {
            rev_queue.push_back(j);
        }
    }

    // Lower any unassigned column with p[j] > lambda to maintain Bertsekas-Castanon asymmetric optimality.
    while let Some(col) = rev_queue.pop_front() {
        if state.col_to_row[col] != NONE || state.p[col] <= lambda {
            continue;
        }
        step_iteration(state, eps)?;

        let (i1, u1, u2, _feasible_count) = best_two(
            state
                .pi
                .iter()
                .enumerate()
                .map(|(i, &pi_i)| (state.benefits[i * state.n + col], pi_i)),
        );
        if i1 == NONE || u1 - eps <= lambda {
            state.p[col] = lambda;
            continue;
        }

        // For single-feasible-row columns (u2 == -inf), (u2 - eps).max(lambda) evaluates cleanly to lambda.
        let p_new = (u2 - eps).max(lambda);
        let pi_old = state.pi[i1];
        let pi_new = state.benefits[i1 * state.n + col] - p_new;
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
        state.col_to_row[prev_col] = NONE;

        lambda = min_assigned_price(&state.row_to_col, &state.p);
        if state.p[prev_col] > lambda {
            rev_queue.push_back(prev_col);
        }
    }

    Ok(())
}

/// Initializes a scaling round by anchoring `min(p) = 0`, clearing assignments, and setting `pi[i] = max_j (b[i,j] - p[j])`.
fn init_scaling_round(state: &mut AuctionState<'_>, is_first_round: bool) {
    if !is_first_round {
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
    if state.m == state.n {
        state.unassigned_cols.clear();
        state.unassigned_cols.extend(0..state.n);
    }

    for (i, pi_slot) in state.pi.iter_mut().enumerate() {
        let row_off = i * state.n;
        let mut best_val = f64::NEG_INFINITY;
        for (&b, &pj) in state.benefits[row_off..row_off + state.n]
            .iter()
            .zip(state.p.iter())
        {
            if b.is_finite() {
                best_val = best_val.max(b - pj);
            }
        }
        *pi_slot = best_val;
    }
}

/// Solves a dense 2D linear assignment problem in row-major layout.
pub fn solve_dense(
    costs: &[f64],
    n_rows: usize,
    n_cols: usize,
    config: &SolveConfig,
) -> Result<SolveOutput, SolverError> {
    validate_inputs(costs, n_rows, n_cols, config)?;

    let effective_unbounded = match config.cost_limit {
        None => true,
        Some(limit) if !config.maximize && limit == f64::INFINITY => true,
        Some(limit) if config.maximize && limit == f64::NEG_INFINITY => true,
        Some(_) => false,
    };
    if !effective_unbounded {
        return Err(SolverError::InvalidInput(
            "finite or empty cost_limit is not yet implemented".to_string(),
        ));
    }

    if n_rows == 0 || n_cols == 0 {
        return Ok(SolveOutput {
            row_ind: Vec::new(),
            col_ind: Vec::new(),
            cost: 0.0,
            iterations: 0,
            scaling_rounds: 0,
            final_eps: 0.0,
            row_prices: vec![0.0; n_rows],
            col_prices: vec![0.0; n_cols],
        });
    }

    let norm = build_complete_benefits(costs, n_rows, n_cols, config.maximize);

    if norm.has_forbidden {
        let feasible: Vec<bool> = norm.benefits.iter().map(|b| b.is_finite()).collect();
        if !has_complete_bipartite_matching(&feasible, norm.m, norm.n) {
            return Err(SolverError::Infeasible(
                "gated cost matrix has no valid complete assignment".to_string(),
            ));
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
        benefits: &norm.benefits,
        m: norm.m,
        n: norm.n,
        pi: vec![0.0; norm.m],
        p: vec![0.0; norm.n],
        row_to_col: vec![NONE; norm.m],
        col_to_row: vec![NONE; norm.n],
        unassigned_rows: VecDeque::with_capacity(norm.m),
        unassigned_cols: if norm.m == norm.n {
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
        init_scaling_round(&mut state, scaling_rounds == 1);
        if state.m == state.n {
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

    if norm.b_shift != 0.0 && norm.b_shift.is_finite() {
        for r_price in &mut state.pi {
            *r_price += norm.b_shift;
        }
    }

    let (row_ind, col_ind, row_prices, col_prices) = if norm.transposed {
        let mut pairs: Vec<(usize, usize)> =
            (0..norm.m).map(|c| (state.row_to_col[c], c)).collect();
        pairs.sort_unstable_by_key(|&(r, _)| r);
        let (rows, cols): (Vec<usize>, Vec<usize>) = pairs.into_iter().unzip();
        (rows, cols, state.p, state.pi)
    } else {
        let rows: Vec<usize> = (0..norm.m).collect();
        (rows, state.row_to_col, state.pi, state.p)
    };

    let cost: f64 = row_ind
        .iter()
        .zip(&col_ind)
        .map(|(&r, &c)| costs[r * n_cols + c])
        .sum();

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

    fn assert_eps_cs(
        costs: &[f64],
        n_rows: usize,
        n_cols: usize,
        maximize: bool,
        out: &SolveOutput,
    ) {
        let tol = 1e-8
            * (out
                .row_prices
                .iter()
                .map(|v| v.abs())
                .fold(1.0_f64, f64::max));
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
    }

    #[test]
    fn test_empty_matrices() {
        let cfg = default_config();
        for (n_rows, n_cols) in [(0, 0), (0, 4), (4, 0)] {
            let out = solve_dense(&[], n_rows, n_cols, &cfg).expect("empty matrix should succeed");
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
}
