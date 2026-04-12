"""
Unified Agent-Based Model: Wind-Power Producer in a Day-Ahead Spot Market
=========================================================================
Specification: ODD protocol (see accompanying model specification document).

Structure
---------
Classes
    Environment          -- shared constants, uncertain parameters, stochastic draws
    ConventionalProducer -- coal producer; stochastic bid price, fixed capacity
    SolarProducer        -- solar producer; cosine bid profile, stochastic price
    WindProducer         -- focal agent (Perspective W); optimises (q, p) bid vector
    MarketOperator       -- merit-order clearing with priority rationing
    SystemRegulator      -- focal agent (Perspective R); sets kappa, evaluates welfare

Data containers
    MarketOutcome        -- cleared outcomes for one interval
    SettlementOutcome    -- wind producer settlement for one interval
    DayOutcome           -- aggregated outcomes across all 24 intervals

Interface
---------
    run_simulation(...)  -- single unified entry point; supports four modes:
                              'simulate'    fixed kappa, fixed u_W
                              'optimise_W'  fixed kappa, optimise u_W
                              'optimise_R'  fixed u_W,   optimise kappa
                              'stackelberg' joint Stackelberg equilibrium

Dependencies: numpy only (standard library otherwise)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import csv
import numpy as np
from numpy.random import default_rng


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MarketOutcome:
    """Cleared outcomes for a single interval t."""
    t: int
    clearing_price: float  # lambda_t  [$/MWh]
    dispatch: dict  # {agent_id: d^i_t [MWh]}
    demand: float  # D_t       [MWh]
    marginal_cohort: list  # M_t       list of agent ids


@dataclass
class SettlementOutcome:
    """Settlement outcomes for the wind producer in a single interval t."""
    t: int
    clearing_price: float  # lambda_t   [$/MWh]
    dispatched_quantity: float  # d^W_t      [MWh]
    actual_production: float  # f^W_t      [MWh]
    shortfall: float  # Delta^W_t  [MWh]
    penalty_cost: float  # kappa * Delta^W_t  [$]
    net_revenue: float  # Pi^W_t     [$]


@dataclass
class DayOutcome:
    """Aggregated outcomes across all 24 intervals for one simulated day."""
    market: list  # list[MarketOutcome]
    settlement: list  # list[SettlementOutcome]
    wind_net_revenue: float  # Pi^W        [$/day]
    social_welfare: float  # W           [$/day]
    balancing_cost: float  # sum C^sys_t [$/day]
    consumer_surplus: float  # [$/day]
    producer_surplus: float  # [$/day]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Environment:
    """
    Shared environment: constants, uncertain parameters, stochastic draws.

    All agents hold a reference to a single Environment instance, ensuring
    full parameter consistency across perspectives.

    Parameters
    ----------
    Q_conv : array-like of shape (3,)
        Installed capacities [MWh] for conventional producers C1, C2, C3.
    a : float
        Vertical offset of solar generation profile [MWh].
    b : float
        Amplitude of solar generation profile [MWh]; must be strictly negative.
        With b < 0 the cosine term peaks at t=12 (noon) and troughs at t=0/24
        (midnight), matching physical solar irradiance.
    p_floor, p_cap : float
        Market price floor and cap [$/MWh].
    kappa_floor, kappa_cap : float
        Admissible range [low, high] for the regulator's penalty rate [$/MWh].
    c_bal : float or array-like of shape (24,)
        System balancing cost per interval [$/MWh]; scalar is broadcast to all
        24 intervals.
    V : float
        Consumers' constant marginal value of electricity [$/MWh].
    Q_wind : float
        Installed wind capacity [MWh]; upper bound on bid quantity q^W_t.
        Decoupled from mu_W_q: the producer may commit up to its full
        installed capacity regardless of its expected output.
    mu_D, sigma2_D : float
        Mean and variance of aggregate demand distribution [MWh, MWh²].
    mu_W_q, sigma2_W_q : float
        Mean and variance of actual wind production distribution [MWh, MWh²].
    prior_conv_mu_p : array-like of shape (3, 2)
        Per-producer uniform prior [low, high] for uncertain mean bid price.
    prior_conv_sigma2_p : array-like of shape (3, 2)
        Per-producer uniform prior [low, high] for uncertain bid price variance.
    prior_solar_mu_p : array-like of shape (2,)
        Uniform prior [low, high] for solar producer's uncertain mean bid price.
    prior_solar_sigma2_p : array-like of shape (2,)
        Uniform prior [low, high] for solar producer's uncertain bid price variance.
    seed : int, optional
        Master random seed for full reproducibility.
    """

    # Canonical priority order for marginal cohort rationing (Section I.3)
    PRIORITY_ORDER: list = ["solar", "wind", "conv1", "conv2", "conv3"]

    def __init__(
            self,
            Q_conv: list,
            a: float,
            b: float,
            p_floor: float,
            p_cap: float,
            kappa_floor: float,
            kappa_cap: float,
            c_bal,
            V: float,
            mu_D: float,
            sigma2_D: float,
            Q_wind: float,
            mu_W_q: float,
            sigma2_W_q: float,
            prior_conv_mu_p: list,
            prior_conv_sigma2_p: list,
            prior_solar_mu_p: list,
            prior_solar_sigma2_p: list,
            seed: Optional[int] = None,
    ):
        if b >= 0:
            raise ValueError(
                "Solar amplitude b must be strictly negative (b < 0). "
                "With b < 0 the cosine profile peaks at noon and is zero at night."
            )
        if len(Q_conv) != 3:
            raise ValueError("Q_conv must contain exactly 3 capacity values.")

        self.Q_conv = np.asarray(Q_conv, dtype=float)
        self.a = float(a)
        self.b = float(b)
        self.p_floor = float(p_floor)
        self.p_cap = float(p_cap)
        self.kappa_floor = float(kappa_floor)
        self.kappa_cap = float(kappa_cap)
        self.c_bal = (
            np.full(24, float(c_bal)) if np.isscalar(c_bal)
            else np.asarray(c_bal, dtype=float)
        )
        self.V = float(V)
        self.mu_D = float(mu_D)
        self.sigma2_D = float(sigma2_D)
        self.Q_wind = float(Q_wind)
        self.mu_W_q = float(mu_W_q)
        self.sigma2_W_q = float(sigma2_W_q)

        # Priors for uncertain parameters (uniform [low, high])
        self.prior_conv_mu_p = np.asarray(prior_conv_mu_p, dtype=float)  # (3,2)
        self.prior_conv_sigma2_p = np.asarray(prior_conv_sigma2_p, dtype=float)  # (3,2)
        self.prior_solar_mu_p = np.asarray(prior_solar_mu_p, dtype=float)  # (2,)
        self.prior_solar_sigma2_p = np.asarray(prior_solar_sigma2_p, dtype=float)  # (2,)

        self.rng = default_rng(seed)

    # ------------------------------------------------------------------
    # Stochastic draws
    # ------------------------------------------------------------------

    def draw_uncertain_parameters(self) -> dict:
        """
        Draw uncertain parameters once per simulation run (Step 2 of schedule).

        These are held constant for the entire day: they represent structural
        uncertainty about competitor bid distributions, not interval-level noise.

        Returns
        -------
        dict with keys:
            conv_mu_p     : ndarray (3,) -- mean bid prices for C1, C2, C3
            conv_sigma2_p : ndarray (3,) -- bid price variances for C1, C2, C3
            solar_mu_p    : float        -- mean bid price for solar producer
            solar_sigma2_p: float        -- bid price variance for solar producer
        """
        conv_mu_p = np.array([
            self.rng.uniform(*self.prior_conv_mu_p[k]) for k in range(3)
        ])
        conv_sigma2_p = np.array([
            self.rng.uniform(*self.prior_conv_sigma2_p[k]) for k in range(3)
        ])
        solar_mu_p = float(self.rng.uniform(*self.prior_solar_mu_p))
        solar_sigma2_p = float(self.rng.uniform(*self.prior_solar_sigma2_p))
        return {
            "conv_mu_p": conv_mu_p,
            "conv_sigma2_p": conv_sigma2_p,
            "solar_mu_p": solar_mu_p,
            "solar_sigma2_p": solar_sigma2_p,
        }

    def draw_demand(self) -> np.ndarray:
        """
        Draw demand realisations for all 24 intervals (eq. 1 / eq. 7).

        D_t ~ N(mu_D, sigma2_D), i.i.d. across t; floored at epsilon > 0
        to enforce strict positivity (inelastic demand assumption).

        Returns
        -------
        ndarray of shape (24,)
        """
        d = self.rng.normal(self.mu_D, np.sqrt(self.sigma2_D), size=24)
        return np.maximum(d, 1e-6)

    def draw_wind_production(self) -> np.ndarray:
        """
        Draw actual wind production realisations for all 24 intervals (eq. 6 / eq. 13).

        f^W_t ~ N(mu_W_q, sigma2_W_q), i.i.d. across t; floored at 0
        to enforce physical non-negativity.

        Returns
        -------
        ndarray of shape (24,)
        """
        f = self.rng.normal(self.mu_W_q, np.sqrt(self.sigma2_W_q), size=24)
        return np.maximum(f, 0.0)

    def solar_profile(self) -> np.ndarray:
        """
        Deterministic solar bid profile for t = 1..24 (eq. 3 / eq. 4).

        q^S_t = max(0, a + b * cos(2*pi*t/24)), with b < 0.
        Profile peaks at t=12 (noon) and is zero during the overnight window
        where a + b*cos(2*pi*t/24) <= 0.  The non-trivial zero-output window
        requires a < |b|.

        Returns
        -------
        ndarray of shape (24,)
        """
        t = np.arange(1, 25)
        profile = self.a + self.b * np.cos(2.0 * np.pi * t / 24.0)
        return np.maximum(profile, 0.0)


# ---------------------------------------------------------------------------
# Conventional producer
# ---------------------------------------------------------------------------

class ConventionalProducer:
    """
    Coal producer: bids full installed capacity at a stochastic price.

    Behavioural logic (Section I.5 / eq. 1-2 / eq. 14-15)
    -------------------------------------------------------
    - Bid quantity: q^{C_k}_t = Q^{C_k}  (constant, full capacity commitment)
    - Bid price:    p^{C_k}_t ~ N(mu^{C_k}_p, sigma2^{C_k}_p), truncated to
                    [p_floor, p_cap]

    This agent is non-strategic and price-taking; it has no objective function
    and does not respond to kappa.

    Parameters
    ----------
    agent_id : str   -- unique identifier, e.g. 'conv1'
    capacity : float -- installed capacity Q^{C_k} [MWh]
    env      : Environment
    """

    def __init__(self, agent_id: str, capacity: float, env: Environment):
        self.agent_id = agent_id
        self.capacity = float(capacity)
        self.env = env

    def get_bids(self, mu_p: float, sigma2_p: float) -> tuple:
        """
        Return bid quantities and prices for all 24 intervals.

        Parameters
        ----------
        mu_p, sigma2_p : float
            Uncertain parameters for this simulation run (drawn by Environment).

        Returns
        -------
        quantities : ndarray (24,) -- constant at self.capacity
        prices     : ndarray (24,) -- i.i.d. N(mu_p, sigma2_p), truncated
        """
        quantities = np.full(24, self.capacity)
        prices = self.env.rng.normal(mu_p, np.sqrt(sigma2_p), size=24)
        prices = np.clip(prices, self.env.p_floor, self.env.p_cap)
        return quantities, prices


# ---------------------------------------------------------------------------
# Solar producer
# ---------------------------------------------------------------------------

class SolarProducer:
    """
    Solar producer: bids a deterministic cosine profile at a stochastic price.

    Behavioural logic (Section I.5 / eq. 3-4 / eq. 16-17)
    -------------------------------------------------------
    - Bid quantity: q^S_t = max(0, a + b*cos(2*pi*t/24)),  b < 0
    - Bid price:    p^S_t ~ N(mu^S_p, sigma2^S_p), truncated to [p_floor, p_cap]

    This agent is non-strategic; it has no objective and does not respond to kappa.

    Parameters
    ----------
    agent_id : str  -- unique identifier ('solar')
    env      : Environment
    """

    def __init__(self, agent_id: str, env: Environment):
        self.agent_id = agent_id
        self.env = env

    def get_bids(self, mu_p: float, sigma2_p: float) -> tuple:
        """
        Return bid quantities and prices for all 24 intervals.

        Returns
        -------
        quantities : ndarray (24,) -- cosine profile, max(0, a + b*cos(...))
        prices     : ndarray (24,) -- i.i.d. N(mu_p, sigma2_p), truncated
        """
        quantities = self.env.solar_profile()
        prices = self.env.rng.normal(mu_p, np.sqrt(sigma2_p), size=24)
        prices = np.clip(prices, self.env.p_floor, self.env.p_cap)
        return quantities, prices


# ---------------------------------------------------------------------------
# Wind producer  (Perspective W focal agent)
# ---------------------------------------------------------------------------

class WindProducer:
    """
    Wind producer: strategic focal agent (Perspective W).

    State variables (Section II.1)
    --------------------------------
    f_W_t    : actual wind production -- stochastic, realised at delivery
    lambda_t : market-clearing price  -- observed post-clearing
    d_W_t    : dispatched quantity    -- outcome of clearing
    Delta_t  : shortfall = max(0, d_W_t - f_W_t)
    Pi_W_t   : net revenue per interval

    Control variables (Section II.2 / eq. 5-6 / eq. 12-13)
    ---------------------------------------------------------
    q^W_t in [0, Q_wind]      -- bid quantity [MWh]
    p^W_t in [p_floor, p_cap] -- bid price    [$/MWh]

    The bid vector u_W = {(q^W_t, p^W_t)}_{t=1..24} may be:
      (a) set externally as an exogenous input, or
      (b) optimised via Monte Carlo grid search (see optimise_bids).

    Objective (Section II.4 / eq. 18-19)
    --------------------------------------
    max_{u_W}  E[Pi^W]
      = max sum_t E[lambda_t * d^W_t - kappa * max(0, d^W_t - f^W_t)]
    s.t. 0 <= q^W_t <= Q_wind,  p_floor <= p^W_t <= p_cap

    The upper bound on q^W_t is the installed wind capacity Q_wind, not the
    mean actual production mu_W_q.  Bidding above mu_W_q increases expected
    shortfall risk but is a legitimate strategic choice, particularly when
    the penalty rate kappa is low relative to the clearing price.

    Parameters
    ----------
    agent_id : str  -- unique identifier ('wind')
    env      : Environment
    """

    def __init__(self, agent_id: str, env: Environment):
        self.agent_id = agent_id
        self.env = env
        self._q_bids: Optional[np.ndarray] = None
        self._p_bids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Bid management
    # ------------------------------------------------------------------

    def set_bids(self, q_bids: np.ndarray, p_bids: np.ndarray) -> None:
        """
        Set exogenous bid vector (used in 'simulate' and 'optimise_R' modes).

        Parameters
        ----------
        q_bids : array-like (24,) -- bid quantities, clipped to [0, Q_wind]
        p_bids : array-like (24,) -- bid prices,    clipped to [p_floor, p_cap]
        """
        q_bids = np.asarray(q_bids, dtype=float)
        p_bids = np.asarray(p_bids, dtype=float)
        if q_bids.shape != (24,) or p_bids.shape != (24,):
            raise ValueError("q_bids and p_bids must each have shape (24,).")
        self._q_bids = np.clip(q_bids, 0.0, self.env.Q_wind)
        self._p_bids = np.clip(p_bids, self.env.p_floor, self.env.p_cap)

    def get_bids(self) -> tuple:
        """
        Return current bid vector (copies).

        Falls back to the price-taking default if no bids have been set:
          q = mu_W_q for all t  (full mean capacity)
          p = p_floor for all t (zero marginal cost)
        """
        if self._q_bids is None:
            q = np.full(24, self.env.mu_W_q)
            p = np.full(24, self.env.p_floor)
        else:
            q = self._q_bids.copy()
            p = self._p_bids.copy()
        return q, p

    # ------------------------------------------------------------------
    # Settlement (process II.3 / eq. 14-16)
    # ------------------------------------------------------------------

    def settle(
            self,
            t: int,
            clearing_price: float,
            dispatched_qty: float,
            actual_production: float,
            kappa: float,
    ) -> SettlementOutcome:
        """
        Compute net revenue for a single interval after delivery.

        Pi^W_t = lambda_t * d^W_t - kappa * max(0, d^W_t - f^W_t)

        Surplus production (f^W_t > d^W_t) is curtailed without compensation.

        Parameters
        ----------
        t                 : interval index (1-based)
        clearing_price    : lambda_t [$/MWh]
        dispatched_qty    : d^W_t [MWh]
        actual_production : f^W_t [MWh] -- realised at delivery
        kappa             : penalty rate [$/MWh] -- set by regulator
        """
        shortfall = max(0.0, dispatched_qty - actual_production)
        penalty = kappa * shortfall
        net_revenue = clearing_price * dispatched_qty - penalty
        return SettlementOutcome(
            t=t,
            clearing_price=clearing_price,
            dispatched_quantity=dispatched_qty,
            actual_production=actual_production,
            shortfall=shortfall,
            penalty_cost=penalty,
            net_revenue=net_revenue,
        )

    # ------------------------------------------------------------------
    # Optimisation  (objective II.4 / eq. 18-19)
    # ------------------------------------------------------------------

    def optimise_bids(
            self,
            kappa: float,
            n_mc: int = 1000,
            n_grid_q: int = 10,
            n_grid_p: int = 10,
            market_operator: Optional["MarketOperator"] = None,
            competitors: Optional[dict] = None,
    ) -> tuple:
        """
        Maximise E[Pi^W] over a uniform grid of (q, p) candidates via Monte Carlo.

        A constant bid is used across all 24 intervals (symmetric strategy).
        The grid search is a tractable surrogate; finer resolution or gradient-free
        methods (e.g. CMA-ES, Nelder-Mead) can replace it in production use.

        Trade-off structure (Section II.4)
        ------------------------------------
        - Higher q_W: more revenue when dispatched; larger E[Delta_t] => larger
          expected penalty.
        - Higher p_W: lower dispatch probability; higher lambda_t when accepted.
        - Optimal (q*, p*) balances these forces given kappa.

        Parameters
        ----------
        kappa          : float -- announced penalty rate ($/MWh); treated as known
        n_mc           : int   -- Monte Carlo samples per candidate
        n_grid_q       : int   -- grid points for q in [0, Q_wind]
        n_grid_p       : int   -- grid points for p in [p_floor, p_cap]
        market_operator: MarketOperator instance (required)
        competitors    : dict {agent_id: producer} (required)

        Returns
        -------
        best_q, best_p : ndarray (24,) -- optimal constant bid vector
        """
        if market_operator is None or competitors is None:
            raise ValueError(
                "market_operator and competitors are required for bid optimisation."
            )

        q_grid = np.linspace(0.0, self.env.Q_wind, n_grid_q)
        p_grid = np.linspace(self.env.p_floor, self.env.p_cap, n_grid_p)

        best_revenue = -np.inf
        best_q = np.full(24, self.env.Q_wind)
        best_p = np.full(24, self.env.p_floor)

        for q_val in q_grid:
            for p_val in p_grid:
                q_candidate = np.full(24, q_val)
                p_candidate = np.full(24, p_val)
                self.set_bids(q_candidate, p_candidate)

                mc_revenues = []
                for _ in range(n_mc):
                    params = self.env.draw_uncertain_parameters()
                    bids = _collect_bids(competitors, self, params)
                    demand = self.env.draw_demand()
                    wind_prod = self.env.draw_wind_production()

                    day_rev = 0.0
                    for t_idx in range(24):
                        t = t_idx + 1
                        interval_bids = {
                            aid: (bids[aid][0][t_idx], bids[aid][1][t_idx])
                            for aid in bids
                        }
                        mo = market_operator.clear_interval(
                            t=t,
                            demand=demand[t_idx],
                            bids=interval_bids,
                        )
                        d_w = mo.dispatch.get(self.agent_id, 0.0)
                        so = self.settle(
                            t, mo.clearing_price, d_w, wind_prod[t_idx], kappa
                        )
                        day_rev += so.net_revenue
                    mc_revenues.append(day_rev)

                mean_rev = float(np.mean(mc_revenues))
                if mean_rev > best_revenue:
                    best_revenue = mean_rev
                    best_q = q_candidate.copy()
                    best_p = p_candidate.copy()

        self.set_bids(best_q, best_p)
        return best_q, best_p


# ---------------------------------------------------------------------------
# Market operator
# ---------------------------------------------------------------------------

class MarketOperator:
    """
    Passive infrastructure agent: merit-order clearing with priority rationing.

    Clearing mechanism (Section I.6 / Submodel B / eq. 7-12)
    ----------------------------------------------------------
    Stage 1 -- Merit-order ranking and price setting
        Rank bids ascending by price; cumulate quantities until demand is met.
        Set lambda_t = bid price of the last accepted (marginal) producer.

    Stage 2 -- Priority rationing within the marginal cohort
        When multiple producers bid at lambda_t and collectively exceed residual
        demand, dispatch sequentially under:
            solar > wind > conv1 > conv2 > conv3
        The last accepted producer may be partially dispatched to exactly meet
        residual demand.

    This agent has no objective; it exercises no discretion beyond these rules.

    Parameters
    ----------
    env : Environment
    """

    PRIORITY: list = ["solar", "wind", "conv1", "conv2", "conv3"]

    def __init__(self, env: Environment):
        self.env = env

    def clear_interval(
            self,
            t: int,
            demand: float,
            bids: dict,
    ) -> MarketOutcome:
        """
        Clear a single interval.

        Parameters
        ----------
        t      : int   -- interval index (1-based)
        demand : float -- D_t [MWh]
        bids   : dict  -- {agent_id: (quantity [MWh], price [$/MWh])}

        Returns
        -------
        MarketOutcome
        """
        # Stage 1: sort by price ascending (merit order)
        sorted_agents = sorted(bids.keys(), key=lambda a: bids[a][1])

        # Find marginal producer and clearing price (eq. 8)
        cumulative = 0.0
        clearing_price = self.env.p_cap  # default if supply never meets demand
        for agent_id in sorted_agents:
            q = bids[agent_id][0]
            cumulative += q
            if cumulative >= demand:
                clearing_price = bids[agent_id][1]
                break

        # Identify cohorts (eq. 9)
        infra = [a for a in bids if bids[a][1] < clearing_price]
        marginal_cohort = [a for a in bids if bids[a][1] == clearing_price]

        dispatch: dict = {}

        # Infra-marginal: fully dispatched (eq. 11, first case)
        for a in infra:
            dispatch[a] = bids[a][0]

        # Residual demand at the margin (eq. 10)
        residual = max(demand - sum(dispatch.values()), 0.0)

        # Stage 2: priority rationing within marginal cohort (eq. 11)
        priority_sorted = sorted(
            marginal_cohort,
            key=lambda a: (
                self.PRIORITY.index(a) if a in self.PRIORITY else len(self.PRIORITY)
            ),
        )
        for a in priority_sorted:
            alloc = min(bids[a][0], residual)
            dispatch[a] = alloc
            residual = max(residual - alloc, 0.0)
            if residual < 1e-9:
                break

        # Supra-marginal: zero dispatch (eq. 11, third case)
        for a in bids:
            if a not in dispatch:
                dispatch[a] = 0.0

        return MarketOutcome(
            t=t,
            clearing_price=clearing_price,
            dispatch=dispatch,
            demand=demand,
            marginal_cohort=marginal_cohort,
        )


# ---------------------------------------------------------------------------
# System regulator  (Perspective R focal agent)
# ---------------------------------------------------------------------------

class SystemRegulator:
    """
    Welfare-maximising Stackelberg leader (Perspective R).

    State variables (Section III.1)
    ---------------------------------
    lambda_t, d_W_t, f_W_t : observed post-clearing/delivery
    Delta_t                 : shortfall
    C_sys_t                 : system balancing cost = c_bal_t * Delta_t
    W                       : total daily social welfare

    Control variable (Section III.2 / eq. 19)
    -------------------------------------------
    kappa in [kappa_floor, kappa_cap]  [$/MWh]
    Set once before bid submission; publicly announced to all producers.

    Objective (Section III.4 / eq. 20-22)
    ----------------------------------------
    max_{kappa} E[W | u_W(kappa)]

    where W = sum_t [ V*D_t - lambda_t*D_t + sum_i Pi^i_t - C^sys_t ]

    Penalty payments kappa*Delta_t are transfers and net out of W;
    only C^sys_t enters as a true social cost.

    Stackelberg structure
    ---------------------
    The regulator is the leader: it announces kappa before bid submission.
    The wind producer is the follower: it observes kappa and optimises u_W.
    The regulator must anticipate u_W(kappa) when selecting kappa.

    Parameters
    ----------
    env : Environment
    """

    def __init__(self, env: Environment):
        self.env = env
        self._kappa: float = env.kappa_floor

    # ------------------------------------------------------------------
    # Penalty management
    # ------------------------------------------------------------------

    def set_kappa(self, kappa: float) -> None:
        """Set exogenous penalty rate (used in 'simulate' and 'optimise_W' modes)."""
        kappa = float(kappa)
        if not (self.env.kappa_floor <= kappa <= self.env.kappa_cap):
            warnings.warn(
                f"kappa={kappa:.4f} outside admissible range "
                f"[{self.env.kappa_floor}, {self.env.kappa_cap}]; clipping.",
                stacklevel=2,
            )
            kappa = float(np.clip(kappa, self.env.kappa_floor, self.env.kappa_cap))
        self._kappa = kappa

    def get_kappa(self) -> float:
        """Return the currently active penalty rate."""
        return self._kappa

    # ------------------------------------------------------------------
    # Welfare accounting  (process III.3 / eq. 20-21)
    # ------------------------------------------------------------------

    def compute_welfare(
            self,
            market_outcomes: list,
            settlement_outcomes: list,
            all_producer_revenues: dict,
    ) -> dict:
        """
        Compute social welfare components for a single simulated day.

        W = sum_t [ V*D_t - lambda_t*D_t + sum_i Pi^i_t - C^sys_t ]

        Penalty payments are transfers and therefore net out of W.

        Parameters
        ----------
        market_outcomes       : list[MarketOutcome]  -- 24 interval outcomes
        settlement_outcomes   : list[SettlementOutcome]  -- 24 wind settlements
        all_producer_revenues : dict {agent_id: [Pi^i_t for t=1..24]}

        Returns
        -------
        dict with keys: consumer_surplus, producer_surplus,
                        balancing_cost, social_welfare
        """
        consumer_surplus = sum(
            (self.env.V - mo.clearing_price) * mo.demand
            for mo in market_outcomes
        )
        balancing_cost = sum(
            self.env.c_bal[so.t - 1] * so.shortfall
            for so in settlement_outcomes
        )
        producer_surplus = sum(
            sum(revenues)
            for revenues in all_producer_revenues.values()
        )
        social_welfare = consumer_surplus + producer_surplus - balancing_cost

        return {
            "consumer_surplus": consumer_surplus,
            "producer_surplus": producer_surplus,
            "balancing_cost": balancing_cost,
            "social_welfare": social_welfare,
        }

    # ------------------------------------------------------------------
    # Optimisation  (objective III.4 / eq. 20-22)
    # ------------------------------------------------------------------

    def optimise_kappa(
            self,
            wind_producer: WindProducer,
            market_operator: MarketOperator,
            competitors: dict,
            n_kappa: int = 10,
            n_mc: int = 500,
            n_grid_q: int = 5,
            n_grid_p: int = 5,
    ) -> float:
        """
        Grid search over kappa to maximise E[W | u_W(kappa)].

        Implements the Stackelberg bi-level approach (eq. 21-22):
          Outer loop : searches over kappa grid
          Inner loop : solves wind producer's best-response u_W(kappa)
                       then evaluates E[W] at each candidate kappa

        Trade-off (Section III.4)
        --------------------------
        - kappa too low  => wind producer bids aggressively; large shortfalls;
          high balancing costs; social welfare reduced.
        - kappa too high => wind producer bids conservatively; low renewable
          dispatch; consumer surplus and wind revenue reduced.
        - kappa* aligns private marginal imbalance cost with c_bal_t.

        Parameters
        ----------
        wind_producer   : WindProducer -- follower agent
        market_operator : MarketOperator
        competitors     : dict {agent_id: producer}
        n_kappa         : int -- grid resolution for kappa
        n_mc            : int -- Monte Carlo samples for welfare evaluation
        n_grid_q, n_grid_p : int -- bid grid resolution for inner optimisation

        Returns
        -------
        kappa_star : float -- welfare-maximising penalty rate
        """
        kappa_grid = np.linspace(self.env.kappa_floor, self.env.kappa_cap, n_kappa)

        best_welfare = -np.inf
        kappa_star = float(kappa_grid[0])

        for kappa in kappa_grid:
            # Inner loop: wind producer best response u_W(kappa)
            wind_producer.optimise_bids(
                kappa=kappa,
                n_mc=max(n_mc // 4, 50),
                n_grid_q=n_grid_q,
                n_grid_p=n_grid_p,
                market_operator=market_operator,
                competitors=competitors,
            )

            # Outer loop: evaluate E[W | u_W(kappa)]
            welfares = [
                _run_one_day(
                    env=self.env,
                    wind_producer=wind_producer,
                    market_operator=market_operator,
                    competitors=competitors,
                    regulator=self,
                    kappa=float(kappa),
                ).social_welfare
                for _ in range(n_mc)
            ]

            mean_welfare = float(np.mean(welfares))
            if mean_welfare > best_welfare:
                best_welfare = mean_welfare
                kappa_star = float(kappa)

        self.set_kappa(kappa_star)
        return kappa_star


# ---------------------------------------------------------------------------
# Internal helpers  (not part of the public API)
# ---------------------------------------------------------------------------

def _collect_bids(
        competitors: dict,
        wind_producer: WindProducer,
        params: dict,
) -> dict:
    """
    Collect bid arrays for all agents for one simulation run.

    Returns
    -------
    dict {agent_id: (quantities ndarray(24,), prices ndarray(24,))}
    """
    bids: dict = {}

    # Conventional producers C1, C2, C3
    for k, key in enumerate(["conv1", "conv2", "conv3"]):
        agent = competitors[key]
        q, p = agent.get_bids(
            mu_p=params["conv_mu_p"][k],
            sigma2_p=params["conv_sigma2_p"][k],
        )
        bids[agent.agent_id] = (q, p)

    # Solar producer
    solar = competitors["solar"]
    q_s, p_s = solar.get_bids(
        mu_p=params["solar_mu_p"],
        sigma2_p=params["solar_sigma2_p"],
    )
    bids[solar.agent_id] = (q_s, p_s)

    # Wind producer
    q_w, p_w = wind_producer.get_bids()
    bids[wind_producer.agent_id] = (q_w, p_w)

    return bids


def _run_one_day(
        env: Environment,
        wind_producer: WindProducer,
        market_operator: MarketOperator,
        competitors: dict,
        regulator: SystemRegulator,
        kappa: float,
) -> DayOutcome:
    """
    Simulate one complete day: stochastic draws -> clearing -> settlement.

    Execution follows the 7-step schedule (Section 1.3):
      Steps 1-3 (penalty + bid formation) are assumed done before calling here.
      Steps 4-7 are executed here.

    Returns
    -------
    DayOutcome
    """
    # Step 2: draw uncertain parameters (structural, held fixed for the day)
    params = env.draw_uncertain_parameters()

    # Step 3: collect bids (wind bids already set by caller)
    bids = _collect_bids(competitors, wind_producer, params)

    # Step 4: draw interval-level stochastics
    demand_series = env.draw_demand()
    wind_prod_series = env.draw_wind_production()

    market_outcomes: list = []
    settlement_outcomes: list = []
    all_revenues: dict = {aid: [] for aid in bids}

    for t_idx in range(24):
        t = t_idx + 1

        # Steps 4-5: market clearing
        interval_bids = {
            aid: (bids[aid][0][t_idx], bids[aid][1][t_idx]) for aid in bids
        }
        mo = market_operator.clear_interval(
            t=t, demand=demand_series[t_idx], bids=interval_bids
        )
        market_outcomes.append(mo)

        # Step 6: settlement -- wind producer
        d_w = mo.dispatch.get(wind_producer.agent_id, 0.0)
        so = wind_producer.settle(
            t=t,
            clearing_price=mo.clearing_price,
            dispatched_qty=d_w,
            actual_production=wind_prod_series[t_idx],
            kappa=kappa,
        )
        settlement_outcomes.append(so)

        # Revenue for all agents
        for aid in bids:
            if aid == wind_producer.agent_id:
                all_revenues[aid].append(so.net_revenue)
            else:
                all_revenues[aid].append(
                    mo.clearing_price * mo.dispatch.get(aid, 0.0)
                )

    # Step 7: welfare accounting
    welfare_components = regulator.compute_welfare(
        market_outcomes, settlement_outcomes, all_revenues
    )

    return DayOutcome(
        market=market_outcomes,
        settlement=settlement_outcomes,
        wind_net_revenue=sum(s.net_revenue for s in settlement_outcomes),
        social_welfare=welfare_components["social_welfare"],
        balancing_cost=welfare_components["balancing_cost"],
        consumer_surplus=welfare_components["consumer_surplus"],
        producer_surplus=welfare_components["producer_surplus"],
    )


# ---------------------------------------------------------------------------
# Interface function
# ---------------------------------------------------------------------------

def run_simulation(
        # -- Environment parameters --
        Q_conv: list,
        a: float,
        b: float,
        p_floor: float,
        p_cap: float,
        kappa_floor: float,
        kappa_cap: float,
        c_bal,
        V: float,
        mu_D: float,
        sigma2_D: float,
        Q_wind: float,
        mu_W_q: float,
        sigma2_W_q: float,
        prior_conv_mu_p: list,
        prior_conv_sigma2_p: list,
        prior_solar_mu_p: list,
        prior_solar_sigma2_p: list,
        # -- Evaluation mode --
        mode: str = "simulate",
        # -- Exogenous controls --
        kappa_exog: float = 0.0,
        q_W_exog: Optional[list] = None,
        p_W_exog: Optional[list] = None,
        # -- Monte Carlo settings --
        n_days: int = 1000,
        n_mc_opt: int = 500,
        n_grid_q: int = 10,
        n_grid_p: int = 10,
        n_kappa: int = 10,
        # -- Misc --
        seed: Optional[int] = None,
        verbose: bool = False,
) -> dict:
    """
    Unified entry point for the wind-market agent-based model.

    Evaluation modes
    ----------------
    'simulate'
        Both kappa and u_W are fixed as exogenous inputs.
        Runs n_days Monte Carlo days and returns summary statistics.
        Use for: scenario analysis, sensitivity testing, model validation.

    'optimise_W'  (Perspective W active; Perspective R passive)
        kappa is fixed at kappa_exog.
        Wind producer's bid vector u_W is optimised via Monte Carlo grid
        search (n_grid_q x n_grid_p candidates, n_mc_opt samples each).
        Then n_days evaluation days are run under the optimal u_W.
        Use for: investment assessment from the wind producer's perspective.

    'optimise_R'  (Perspective R active; Perspective W passive)
        u_W is fixed at (q_W_exog, p_W_exog).
        Regulator's penalty rate kappa is optimised via grid search over
        n_kappa candidates.
        Then n_days evaluation days are run under the optimal kappa.
        Use for: regulatory design from the system regulator's perspective.

    'stackelberg'  (both perspectives active; Stackelberg equilibrium)
        Regulator is Stackelberg leader; wind producer is follower.
        Bi-level optimisation: outer loop over kappa; inner loop solves
        u_W(kappa) at each candidate; outer loop evaluates E[W | u_W(kappa)].
        Equilibrium (kappa*, u_W*) satisfies eq. 21-22.
        Then n_days evaluation days are run at the equilibrium.
        Use for: joint investment and regulatory assessment.

    Parameters
    ----------
    Q_conv : list[float] of length 3
        Installed capacities [MWh] for conventional producers C1, C2, C3.
    a : float
        Vertical offset of the solar generation profile [MWh].
    b : float
        Amplitude of the solar profile [MWh]; must be < 0.
    p_floor, p_cap : float
        Market price floor and cap [$/MWh].
    kappa_floor, kappa_cap : float
        Admissible range for the penalty rate [$/MWh].
    c_bal : float or list[float] of length 24
        System balancing cost per interval [$/MWh].
    V : float
        Consumers' marginal value of electricity [$/MWh].
    mu_D, sigma2_D : float
        Demand distribution parameters [MWh, MWh²].
    Q_wind : float
        Installed wind capacity [MWh]; upper bound on q^W_t.
        May exceed mu_W_q; the producer's bidding strategy determines
        how aggressively it commits relative to expected output.
    mu_W_q, sigma2_W_q : float
        Wind production distribution parameters [MWh, MWh²].
    prior_conv_mu_p : list of shape (3, 2)
        Uniform prior [low, high] for each conventional producer's mean bid price.
    prior_conv_sigma2_p : list of shape (3, 2)
        Uniform prior [low, high] for each conventional producer's bid variance.
    prior_solar_mu_p : list of shape (2,)
        Uniform prior [low, high] for the solar producer's mean bid price.
    prior_solar_sigma2_p : list of shape (2,)
        Uniform prior [low, high] for the solar producer's bid price variance.
    mode : str
        Evaluation mode; one of {'simulate', 'optimise_W', 'optimise_R',
        'stackelberg'}.
    kappa_exog : float
        Exogenous penalty rate used when Perspective R is passive.
    q_W_exog : list[float] of length 24, optional
        Exogenous wind bid quantities [MWh]; defaults to mu_W_q at all intervals.
    p_W_exog : list[float] of length 24, optional
        Exogenous wind bid prices [$/MWh]; defaults to p_floor at all intervals.
    n_days : int
        Monte Carlo days for final evaluation (default 1000).
    n_mc_opt : int
        Monte Carlo samples per candidate bid during optimisation (default 500).
    n_grid_q : int
        Grid resolution for wind quantity optimisation (default 10).
    n_grid_p : int
        Grid resolution for wind price optimisation (default 10).
    n_kappa : int
        Grid resolution for kappa optimisation (default 10).
    seed : int, optional
        Master random seed for full reproducibility.
    verbose : bool
        Print progress messages if True.

    Returns
    -------
    dict with keys
        'mode'                  : str
        'kappa'                 : float  -- penalty rate used / optimised
        'wind_bid_quantities'   : ndarray (24,) -- q^W_t used in evaluation
        'wind_bid_prices'       : ndarray (24,) -- p^W_t used in evaluation
        'mean_wind_revenue'     : float  -- E[Pi^W]  [$/day]
        'std_wind_revenue'      : float  -- std[Pi^W] [$/day]
        'mean_social_welfare'   : float  -- E[W]     [$/day]
        'std_social_welfare'    : float  -- std[W]   [$/day]
        'mean_consumer_surplus' : float  -- E[CS]    [$/day]
        'mean_producer_surplus' : float  -- E[PS]    [$/day]
        'mean_balancing_cost'   : float  -- E[C_bal] [$/day]
        'mean_clearing_price'   : ndarray (24,) -- E[lambda_t]
        'mean_dispatch_wind'    : ndarray (24,) -- E[d^W_t]
        'mean_shortfall'        : ndarray (24,) -- E[Delta^W_t]
        'day_outcomes'          : list[DayOutcome] -- raw simulation data
    """
    VALID_MODES = {"simulate", "optimise_W", "optimise_R", "stackelberg"}
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}; got '{mode}'.")

    # ------------------------------------------------------------------
    # 1. Build shared environment
    # ------------------------------------------------------------------
    env = Environment(
        Q_conv=Q_conv, a=a, b=b,
        p_floor=p_floor, p_cap=p_cap,
        kappa_floor=kappa_floor, kappa_cap=kappa_cap,
        c_bal=c_bal, V=V,
        mu_D=mu_D, sigma2_D=sigma2_D,
        Q_wind=Q_wind,
        mu_W_q=mu_W_q, sigma2_W_q=sigma2_W_q,
        prior_conv_mu_p=prior_conv_mu_p,
        prior_conv_sigma2_p=prior_conv_sigma2_p,
        prior_solar_mu_p=prior_solar_mu_p,
        prior_solar_sigma2_p=prior_solar_sigma2_p,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 2. Instantiate agents
    # ------------------------------------------------------------------
    conv_agents = {
        f"conv{k + 1}": ConventionalProducer(f"conv{k + 1}", env.Q_conv[k], env)
        for k in range(3)
    }
    solar_agent = SolarProducer("solar", env)
    wind_agent = WindProducer("wind", env)
    market_op = MarketOperator(env)
    regulator = SystemRegulator(env)

    competitors = {**conv_agents, "solar": solar_agent}

    # ------------------------------------------------------------------
    # 3. Resolve exogenous defaults
    # ------------------------------------------------------------------
    q_W_init = (
        np.asarray(q_W_exog, dtype=float) if q_W_exog is not None
        else np.full(24, env.Q_wind)
    )
    p_W_init = (
        np.asarray(p_W_exog, dtype=float) if p_W_exog is not None
        else np.full(24, env.p_floor)
    )

    # ------------------------------------------------------------------
    # 4. Mode-specific setup  (Step 1: penalty announcement; Step 3: bids)
    # ------------------------------------------------------------------
    if mode == "simulate":
        # Both perspectives passive: fix kappa and u_W
        regulator.set_kappa(kappa_exog)
        wind_agent.set_bids(q_W_init, p_W_init)
        if verbose:
            print(f"[simulate] kappa={kappa_exog:.4f}; using fixed bid vector.")

    elif mode == "optimise_W":
        # Perspective R passive; Perspective W active
        regulator.set_kappa(kappa_exog)
        if verbose:
            print(f"[optimise_W] kappa={kappa_exog:.4f}; "
                  f"optimising wind bids over {n_grid_q}x{n_grid_p} grid ...")
        wind_agent.optimise_bids(
            kappa=kappa_exog,
            n_mc=n_mc_opt,
            n_grid_q=n_grid_q,
            n_grid_p=n_grid_p,
            market_operator=market_op,
            competitors=competitors,
        )
        if verbose:
            q_opt, p_opt = wind_agent.get_bids()
            print(f"[optimise_W] optimal q={q_opt[0]:.2f} MWh, "
                  f"p=${p_opt[0]:.2f}/MWh (constant across intervals)")

    elif mode == "optimise_R":
        # Perspective W passive; Perspective R active
        wind_agent.set_bids(q_W_init, p_W_init)
        if verbose:
            print(f"[optimise_R] fixed wind bids; "
                  f"optimising kappa over {n_kappa} grid points ...")
        kappa_star = regulator.optimise_kappa(
            wind_producer=wind_agent,
            market_operator=market_op,
            competitors=competitors,
            n_kappa=n_kappa,
            n_mc=n_mc_opt,
            n_grid_q=n_grid_q,
            n_grid_p=n_grid_p,
        )
        # Restore exogenous bids for evaluation (regulator took control of
        # wind bids during its inner-loop; reset to caller's specification)
        wind_agent.set_bids(q_W_init, p_W_init)
        if verbose:
            print(f"[optimise_R] optimal kappa*={kappa_star:.4f} $/MWh")

    elif mode == "stackelberg":
        # Both perspectives active: bi-level Stackelberg equilibrium
        if verbose:
            print(f"[stackelberg] bi-level optimisation: "
                  f"{n_kappa} kappa points x inner wind bid search ...")
        kappa_star = regulator.optimise_kappa(
            wind_producer=wind_agent,
            market_operator=market_op,
            competitors=competitors,
            n_kappa=n_kappa,
            n_mc=n_mc_opt,
            n_grid_q=n_grid_q,
            n_grid_p=n_grid_p,
        )
        # Re-solve wind producer's problem at kappa* for final evaluation
        wind_agent.optimise_bids(
            kappa=kappa_star,
            n_mc=n_mc_opt,
            n_grid_q=n_grid_q,
            n_grid_p=n_grid_p,
            market_operator=market_op,
            competitors=competitors,
        )
        if verbose:
            q_eq, p_eq = wind_agent.get_bids()
            print(f"[stackelberg] kappa*={kappa_star:.4f} $/MWh; "
                  f"q*={q_eq[0]:.2f} MWh; p*=${p_eq[0]:.2f}/MWh")

    kappa_used = regulator.get_kappa()
    q_final, p_final = wind_agent.get_bids()

    # ------------------------------------------------------------------
    # 5. Evaluation: run n_days Monte Carlo days
    # ------------------------------------------------------------------
    if verbose:
        print(f"[eval] running {n_days} evaluation days ...")

    day_outcomes: list = []
    for _ in range(n_days):
        day_outcomes.append(
            _run_one_day(
                env=env,
                wind_producer=wind_agent,
                market_operator=market_op,
                competitors=competitors,
                regulator=regulator,
                kappa=kappa_used,
            )
        )

    # ------------------------------------------------------------------
    # 6. Aggregate statistics
    # ------------------------------------------------------------------
    wind_revenues = np.array([d.wind_net_revenue for d in day_outcomes])
    social_welfares = np.array([d.social_welfare for d in day_outcomes])
    consumer_surpluses = np.array([d.consumer_surplus for d in day_outcomes])
    producer_surpluses = np.array([d.producer_surplus for d in day_outcomes])
    balancing_costs = np.array([d.balancing_cost for d in day_outcomes])

    # Per-interval averages across all simulated days
    mean_clearing_price = np.zeros(24)
    mean_dispatch_wind = np.zeros(24)
    mean_shortfall = np.zeros(24)
    for day in day_outcomes:
        for t_idx, (mo, so) in enumerate(zip(day.market, day.settlement)):
            mean_clearing_price[t_idx] += mo.clearing_price
            mean_dispatch_wind[t_idx] += so.dispatched_quantity
            mean_shortfall[t_idx] += so.shortfall
    mean_clearing_price /= n_days
    mean_dispatch_wind /= n_days
    mean_shortfall /= n_days

    return {
        "mode": mode,
        "kappa": kappa_used,
        "wind_bid_quantities": q_final,
        "wind_bid_prices": p_final,
        "mean_wind_revenue": float(np.mean(wind_revenues)),
        "std_wind_revenue": float(np.std(wind_revenues)),
        "mean_social_welfare": float(np.mean(social_welfares)),
        "std_social_welfare": float(np.std(social_welfares)),
        "mean_consumer_surplus": float(np.mean(consumer_surpluses)),
        "mean_producer_surplus": float(np.mean(producer_surpluses)),
        "mean_balancing_cost": float(np.mean(balancing_costs)),
        "mean_clearing_price": mean_clearing_price,
        "mean_dispatch_wind": mean_dispatch_wind,
        "mean_shortfall": mean_shortfall,
        "day_outcomes": day_outcomes,
    }


# ---------------------------------------------------------------------------
# Table 4 parameter settings
# ---------------------------------------------------------------------------

# Competitor bid price distributions are given as fixed values in Table 4,
# so priors are set as point masses (low == high) at the stated values.
# mu_pi = [45, 50, 60] $/MWh; sigma_pi = [2, 2, 2] $/MWh => sigma2_pi = [4, 4, 4]
# mu_ps = 35 $/MWh;           sigma_ps = 4 $/MWh           => sigma2_ps = 16

PARAMS = dict(
    # Conventional producer capacities [MWh]: C1=300, C2=250, C3=1000
    Q_conv=[300.0, 250.0, 1000.0],

    # Solar profile: b_st = max(0, a + b*cos(2*pi*t/24)), (a, b) = (0, -400)
    a=0.0,
    b=-400.0,

    # Market price bounds [$/MWh]
    p_floor=0.0,
    p_cap=500.0,

    # Penalty rate bounds [$/MWh] — must cover all three qu values
    kappa_floor=0.0,
    kappa_cap=200.0,

    # System balancing cost and consumer value [$/MWh]
    # (affect welfare accounting only, not wind dispatch or revenue)
    c_bal=0,
    V=0,

    # Demand distribution: mu_D = 800 MWh, sigma_D = 20 MWh => sigma2_D = 400
    mu_D=800.0,
    sigma2_D=400.0,

    Q_wind=3000.0,

    # Wind production distribution: mu_G = 300 MWh, sigma_G = 50 MWh => sigma2_G = 2500
    mu_W_q=275.0,
    sigma2_W_q=2500.0,

    # Conventional producer bid price priors (point masses at Table 4 values)
    # Shape (3, 2): [[low, high], ...] — low == high gives a degenerate uniform
    prior_conv_mu_p=[[45.0, 45.0], [50.0, 50.0], [60.0, 60.0]],
    prior_conv_sigma2_p=[[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]],

    # Solar producer bid price prior (point mass at Table 4 values)
    prior_solar_mu_p=[35.0, 35.0],
    prior_solar_sigma2_p=[16.0, 16.0],

    # Fixed wind bids: (b_wt, p_wt) = (300 MWh, 50 $/MWh) for all t
    q_W_exog=[300.0] * 24,
    p_W_exog=[50.0] * 24,

    # 10 simulated days per penalty rate
    n_days=10
)

PENALTY_RATES = [20, 100, 180]  # qu values [$/MWh]

# ---------------------------------------------------------------------------
# Run simulations and collect per-interval results
# ---------------------------------------------------------------------------

rows = []  # each row: (qu, run, t, dispatched_wind_MWh, wind_revenue_USD)

for qu in PENALTY_RATES:
    print(f"Running qu = {qu} $/MWh ...")

    results = run_simulation(
        **PARAMS,
        mode="simulate",
        kappa_exog=float(qu),
    )

    for run_idx, day in enumerate(results["day_outcomes"], start=1):
        for so in day.settlement:
            rows.append({
                "qu": qu,
                "run": run_idx,
                "t": so.t,
                "dispatched_wind_MWh": round(so.dispatched_quantity, 6),
                "wind_revenue_USD": round(so.net_revenue, 6),
            })

# ---------------------------------------------------------------------------
# Save to CSV
# ---------------------------------------------------------------------------

output_path = "market2_results.csv"
fieldnames = ["qu", "run", "t", "dispatched_wind_MWh", "wind_revenue_USD"]

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} rows to '{output_path}'.")
print(f"Structure: {len(PENALTY_RATES)} penalty rates "
      f"x {PARAMS['n_days']} runs x 24 intervals = "
      f"{len(PENALTY_RATES) * PARAMS['n_days'] * 24} rows.")