"""
wind_market_abm.py
==================
Unified Agent-Based Model: Day-Ahead Wind-Power Revenue Estimation
ODD-compliant implementation.

Agents
------
  ConventionalProducer  – coal-fired, fixed quantity, stochastic price bids
  SolarProducer         – diurnal cosine profile, stochastic price bids
  WindProducer          – stochastic output, controllable bids, penalty exposure
  MarketOperator        – merit-order clearing with priority tie-breaking
  SystemRegulator       – observes system outcomes; owns κ and priority order

Interface
---------
  run_simulation(...)   – single entry point; see docstring for parameters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.random import Generator

# ---------------------------------------------------------------------------
# Section 1 – Shared data structures
# ---------------------------------------------------------------------------

INTERVALS = 24  # |T|
N_PRODUCERS = 5  # |I|

# Priority ordering indices used for tie-breaking (lower index = higher priority)
DEFAULT_PRIORITY: dict[str, int] = {
    "solar": 0,
    "wind": 1,
    "conv1": 2,
    "conv2": 3,
    "conv3": 4,
}


@dataclass
class BidSet:
    """Bids submitted by a single producer for all 24 intervals."""
    producer_id: str
    quantities: np.ndarray  # shape (24,), MWh  ≥ 0
    prices: np.ndarray  # shape (24,), $/MWh ≥ 0


@dataclass
class ClearingResult:
    """Outcome of market clearing for a single interval t."""
    interval: int
    clearing_price: float  # λ_t  $/MWh
    dispatched: dict[str, float]  # producer_id → MWh dispatched
    marginal_producer: str
    feasible: bool  # False if aggregate supply < demand
    demand: float  # D_t MWh


@dataclass
class DeliveryResult:
    """Post-delivery outcome for the wind producer in a single interval."""
    interval: int
    dispatched_qty: float  # q^disp_{W,t}
    actual_output: float  # Ã_{W,t}
    shortfall: float  # s_{W,t}
    gross_revenue: float  # q^disp_{W,t} · λ_t
    penalty: float  # κ · s_{W,t}
    net_revenue: float  # gross – penalty


@dataclass
class SimulationResult:
    """Complete output of one simulation run."""
    # Per-interval
    clearing: list[ClearingResult]
    delivery: list[DeliveryResult]

    # Wind-producer aggregates
    total_net_revenue: float  # R^net_W  $

    # Regulator aggregates
    mean_clearing_price: float  # λ̄  $/MWh
    mean_renewable_share: float  # φ̄  ∈ [0,1]
    total_wind_shortfall: float  # S^total_W  MWh
    adequacy_count: int  # Σ_t δ_t  (max 24)

    # Interval-level regulator series
    clearing_prices: np.ndarray  # shape (24,)
    renewable_shares: np.ndarray  # shape (24,)
    adequacy_flags: np.ndarray  # shape (24,), bool


# ---------------------------------------------------------------------------
# Section 2 – Producer agents
# ---------------------------------------------------------------------------

class ConventionalProducer:
    """
    Conventional (coal-fired) producer agent.

    Bid quantity is fixed at installed capacity Q̄_i.
    Bid price is drawn i.i.d. from N(μ^p_i, (σ^p_i)²), truncated below at 0.

    Parameters
    ----------
    producer_id  : unique label, e.g. 'conv1'
    capacity     : Q̄_i  installed capacity (MW = MWh per interval)
    mu_price     : μ^p_i  mean bid price  ($/MWh)
    sigma_price  : σ^p_i  std dev of bid price  ($/MWh)
    """

    def __init__(
            self,
            producer_id: str,
            capacity: float,
            mu_price: float,
            sigma_price: float,
    ) -> None:
        self.producer_id = producer_id
        self.capacity = capacity
        self.mu_price = mu_price
        self.sigma_price = sigma_price

    # State variables ---------------------------------------------------------

    def reset(self) -> None:
        """Clear per-run state (bids set during bid_formation)."""
        self._bids: Optional[BidSet] = None

    # Processes ---------------------------------------------------------------

    def submit_bids(self, rng: Generator) -> BidSet:
        """
        Phase 1 – Bid formation.
        Draw 24 independent price realisations; fix quantity at capacity.
        """
        quantities = np.full(INTERVALS, self.capacity)
        prices = np.maximum(
            0.0,
            rng.normal(self.mu_price, self.sigma_price, size=INTERVALS)
        )
        self._bids = BidSet(self.producer_id, quantities, prices)
        return self._bids


class SolarProducer:
    """
    Solar producer agent.

    Bid quantity follows the diurnal profile:
        q^bid_{S,t} = max(0,  a + b·cos(2πt/24)),   b < 0
    Bid price is drawn i.i.d. from N(μ^p_S, (σ^p_S)²), truncated below at 0.

    Parameters
    ----------
    a            : baseline offset (MWh);  a > 0
    b            : amplitude parameter (MWh);  b < 0,  |b| ≥ a  for zero at midnight
    mu_price     : μ^p_S  mean bid price  ($/MWh)
    sigma_price  : σ^p_S  std dev of bid price  ($/MWh)
    """

    def __init__(
            self,
            a: float,
            b: float,
            mu_price: float,
            sigma_price: float,
    ) -> None:
        if b >= 0:
            raise ValueError(
                f"Solar amplitude b must be negative (got {b}). "
                "A negative b ensures the profile peaks at noon (t=12) "
                "and reaches its minimum at midnight (t=0/24)."
            )
        self.producer_id = "solar"
        self.a = a
        self.b = b
        self.mu_price = mu_price
        self.sigma_price = sigma_price

        # Pre-compute the deterministic quantity profile once.
        # t runs from 1 to 24 inclusive, consistent with T = {1,...,24}.
        t = np.arange(1, INTERVALS + 1)
        self._quantity_profile = np.maximum(
            0.0,
            self.a + self.b * np.cos(2 * np.pi * t / INTERVALS)
        )

    # Properties --------------------------------------------------------------

    @property
    def quantity_profile(self) -> np.ndarray:
        """Deterministic 24-hour bid quantity profile (MWh)."""
        return self._quantity_profile.copy()

    # Processes ---------------------------------------------------------------

    def reset(self) -> None:
        self._bids: Optional[BidSet] = None

    def submit_bids(self, rng: Generator) -> BidSet:
        """
        Phase 1 – Bid formation.
        Quantity from deterministic cosine profile; price drawn stochastically.
        """
        prices = np.maximum(
            0.0,
            rng.normal(self.mu_price, self.sigma_price, size=INTERVALS)
        )
        self._bids = BidSet(self.producer_id, self._quantity_profile.copy(), prices)
        return self._bids


class WindProducer:
    """
    Wind producer agent (Perspective Agent 1).

    Control variables  :  q^bid_{W,t}  ∈ [0, Q̄_W]
                          p^bid_{W,t}  ∈ [0, ∞)
    Actual output      :  Ã_{W,t} ~ N(μ_W, σ²_W), drawn after clearing
    Delivery shortfall :  s_{W,t}  = max(0, q^disp_{W,t} − Ã_{W,t})
    Net revenue        :  r^net_{W,t} = q^disp_{W,t}·λ_t − κ·s_{W,t}

    Bidding strategies
    ------------------
    The bid quantities and prices are set via the `bid_quantities` and
    `bid_prices` constructor arguments (arrays of length 24).  These serve
    as the control variables and can be supplied:
      • directly (analytical optimum, heuristic, or scenario value), or
      • as the result of an external optimisation routine that calls
        evaluate_expected_revenue() over candidate bid vectors.

    Parameters
    ----------
    capacity      : Q̄_W  nameplate capacity (MW)
    mu_output     : μ_W   mean wind production per interval (MWh)
    sigma_output  : σ_W   std dev of wind production (MWh)
    bid_quantities: q^bid_{W,t} for t=0..23  (MWh); default = μ_W for all t
    bid_prices    : p^bid_{W,t} for t=0..23  ($/MWh); default = 0 for all t
    penalty_rate  : κ  ($/MWh); received from regulator as exogenous input
    """

    def __init__(
            self,
            capacity: float,
            mu_output: float,
            sigma_output: float,
            penalty_rate: float,
            bid_quantities: Optional[np.ndarray] = None,
            bid_prices: Optional[np.ndarray] = None,
    ) -> None:
        self.producer_id = "wind"
        self.capacity = capacity
        self.mu_output = mu_output
        self.sigma_output = sigma_output
        self.penalty_rate = penalty_rate  # κ – exogenous from regulator

        # Control variables: default strategy is bid mean output at price 0
        self.bid_quantities = (
            np.clip(bid_quantities, 0.0, capacity)
            if bid_quantities is not None
            else np.full(INTERVALS, min(mu_output, capacity))
        )
        self.bid_prices = (
            np.maximum(0.0, bid_prices)
            if bid_prices is not None
            else np.zeros(INTERVALS)
        )

        if len(self.bid_quantities) != INTERVALS or len(self.bid_prices) != INTERVALS:
            raise ValueError("bid_quantities and bid_prices must each have length 24.")

    # Processes ---------------------------------------------------------------

    def reset(self) -> None:
        """Clear per-run delivery state."""
        self._delivery_results: list[DeliveryResult] = []
        self._total_net_revenue: float = 0.0

    def submit_bids(self, rng: Generator) -> BidSet:  # noqa: ARG002
        """
        Phase 1 – Bid formation.
        Returns the pre-specified control variables as the bid set.
        rng is accepted for interface uniformity but not used here.
        """
        return BidSet(
            self.producer_id,
            self.bid_quantities.copy(),
            self.bid_prices.copy(),
        )

    def record_delivery(
            self,
            interval: int,
            dispatched_qty: float,
            clearing_price: float,
            rng: Generator,
    ) -> DeliveryResult:
        """
        Phase 3 – Physical delivery and penalty assessment.

        Draws Ã_{W,t}, computes shortfall and net revenue for interval t.
        """
        actual = float(np.maximum(
            0.0,
            rng.normal(self.mu_output, self.sigma_output)
        ))
        shortfall = max(0.0, dispatched_qty - actual)
        gross_revenue = dispatched_qty * clearing_price
        penalty = self.penalty_rate * shortfall
        net_revenue = gross_revenue - penalty

        result = DeliveryResult(
            interval=interval,
            dispatched_qty=dispatched_qty,
            actual_output=actual,
            shortfall=shortfall,
            gross_revenue=gross_revenue,
            penalty=penalty,
            net_revenue=net_revenue,
        )
        self._delivery_results.append(result)
        self._total_net_revenue += net_revenue
        return result

    # Objective ---------------------------------------------------------------

    def evaluate_expected_revenue(
            self,
            bid_quantities: np.ndarray,
            bid_prices: np.ndarray,
            n_samples: int,
            market_operator: "MarketOperator",
            solar_producer: "SolarProducer",
            conventional_producers: list["ConventionalProducer"],
            rng: Generator,
    ) -> float:
        """
        Estimate E[R^net_W] for a candidate bid vector via Monte Carlo.

        This is the inner loop of the wind producer's optimisation problem.
        It temporarily overrides the bid controls, runs n_samples clearing
        and delivery draws, and returns the sample mean net revenue.

        Parameters
        ----------
        bid_quantities : candidate q^bid_{W,t}  shape (24,)
        bid_prices     : candidate p^bid_{W,t}  shape (24,)
        n_samples      : number of Monte Carlo draws
        market_operator, solar_producer, conventional_producers :
                         shared environment agents (not mutated)
        rng            : random number generator

        Returns
        -------
        float : sample estimate of E[R^net_W]
        """
        revenues = np.zeros(n_samples)

        for s in range(n_samples):
            total = 0.0
            wind_bids = BidSet("wind", bid_quantities, bid_prices)

            # Collect all bids for this sample
            all_bids = {
                p.producer_id: p.submit_bids(rng)
                for p in conventional_producers
            }
            all_bids["solar"] = solar_producer.submit_bids(rng)
            all_bids["wind"] = wind_bids

            for t in range(INTERVALS):
                # Clear interval t
                cr = market_operator.clear_interval(t, all_bids, rng)
                if not cr.feasible:
                    continue
                disp_qty = cr.dispatched.get("wind", 0.0)

                # Draw actual wind output
                actual = float(np.maximum(
                    0.0, rng.normal(self.mu_output, self.sigma_output)
                ))
                shortfall = max(0.0, disp_qty - actual)
                total += disp_qty * cr.clearing_price - self.penalty_rate * shortfall

            revenues[s] = total

        return float(revenues.mean())


# ---------------------------------------------------------------------------
# Section 3 – Market operator agent
# ---------------------------------------------------------------------------

class MarketOperator:
    """
    Market operator agent.

    Clears each interval via the merit-order mechanism with priority-based
    tie-breaking.  Non-strategic: executes the clearing rule deterministically
    given submitted bids and realised demand.

    Parameters
    ----------
    mu_demand    : μ_D   mean hourly demand (MWh)
    sigma_demand : σ_D   std dev of hourly demand (MWh)
    priority     : dict mapping producer_id → priority rank (int);
                   lower rank = higher dispatch priority for tied prices.
                   Default: solar(0) > wind(1) > conv1(2) > conv2(3) > conv3(4)
    """

    def __init__(
            self,
            mu_demand: float,
            sigma_demand: float,
            priority: dict[str, int] = DEFAULT_PRIORITY,
    ) -> None:
        self.mu_demand = mu_demand
        self.sigma_demand = sigma_demand
        self.priority = priority

    def _sort_key(self, producer_id: str, price: float) -> tuple[float, int]:
        """Sort key: (bid price, priority rank). Lower values cleared first."""
        return (price, self.priority.get(producer_id, 99))

    def clear_interval(
            self,
            interval: int,
            bids: dict[str, BidSet],
            rng: Generator,
    ) -> ClearingResult:
        """
        Phase 2 – Merit-order clearing for a single interval.

        Steps
        -----
        1. Draw demand D_t.
        2. Sort producers by (price, priority).
        3. Traverse the stack until cumulative supply ≥ D_t.
        4. Set λ_t = marginal producer's price.
        5. Dispatch: full for infra-marginal, partial for marginal, zero otherwise.

        Returns ClearingResult with feasible=False if supply is insufficient.
        """
        t = interval

        # Draw demand (truncated below at 0)
        demand = float(np.maximum(
            0.0,
            rng.normal(self.mu_demand, self.sigma_demand)
        ))

        # Build (producer_id, quantity, price) list for interval t
        stack = [
            (pid, float(b.quantities[t]), float(b.prices[t]))
            for pid, b in bids.items()
        ]
        # Sort by (price asc, priority asc)
        stack.sort(key=lambda x: self._sort_key(x[0], x[2]))

        # Traverse merit-order stack
        cumulative = 0.0
        dispatched = {pid: 0.0 for pid in bids}
        clearing_price = 0.0
        marginal = ""
        feasible = False

        for pid, qty, price in stack:
            if cumulative >= demand:
                break
            residual = demand - cumulative
            if qty <= residual:
                # Infra-marginal: fully dispatched
                dispatched[pid] = qty
                cumulative += qty
            else:
                # Marginal: partially dispatched to exactly meet demand
                dispatched[pid] = residual
                cumulative += residual
            clearing_price = price
            marginal = pid
            if cumulative >= demand:
                feasible = True
                break

        if not feasible:
            warnings.warn(
                f"Interval {t}: aggregate supply ({cumulative:.1f} MWh) "
                f"< demand ({demand:.1f} MWh). Shortfall event recorded.",
                stacklevel=2,
            )

        return ClearingResult(
            interval=t,
            clearing_price=clearing_price,
            dispatched=dispatched,
            marginal_producer=marginal,
            feasible=feasible,
            demand=demand,
        )


# ---------------------------------------------------------------------------
# Section 4 – System regulator agent
# ---------------------------------------------------------------------------

class SystemRegulator:
    """
    System regulator agent (Perspective Agent 2).

    Observes post-clearing and post-delivery outcomes across all agents and
    intervals.  Owns the penalty rate κ and the dispatch priority ordering
    Π_priority.  In regulator mode these are the control variables; in
    wind-producer mode they are fixed exogenous parameters.

    Parameters
    ----------
    penalty_rate        : κ  ($/MWh)
    priority            : Π_priority — dict of producer_id → rank
    renewable_target    : φ^min  minimum acceptable mean renewable share
    shortfall_tolerance : S̄  maximum acceptable expected total shortfall (MWh)
    """

    def __init__(
            self,
            penalty_rate: float,
            priority: dict[str, int] = DEFAULT_PRIORITY,
            renewable_target: float = 0.0,
            shortfall_tolerance: float = np.inf,
    ) -> None:
        self.penalty_rate = penalty_rate  # κ  – control variable
        self.priority = priority  # Π  – control variable
        self.renewable_target = renewable_target  # φ^min
        self.shortfall_tolerance = shortfall_tolerance  # S̄

    # Processes ---------------------------------------------------------------

    def observe(
            self,
            clearing_results: list[ClearingResult],
            delivery_results: list[DeliveryResult],
    ) -> dict:
        """
        Phase 4 – Regulator observation submodel.

        Computes all system-level metrics from a completed simulation run.

        Returns
        -------
        dict with keys:
            clearing_prices    : np.ndarray shape (24,)
            renewable_shares   : np.ndarray shape (24,)
            adequacy_flags     : np.ndarray shape (24,), bool
            shortfalls         : np.ndarray shape (24,)
            mean_price         : float   λ̄
            mean_renewable     : float   φ̄
            total_shortfall    : float   S^total_W
            adequacy_count     : int     Σ δ_t
            price_range        : float   max(λ) – min(λ)
            meets_renewable_target   : bool
            meets_shortfall_tolerance: bool
        """
        n = len(clearing_results)

        clearing_prices = np.array([cr.clearing_price for cr in clearing_results])
        demands = np.array([cr.demand for cr in clearing_results])
        adequacy_flags = np.array([cr.feasible for cr in clearing_results])
        solar_dispatch = np.array([cr.dispatched.get("solar", 0.0) for cr in clearing_results])
        wind_dispatch = np.array([cr.dispatched.get("wind", 0.0) for cr in clearing_results])
        shortfalls = np.array([dr.shortfall for dr in delivery_results])

        # Renewable share φ_t = (q^disp_S,t + q^disp_W,t) / D_t
        with np.errstate(invalid="ignore", divide="ignore"):
            renewable_shares = np.where(
                demands > 0,
                (solar_dispatch + wind_dispatch) / demands,
                0.0
            )

        mean_price = float(clearing_prices.mean())
        mean_renewable = float(renewable_shares.mean())
        total_shortfall = float(shortfalls.sum())
        adequacy_count = int(adequacy_flags.sum())

        return {
            "clearing_prices": clearing_prices,
            "renewable_shares": renewable_shares,
            "adequacy_flags": adequacy_flags,
            "shortfalls": shortfalls,
            "mean_price": mean_price,
            "mean_renewable": mean_renewable,
            "total_shortfall": total_shortfall,
            "adequacy_count": adequacy_count,
            "price_range": float(clearing_prices.max() - clearing_prices.min()),
            "meets_renewable_target": mean_renewable >= self.renewable_target,
            "meets_shortfall_tolerance": total_shortfall <= self.shortfall_tolerance,
        }

    def sweep_penalty_rate(
            self,
            kappa_values: np.ndarray,
            wind_producer: WindProducer,
            market_operator: MarketOperator,
            solar_producer: SolarProducer,
            conv_producers: list[ConventionalProducer],
            n_runs: int,
            rng: Generator,
    ) -> list[dict]:
        """
        Regulator mode – penalty calibration submodel (Section 7.8).

        For each candidate κ in kappa_values:
          1. Update the wind producer's penalty rate.
          2. Run n_runs simulation draws.
          3. Record mean shortfall, mean renewable share, mean net revenue.

        Returns a list of result dicts, one per κ value, enabling the
        regulator to identify κ* that satisfies its objectives.
        """
        results = []
        for kappa in kappa_values:
            # Update penalty rate for this sweep step
            wind_producer.penalty_rate = kappa
            self.penalty_rate = kappa

            run_shortfalls = []
            run_renewables = []
            run_revenues = []

            for _ in range(n_runs):
                res = _single_run(
                    wind_producer, solar_producer, conv_producers,
                    market_operator, self, rng
                )
                run_shortfalls.append(res.total_wind_shortfall)
                run_renewables.append(res.mean_renewable_share)
                run_revenues.append(res.total_net_revenue)

            results.append({
                "kappa": float(kappa),
                "mean_shortfall": float(np.mean(run_shortfalls)),
                "mean_renewable_share": float(np.mean(run_renewables)),
                "mean_net_revenue": float(np.mean(run_revenues)),
                "meets_renewable_target": float(np.mean(run_renewables)) >= self.renewable_target,
                "meets_shortfall_tol": float(np.mean(run_shortfalls)) <= self.shortfall_tolerance,
            })

        # Restore original penalty rate
        wind_producer.penalty_rate = self.penalty_rate
        return results


# ---------------------------------------------------------------------------
# Section 5 – Single-run execution kernel (internal)
# ---------------------------------------------------------------------------

def _single_run(
        wind_producer: WindProducer,
        solar_producer: SolarProducer,
        conv_producers: list[ConventionalProducer],
        market_operator: MarketOperator,
        regulator: SystemRegulator,
        rng: Generator,
) -> SimulationResult:
    """
    Execute one complete simulation run (Phases 0–4).

    This internal function is called by run_simulation() for each replicate.
    It is also used as the evaluation kernel in
    WindProducer.evaluate_expected_revenue() and
    SystemRegulator.sweep_penalty_rate().
    """

    # Phase 0 – Reset per-run state
    wind_producer.reset()
    solar_producer.reset()
    for cp in conv_producers:
        cp.reset()

    # Phase 1 – Bid formation
    all_bids: dict[str, BidSet] = {}
    for cp in conv_producers:
        all_bids[cp.producer_id] = cp.submit_bids(rng)
    all_bids["solar"] = solar_producer.submit_bids(rng)
    all_bids["wind"] = wind_producer.submit_bids(rng)

    # Update market operator priority from regulator
    market_operator.priority = regulator.priority

    # Phases 2 & 3 – Market clearing and physical delivery, interval by interval
    clearing_results: list[ClearingResult] = []
    delivery_results: list[DeliveryResult] = []

    for t in range(INTERVALS):
        # Phase 2 – Market clearing
        cr = market_operator.clear_interval(t, all_bids, rng)
        clearing_results.append(cr)

        # Phase 3 – Physical delivery (wind only; other producers are firm)
        wind_dispatched = cr.dispatched.get("wind", 0.0)
        dr = wind_producer.record_delivery(
            interval=t,
            dispatched_qty=wind_dispatched,
            clearing_price=cr.clearing_price if cr.feasible else 0.0,
            rng=rng,
        )
        delivery_results.append(dr)

    # Phase 4 – Aggregation
    total_net_revenue = wind_producer._total_net_revenue

    reg_obs = regulator.observe(clearing_results, delivery_results)

    return SimulationResult(
        clearing=clearing_results,
        delivery=delivery_results,
        total_net_revenue=total_net_revenue,
        mean_clearing_price=reg_obs["mean_price"],
        mean_renewable_share=reg_obs["mean_renewable"],
        total_wind_shortfall=reg_obs["total_shortfall"],
        adequacy_count=reg_obs["adequacy_count"],
        clearing_prices=reg_obs["clearing_prices"],
        renewable_shares=reg_obs["renewable_shares"],
        adequacy_flags=reg_obs["adequacy_flags"],
    )


# ---------------------------------------------------------------------------
# Section 6 – Interface function
# ---------------------------------------------------------------------------

def run_simulation(
        # --- Conventional producers ---
        conv_capacities: list[float],
        conv_mu_prices: list[float],
        conv_sigma_prices: list[float],
        # --- Solar producer ---
        solar_a: float,
        solar_b: float,
        solar_mu_price: float,
        solar_sigma_price: float,
        # --- Wind producer ---
        wind_capacity: float,
        wind_mu_output: float,
        wind_sigma_output: float,
        wind_bid_quantities: Optional[np.ndarray] = None,
        wind_bid_prices: Optional[np.ndarray] = None,
        # --- Market ---
        mu_demand: float = 300.0,
        sigma_demand: float = 30.0,
        # --- Regulator ---
        penalty_rate: float = 50.0,
        priority: dict[str, int] = DEFAULT_PRIORITY,
        renewable_target: float = 0.0,
        shortfall_tolerance: float = np.inf,
        # --- Simulation control ---
        n_runs: int = 1000,
        seed: Optional[int] = None,
        # --- Evaluation mode ---
        mode: str = "wind_producer",
        kappa_sweep: Optional[np.ndarray] = None,
) -> dict:
    """
    Run the unified day-ahead wind-power market ABM.

    Parameters
    ----------
    conv_capacities     : list of 3 installed capacities Q̄_i  (MW)
    conv_mu_prices      : list of 3 mean bid prices μ^p_i  ($/MWh)
    conv_sigma_prices   : list of 3 bid price std devs σ^p_i  ($/MWh)
    solar_a             : solar profile baseline offset a  (MWh);  a > 0
    solar_b             : solar profile amplitude b  (MWh);  b < 0
    solar_mu_price      : solar mean bid price μ^p_S  ($/MWh)
    solar_sigma_price   : solar bid price std dev σ^p_S  ($/MWh)
    wind_capacity       : wind nameplate capacity Q̄_W  (MW)
    wind_mu_output      : mean wind production μ_W  (MWh)
    wind_sigma_output   : std dev of wind production σ_W  (MWh)
    wind_bid_quantities : control variable q^bid_{W,t}, shape (24,);
                          defaults to μ_W for all t if None
    wind_bid_prices     : control variable p^bid_{W,t}, shape (24,);
                          defaults to 0 for all t if None
    mu_demand           : mean hourly demand μ_D  (MWh)
    sigma_demand        : std dev of hourly demand σ_D  (MWh)
    penalty_rate        : under-delivery penalty rate κ  ($/MWh)
    priority            : dispatch priority dict; default solar>wind>conv1-3
    renewable_target    : regulator minimum renewable share φ^min  ∈ [0,1]
    shortfall_tolerance : regulator maximum tolerated mean shortfall S̄  (MWh)
    n_runs              : number of Monte Carlo simulation runs
    seed                : random seed for reproducibility
    mode                : evaluation mode:
                            'wind_producer' – wind bids are control variables;
                                             κ and priority are exogenous
                            'regulator'     – κ swept over kappa_sweep;
                                             wind bids are fixed strategy
                            'composed'      – both perspectives active;
                                             returns combined output
    kappa_sweep         : array of κ values for regulator mode

    Returns
    -------
    dict with keys depending on mode:

    All modes
    ---------
      'runs'                 : list of SimulationResult (length n_runs)
      'mean_net_revenue'     : float   E[R^net_W]
      'std_net_revenue'      : float   std(R^net_W)
      'quantiles'            : dict {0.05, 0.25, 0.50, 0.75, 0.95} of R^net_W
      'mean_clearing_price'  : float   E[λ̄]
      'mean_renewable_share' : float   E[φ̄]
      'mean_shortfall'       : float   E[S^total_W]
      'adequacy_rate'        : float   fraction of intervals cleared (∈ [0,1])

    Regulator / composed mode additionally
    ---------------------------------------
      'kappa_sweep_results'  : list of dicts from sweep_penalty_rate()
    """

    # Validate mode
    if mode not in ("wind_producer", "regulator", "composed"):
        raise ValueError(f"mode must be 'wind_producer', 'regulator', or 'composed'. Got '{mode}'.")

    if mode in ("regulator", "composed") and kappa_sweep is None:
        kappa_sweep = np.linspace(0.0, penalty_rate * 4, 20)

    # Initialise random number generator
    rng = np.random.default_rng(seed)

    # Instantiate agents
    if len(conv_capacities) != 3 or len(conv_mu_prices) != 3 or len(conv_sigma_prices) != 3:
        raise ValueError("Exactly 3 conventional producers must be specified.")

    conv_producers = [
        ConventionalProducer(
            producer_id=f"conv{i + 1}",
            capacity=conv_capacities[i],
            mu_price=conv_mu_prices[i],
            sigma_price=conv_sigma_prices[i],
        )
        for i in range(3)
    ]

    solar_producer = SolarProducer(
        a=solar_a,
        b=solar_b,
        mu_price=solar_mu_price,
        sigma_price=solar_sigma_price,
    )

    wind_producer = WindProducer(
        capacity=wind_capacity,
        mu_output=wind_mu_output,
        sigma_output=wind_sigma_output,
        penalty_rate=penalty_rate,
        bid_quantities=wind_bid_quantities,
        bid_prices=wind_bid_prices,
    )

    market_operator = MarketOperator(
        mu_demand=mu_demand,
        sigma_demand=sigma_demand,
        priority=priority,
    )

    regulator = SystemRegulator(
        penalty_rate=penalty_rate,
        priority=priority,
        renewable_target=renewable_target,
        shortfall_tolerance=shortfall_tolerance,
    )

    # Execute n_runs simulation replicates
    runs: list[SimulationResult] = [
        _single_run(
            wind_producer, solar_producer, conv_producers,
            market_operator, regulator, rng
        )
        for _ in range(n_runs)
    ]

    # Aggregate across runs
    revenues = np.array([r.total_net_revenue for r in runs])
    mean_prices = np.array([r.mean_clearing_price for r in runs])
    mean_renewables = np.array([r.mean_renewable_share for r in runs])
    shortfalls = np.array([r.total_wind_shortfall for r in runs])
    adequacy_counts = np.array([r.adequacy_count for r in runs])

    output = {
        "runs": runs,
        "mean_net_revenue": float(revenues.mean()),
        "std_net_revenue": float(revenues.std()),
        "quantiles": {
            q: float(np.quantile(revenues, q))
            for q in (0.05, 0.25, 0.50, 0.75, 0.95)
        },
        "mean_clearing_price": float(mean_prices.mean()),
        "mean_renewable_share": float(mean_renewables.mean()),
        "mean_shortfall": float(shortfalls.mean()),
        "adequacy_rate": float(adequacy_counts.mean()) / INTERVALS,
    }

    # Regulator / composed mode: penalty sweep
    if mode in ("regulator", "composed"):
        sweep = regulator.sweep_penalty_rate(
            kappa_values=kappa_sweep,
            wind_producer=wind_producer,
            market_operator=market_operator,
            solar_producer=solar_producer,
            conv_producers=conv_producers,
            n_runs=n_runs,
            rng=rng,
        )
        output["kappa_sweep_results"] = sweep

    return output


# ---------------------------------------------------------------------------
# Section 7 – Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv

    T = 24
    KAPPA_VALUES = [20, 100, 180]
    N_RUNS = 10

    BASE_PARAMS = dict(
        conv_capacities=[300.0, 250.0, 1000.0],
        conv_mu_prices=[45.0, 50.0, 60.0],
        conv_sigma_prices=[2.0, 2.0, 2.0],
        solar_a=0.0,
        solar_b=-400.0,
        solar_mu_price=35.0,
        solar_sigma_price=4.0,
        wind_capacity=3000.0,
        wind_mu_output=275.0,
        wind_sigma_output=50.0,
        wind_bid_quantities=np.full(T, 300.0),
        wind_bid_prices=np.full(T, 50.0),
        mu_demand=800.0,
        sigma_demand=20.0,
        n_runs=N_RUNS,
        mode="wind_producer",
    )

    # ------------------------------------------------------------------
    # Run simulation and collect per-interval results
    # ------------------------------------------------------------------
    # Rows: one per (kappa, run, interval)
    rows = []

    for kappa in KAPPA_VALUES:
        results = run_simulation(
            **BASE_PARAMS,
            penalty_rate=float(kappa),
            seed=None,
        )

        for run_idx, sim in enumerate(results["runs"]):
            for t in range(T):
                cr = sim.clearing[t]
                dr = sim.delivery[t]
                rows.append({
                    "kappa": kappa,
                    "run": run_idx + 1,
                    "interval": t + 1,
                    "wind_dispatched": cr.dispatched.get("wind", 0.0),
                    "wind_revenue": dr.net_revenue,
                })

    # ------------------------------------------------------------------
    # Save to CSV
    # ------------------------------------------------------------------
    output_path = "market4_results.csv"
    fieldnames = ["kappa", "run", "interval", "wind_dispatched", "wind_revenue"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to '{output_path}'.")
    print(f"  ({len(KAPPA_VALUES)} kappa values × {N_RUNS} runs × {T} intervals)")
