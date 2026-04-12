"""
Agent-Based Model: Wind-Power Producer in a Day-Ahead Spot Market
=================================================================
Unified implementation following the ODD protocol specification.

Agents
------
    WindProducer          – strategic bidding agent (Perspective 1)
    ConventionalProducer  – exogenous coal producer (x3)
    SolarProducer         – exogenous renewable producer
    MarketOperator        – deterministic clearing and settlement agent
    SystemRegulator       – penalty-design agent (Perspective 2)

Interface
---------
    run_simulation(...)   – single entry point for all use cases
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import default_rng
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class Bid:
    """A single (quantity, price) bid submitted by a producer for one interval."""
    quantity: float  # MWh,   >= 0
    price: float  # $/MWh, >= 0


@dataclass
class ClearingResult:
    """Outputs produced by the MarketOperator for one interval."""
    clearing_price: float  # lambda_t
    dispatched: dict  # producer_id -> MWh dispatched
    marginal_producer: str  # id of the marginal unit


@dataclass
class SettlementResult:
    """Outputs produced by the MarketOperator settlement step for one interval."""
    revenue: float  # lambda_t * q_disp^W
    penalty: float  # rho * e^W_t
    net_revenue: float  # revenue - penalty
    shortfall: float  # e^W_t = max(0, q_disp^W - q_actual^W)


# ---------------------------------------------------------------------------
# Agent 1 – Wind Producer
# ---------------------------------------------------------------------------

class WindProducer:
    """
    Strategic optimising agent (Perspective 1).

    The producer commits a bid schedule {(q_t^W, p_t^W)} for all 24 intervals
    once, before t=1, under uncertainty about its own output and competitor
    behaviour.  Given a penalty rate rho (exogenous from Perspective 2), it
    maximises expected daily net revenue via Monte Carlo gradient-free
    optimisation over bid quantities.  Bid prices are fixed at zero (price-
    taking, near-zero marginal cost).

    Parameters
    ----------
    capacity : float
        Installed wind capacity Q_bar^W (MWh).
    mu_wind : float
        Mean actual wind-power production mu^W (MWh).
    sigma_wind : float
        Standard deviation of actual wind-power production sigma^W (MWh).
    rho : float
        Penalty rate rho ($/MWh); treated as exogenous in Perspective 1.
    seed : int, optional
        Random seed for reproducibility.
    """

    INTERVALS = 24
    id = "wind"

    def __init__(
            self,
            capacity: float,
            mu_wind: float,
            sigma_wind: float,
            rho: float,
            seed: Optional[int] = None,
    ) -> None:
        self.capacity = capacity
        self.mu_wind = mu_wind
        self.sigma_wind = sigma_wind
        self.rho = rho
        self._rng = default_rng(seed)

        # Optimised bid schedule (set by optimise_bids)
        self.bid_quantities: np.ndarray = np.full(self.INTERVALS, mu_wind)
        self.bid_prices: np.ndarray = np.zeros(self.INTERVALS)

        # Interval-level state (updated during simulation)
        self.realised_output: np.ndarray = np.zeros(self.INTERVALS)
        self.dispatched_quantity: np.ndarray = np.zeros(self.INTERVALS)
        self.shortfall: np.ndarray = np.zeros(self.INTERVALS)
        self.interval_net_revenue: np.ndarray = np.zeros(self.INTERVALS)

    # ------------------------------------------------------------------
    # Submodel 1 – Bid optimisation
    # ------------------------------------------------------------------

    def optimise_bids(
            self,
            market_operator: "MarketOperator",
            n_mc: int = 2000,
    ) -> np.ndarray:
        """
        Solve the wind producer's optimisation problem (Submodel 1).

        The objective is separable across intervals, so each interval's bid
        quantity is optimised independently via bounded scalar minimisation
        over E[r_t^W(q_t^W)] estimated by Monte Carlo.

        Parameters
        ----------
        market_operator : MarketOperator
            Provides clearing simulation and competitor bid sampling.
        n_mc : int
            Monte Carlo sample size for expectation estimation.

        Returns
        -------
        np.ndarray
            Optimised bid quantities for all 24 intervals.
        """
        rng = default_rng(self._rng.integers(1 << 31))

        for t in range(self.INTERVALS):
            # Pre-draw all stochastic inputs for this interval
            q_actual_samples = np.clip(
                rng.normal(self.mu_wind, self.sigma_wind, n_mc),
                0.0, self.capacity,
            )
            D_samples = np.maximum(
                rng.normal(market_operator.mu_demand, market_operator.sigma_demand, n_mc),
                0.0,
            )
            comp_bids = market_operator.sample_competitor_bids(t, n_mc, rng)

            def neg_expected_net_revenue(q_bid: float) -> float:
                total_net = 0.0
                for s in range(n_mc):
                    my_bid = Bid(quantity=float(q_bid), price=0.0)
                    result = market_operator.clear_interval(
                        t=t,
                        bids={self.id: my_bid, **comp_bids[s]},
                        demand=float(D_samples[s]),
                    )
                    q_disp = result.dispatched.get(self.id, 0.0)
                    shortfall = max(0.0, q_disp - float(q_actual_samples[s]))
                    net = result.clearing_price * q_disp - self.rho * shortfall
                    total_net += net
                return -(total_net / n_mc)

            opt = minimize_scalar(
                neg_expected_net_revenue,
                bounds=(0.0, self.capacity),
                method="bounded",
            )
            self.bid_quantities[t] = float(opt.x)

        return self.bid_quantities

    # ------------------------------------------------------------------
    # Accessors used during the simulation loop
    # ------------------------------------------------------------------

    def get_bid(self, t: int) -> Bid:
        """Return the committed bid for interval t."""
        return Bid(
            quantity=float(self.bid_quantities[t]),
            price=float(self.bid_prices[t]),
        )

    def realise_output(self, t: int) -> float:
        """
        Draw actual wind production for interval t (Step 3.1).
        Result is stored and truncated to [0, capacity].
        """
        q = float(np.clip(
            self._rng.normal(self.mu_wind, self.sigma_wind),
            0.0, self.capacity,
        ))
        self.realised_output[t] = q
        return q

    def record_dispatch(self, t: int, q_disp: float) -> None:
        """Store the dispatched quantity assigned by the market operator."""
        self.dispatched_quantity[t] = q_disp

    def record_settlement(self, t: int, result: SettlementResult) -> None:
        """Store settlement outcomes for interval t."""
        self.shortfall[t] = result.shortfall
        self.interval_net_revenue[t] = result.net_revenue

    @property
    def accumulated_net_revenue(self) -> float:
        """R_d^W: total daily net revenue across all settled intervals."""
        return float(np.sum(self.interval_net_revenue))

    def reset(self) -> None:
        """Reset all interval-level state for a fresh simulation run."""
        self.realised_output[:] = 0.0
        self.dispatched_quantity[:] = 0.0
        self.shortfall[:] = 0.0
        self.interval_net_revenue[:] = 0.0


# ---------------------------------------------------------------------------
# Agents 2–4 – Conventional Producers
# ---------------------------------------------------------------------------

class ConventionalProducer:
    """
    Exogenous coal producer (Agents 2, 3, and 4).

    Bid quantity is fixed at full nameplate capacity in every interval.
    Bid price is drawn i.i.d. from N(mu_price, sigma_price^2), truncated to >= 0.

    Parameters
    ----------
    producer_id : str
        Unique identifier, e.g. 'conv1'.
    capacity : float
        Nameplate capacity K_bar^i (MWh).
    mu_price : float
        Mean bid price mu_p^i ($/MWh).
    sigma_price : float
        Standard deviation of bid price sigma_p^i ($/MWh).
    seed : int, optional
    """

    def __init__(
            self,
            producer_id: str,
            capacity: float,
            mu_price: float,
            sigma_price: float,
            seed: Optional[int] = None,
    ) -> None:
        self.id = producer_id
        self.capacity = capacity
        self.mu_price = mu_price
        self.sigma_price = sigma_price
        self._rng = default_rng(seed)

        self.dispatched_quantity: np.ndarray = np.zeros(24)

    def get_bid(self, t: int) -> Bid:
        """Draw a stochastic bid price and return the bid with fixed capacity."""
        price = max(0.0, float(self._rng.normal(self.mu_price, self.sigma_price)))
        return Bid(quantity=self.capacity, price=price)

    def record_dispatch(self, t: int, q_disp: float) -> None:
        self.dispatched_quantity[t] = q_disp

    def reset(self) -> None:
        self.dispatched_quantity[:] = 0.0


# ---------------------------------------------------------------------------
# Agent 5 – Solar Producer
# ---------------------------------------------------------------------------

class SolarProducer:
    """
    Exogenous solar producer (Agent 5).

    Bid quantity follows the deterministic diurnal profile:
        q_t^S = max(0, a + b * cos(2 * pi * t / 24))

    Bid price is drawn i.i.d. from N(mu_price, sigma_price^2), truncated to >= 0.

    Parameters
    ----------
    a : float
        Baseline offset in the cosine profile (MWh).
    b : float
        Amplitude of the cosine profile (MWh).
    mu_price : float
        Mean bid price mu_p^S ($/MWh).
    sigma_price : float
        Standard deviation of bid price sigma_p^S ($/MWh).
    seed : int, optional
    """

    id = "solar"
    INTERVALS = 24

    def __init__(
            self,
            a: float,
            b: float,
            mu_price: float,
            sigma_price: float,
            seed: Optional[int] = None,
    ) -> None:
        self.a = a
        self.b = b
        self.mu_price = mu_price
        self.sigma_price = sigma_price
        self._rng = default_rng(seed)

        self.dispatched_quantity: np.ndarray = np.zeros(self.INTERVALS)

    def bid_quantity(self, t: int) -> float:
        """
        Evaluate the deterministic solar output profile at interval t.

        t is the 0-indexed interval from the simulation loop (t in {0,...,23}).
        The ODD specification defines intervals as 1-indexed (t in {1,...,24}),
        so the formula is evaluated at t+1 to preserve the correct diurnal phase:
            q_t^S = max(0, a + b * cos(2 * pi * (t+1) / 24))
        """
        return float(max(0.0, self.a + self.b * np.cos(2 * np.pi * (t + 1) / 24)))

    def get_bid(self, t: int) -> Bid:
        """
        Return a bid with deterministic quantity and stochastic price.
        t is 0-indexed in the simulation loop; bid_quantity handles the
        conversion to 1-indexed internally.
        """
        price = max(0.0, float(self._rng.normal(self.mu_price, self.sigma_price)))
        return Bid(quantity=self.bid_quantity(t), price=price)

    def record_dispatch(self, t: int, q_disp: float) -> None:
        self.dispatched_quantity[t] = q_disp

    def reset(self) -> None:
        self.dispatched_quantity[:] = 0.0


# ---------------------------------------------------------------------------
# Agent 6 – Market Operator
# ---------------------------------------------------------------------------

class MarketOperator:
    """
    Passive deterministic clearing and settlement agent (Agent 6).

    Implements Submodel 3 (merit-order clearing) and Submodel 4 (settlement).
    Also provides competitor-bid sampling used by WindProducer.optimise_bids.

    The fixed tie-breaking priority ordering is:
        solar > wind > conv1 > conv2 > conv3

    Parameters
    ----------
    mu_demand : float
        Mean market demand mu^D (MWh).
    sigma_demand : float
        Standard deviation of demand sigma^D (MWh).
    conventional_producers : list[ConventionalProducer]
        The three conventional producer agents.
    solar_producer : SolarProducer
        The solar producer agent.
    seed : int, optional
    """

    PRIORITY = ["solar", "wind", "conv1", "conv2", "conv3"]
    INTERVALS = 24

    def __init__(
            self,
            mu_demand: float,
            sigma_demand: float,
            conventional_producers: list,
            solar_producer: SolarProducer,
            seed: Optional[int] = None,
    ) -> None:
        self.mu_demand = mu_demand
        self.sigma_demand = sigma_demand
        self.conventional_producers = conventional_producers
        self.solar_producer = solar_producer
        self._rng = default_rng(seed)

        # Interval-level state
        self.clearing_prices: np.ndarray = np.zeros(self.INTERVALS)
        self.demand_realisations: np.ndarray = np.zeros(self.INTERVALS)

    # ------------------------------------------------------------------
    # Submodel 3 – Merit-order clearing
    # ------------------------------------------------------------------

    def clear_interval(
            self,
            t: int,
            bids: dict,
            demand: float,
    ) -> ClearingResult:
        """
        Apply the merit-order mechanism for one interval.

        Bids are sorted by ascending price; ties are resolved by the fixed
        priority ordering pi.  The marginal unit is the lowest-ranked
        producer whose cumulative quantity first meets or exceeds demand.
        The marginal unit may be dispatched partially.

        Parameters
        ----------
        t : int
            Interval index (0-based, not used in core logic but retained for
            interface consistency with optimisation submodels).
        bids : dict[str, Bid]
            Mapping from producer_id to Bid for this interval.
        demand : float
            Realised demand D_tilde_t (MWh).

        Returns
        -------
        ClearingResult
        """
        priority_rank = {pid: i for i, pid in enumerate(self.PRIORITY)}
        stack = sorted(
            bids.items(),
            key=lambda x: (x[1].price, priority_rank.get(x[0], 99)),
        )

        cumulative = 0.0
        dispatched = {pid: 0.0 for pid in bids}
        marginal_producer = stack[-1][0]
        clearing_price = stack[-1][1].price

        for pid, bid in stack:
            if cumulative >= demand:
                break
            residual = demand - cumulative
            if bid.quantity <= residual:
                # Infra-marginal or exact-match dispatch
                dispatched[pid] = bid.quantity
                cumulative += bid.quantity
                if cumulative >= demand:
                    # Full dispatch exactly met demand: this unit is marginal
                    marginal_producer = pid
                    clearing_price = bid.price
            else:
                # Partial dispatch: this unit is marginal and clears the market
                dispatched[pid] = residual
                cumulative = demand
                marginal_producer = pid
                clearing_price = bid.price
                break

        return ClearingResult(
            clearing_price=clearing_price,
            dispatched=dispatched,
            marginal_producer=marginal_producer,
        )

    # ------------------------------------------------------------------
    # Submodel 4 – Settlement
    # ------------------------------------------------------------------

    def settle_interval(
            self,
            clearing: ClearingResult,
            q_actual_wind: float,
            rho: float,
    ) -> SettlementResult:
        """
        Compute the wind producer's net revenue and shortfall penalty.

        The wind producer is remunerated at the clearing price for its full
        dispatched quantity, then separately penalised for any shortfall.

        Parameters
        ----------
        clearing : ClearingResult
            Output of clear_interval for this interval.
        q_actual_wind : float
            Realised wind output q_tilde_t^W (MWh).
        rho : float
            Penalty rate rho ($/MWh).

        Returns
        -------
        SettlementResult
        """
        q_disp = clearing.dispatched.get("wind", 0.0)
        shortfall = max(0.0, q_disp - q_actual_wind)
        revenue = clearing.clearing_price * q_disp
        penalty = rho * shortfall
        return SettlementResult(
            revenue=revenue,
            penalty=penalty,
            net_revenue=revenue - penalty,
            shortfall=shortfall,
        )

    # ------------------------------------------------------------------
    # Helper: sample competitor bids for MC optimisation
    # ------------------------------------------------------------------

    def sample_competitor_bids(
            self,
            t: int,
            n: int,
            rng: np.random.Generator,
    ) -> list:
        """
        Draw n independent realisations of competitor bids for interval t.
        Used internally by WindProducer.optimise_bids.

        Returns
        -------
        list of dict[str, Bid], length n
        """
        # Pre-draw all prices at once for efficiency
        solar_prices = np.maximum(
            rng.normal(self.solar_producer.mu_price, self.solar_producer.sigma_price, n),
            0.0,
        )
        conv_prices = {
            cp.id: np.maximum(rng.normal(cp.mu_price, cp.sigma_price, n), 0.0)
            for cp in self.conventional_producers
        }

        solar_qty = self.solar_producer.bid_quantity(t)

        samples = []
        for s in range(n):
            bids = {"solar": Bid(quantity=solar_qty, price=float(solar_prices[s]))}
            for cp in self.conventional_producers:
                bids[cp.id] = Bid(
                    quantity=cp.capacity,
                    price=float(conv_prices[cp.id][s]),
                )
            samples.append(bids)
        return samples

    # ------------------------------------------------------------------
    # Demand realisation (Step 3.1)
    # ------------------------------------------------------------------

    def realise_demand(self, t: int) -> float:
        """Draw D_tilde_t ~ N(mu^D, sigma^D^2), truncated to >= 0."""
        d = max(0.0, float(self._rng.normal(self.mu_demand, self.sigma_demand)))
        self.demand_realisations[t] = d
        return d

    def reset(self) -> None:
        """Reset interval-level state for a fresh simulation run."""
        self.clearing_prices[:] = 0.0
        self.demand_realisations[:] = 0.0


# ---------------------------------------------------------------------------
# Agent 7 – System Regulator
# ---------------------------------------------------------------------------

class SystemRegulator:
    """
    Policy-setting agent (Perspective 2, Agent 7).

    Chooses the penalty rate rho to minimise expected system costs arising
    from wind under-delivery, subject to the wind producer's participation
    constraint.  The regulator acts as Stackelberg leader: it anticipates
    the wind producer's best-response bid schedule q_t^W(rho).

    The participation constraint E[R_d^W(rho)] >= R_min is enforced via an
    exterior penalty term added to the objective.

    Parameters
    ----------
    cost_of_shortfall : float
        Social cost of undelivered electricity c_short ($/MWh).
    min_producer_revenue : float
        Minimum expected daily net revenue R_min for participation ($).
    rho_bounds : tuple[float, float]
        Search interval for rho.
    """

    def __init__(
            self,
            cost_of_shortfall: float,
            min_producer_revenue: float,
            rho_bounds: tuple = (0.0, 500.0),
    ) -> None:
        self.cost_of_shortfall = cost_of_shortfall
        self.min_producer_revenue = min_producer_revenue
        self.rho_bounds = rho_bounds

        self.optimal_rho: float = 0.0
        self.daily_shortfall: float = 0.0
        self.daily_penalty_collected: float = 0.0

    # ------------------------------------------------------------------
    # Submodel 2 – Penalty design
    # ------------------------------------------------------------------

    def optimise_penalty(
            self,
            wind_producer: WindProducer,
            market_operator: MarketOperator,
            n_mc_outer: int = 500,
            n_mc_inner: int = 1000,
            n_rho_grid: int = 20,
    ) -> float:
        """
        Solve the regulator's optimisation problem (Submodel 2).

        Strategy: coarse grid search over rho, followed by bounded scalar
        refinement around the grid minimum.  At each candidate rho, the
        wind producer's best-response bids are re-computed (lower-level
        problem), then expected system cost and wind revenue are estimated
        by Monte Carlo.

        Parameters
        ----------
        wind_producer : WindProducer
            The lower-level agent whose best response is computed for each rho.
        market_operator : MarketOperator
            Provides clearing simulation.
        n_mc_outer : int
            MC samples for evaluating E[C_d(rho)] given a candidate rho.
        n_mc_inner : int
            MC samples passed to wind producer bid optimisation.
        n_rho_grid : int
            Grid points for the coarse rho search.

        Returns
        -------
        float
            Optimal penalty rate rho*.
        """
        rng = default_rng(99)

        def system_cost_objective(rho: float) -> float:
            """
            Estimate E[C_d(rho)] + exterior participation-constraint penalty.
            """
            # Lower-level: re-optimise wind producer bids at this rho
            wind_producer.rho = float(rho)
            wind_producer.optimise_bids(market_operator, n_mc=n_mc_inner)

            total_shortfall = 0.0
            total_wind_revenue = 0.0

            for _ in range(n_mc_outer):
                day_shortfall = 0.0
                day_revenue = 0.0

                for t in range(24):
                    D = max(0.0, float(rng.normal(
                        market_operator.mu_demand, market_operator.sigma_demand,
                    )))
                    q_actual = float(np.clip(
                        rng.normal(wind_producer.mu_wind, wind_producer.sigma_wind),
                        0.0, wind_producer.capacity,
                    ))

                    bids = {
                        "wind": wind_producer.get_bid(t),
                        "solar": Bid(
                            quantity=market_operator.solar_producer.bid_quantity(t),
                            price=max(0.0, float(rng.normal(
                                market_operator.solar_producer.mu_price,
                                market_operator.solar_producer.sigma_price,
                            ))),
                        ),
                    }
                    for cp in market_operator.conventional_producers:
                        bids[cp.id] = Bid(
                            quantity=cp.capacity,
                            price=max(0.0, float(rng.normal(cp.mu_price, cp.sigma_price))),
                        )

                    result = market_operator.clear_interval(t, bids, D)
                    q_disp = result.dispatched.get("wind", 0.0)
                    shortfall = max(0.0, q_disp - q_actual)
                    net_rev = result.clearing_price * q_disp - float(rho) * shortfall

                    day_shortfall += shortfall
                    day_revenue += net_rev

                total_shortfall += day_shortfall
                total_wind_revenue += day_revenue

            exp_system_cost = self.cost_of_shortfall * total_shortfall / n_mc_outer
            exp_wind_revenue = total_wind_revenue / n_mc_outer

            # Exterior penalty for participation constraint violation
            violation = max(0.0, self.min_producer_revenue - exp_wind_revenue)
            return exp_system_cost + 1e4 * violation

        # --- Coarse grid search ---
        rho_grid = np.linspace(self.rho_bounds[0], self.rho_bounds[1], n_rho_grid)
        costs = np.array([system_cost_objective(r) for r in rho_grid])
        best_idx = int(np.argmin(costs))

        # --- Bounded refinement around grid minimum ---
        lo = rho_grid[max(best_idx - 1, 0)]
        hi = rho_grid[min(best_idx + 1, n_rho_grid - 1)]
        refined = minimize_scalar(
            system_cost_objective,
            bounds=(lo, hi),
            method="bounded",
        )
        self.optimal_rho = float(refined.x)

        # Restore wind producer with optimal rho and re-optimise its bids
        wind_producer.rho = self.optimal_rho
        wind_producer.optimise_bids(market_operator, n_mc=n_mc_inner)

        return self.optimal_rho

    def record_day_outcomes(self, E_d: float, rho: float) -> None:
        """Store aggregate shortfall and penalty collected at end of day."""
        self.daily_shortfall = E_d
        self.daily_penalty_collected = rho * E_d


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Collected outputs from one simulated day.

    Perspective 1 outputs (wind producer)
    --------------------------------------
    daily_net_revenue : float
        R_d^W = sum of r_t^W across all 24 intervals.
    interval_net_revenue : np.ndarray (24,)
        r_t^W for each interval t.
    interval_shortfall : np.ndarray (24,)
        e_tilde_t^W for each interval t.
    clearing_prices : np.ndarray (24,)
        lambda_t for each interval t.
    dispatched_wind : np.ndarray (24,)
        q_tilde_t^{W,disp} for each interval t.
    realised_wind : np.ndarray (24,)
        q_tilde_t^W for each interval t.
    demand_realisations : np.ndarray (24,)
        D_tilde_t for each interval t.

    Perspective 2 outputs (system regulator)
    -----------------------------------------
    total_daily_shortfall : float
        E_d^W = sum of e_tilde_t^W across all intervals.
    total_penalty_collected : float
        P_d^W = rho * E_d^W.
    rho : float
        Penalty rate applied in this simulation run.
    """
    daily_net_revenue: float
    interval_net_revenue: np.ndarray
    interval_shortfall: np.ndarray
    clearing_prices: np.ndarray
    dispatched_wind: np.ndarray
    realised_wind: np.ndarray
    demand_realisations: np.ndarray
    total_daily_shortfall: float
    total_penalty_collected: float
    rho: float


# ---------------------------------------------------------------------------
# Internal simulation engine
# ---------------------------------------------------------------------------

def _run_single_day(
        wind_producer: WindProducer,
        conventional_producers: list,
        solar_producer: SolarProducer,
        market_operator: MarketOperator,
        system_regulator: SystemRegulator,
        rho: float,
) -> SimulationResult:
    """
    Execute one full simulated day (Phases 2 and 3 of the ODD schedule).

    Phase 2 (bid submission) has already been completed by the time this
    function is called; it reads the pre-committed bid schedules from each
    agent.  Phase 3 iterates over all 24 intervals.
    """
    # Reset all interval-level agent state
    wind_producer.reset()
    market_operator.reset()
    for cp in conventional_producers:
        cp.reset()
    solar_producer.reset()

    clearing_prices = np.zeros(24)
    dispatched_wind = np.zeros(24)

    for t in range(24):

        # Step 3.1 – Stochastic realisations
        demand = market_operator.realise_demand(t)
        q_actual_wind = wind_producer.realise_output(t)

        # Step 3.2 – Collect bids and clear market
        bids: dict = {"wind": wind_producer.get_bid(t)}
        bids["solar"] = solar_producer.get_bid(t)
        for cp in conventional_producers:
            bids[cp.id] = cp.get_bid(t)

        clearing = market_operator.clear_interval(t, bids, demand)
        clearing_prices[t] = clearing.clearing_price
        market_operator.clearing_prices[t] = clearing.clearing_price

        # Record dispatch for all agents
        q_disp_wind = clearing.dispatched.get("wind", 0.0)
        wind_producer.record_dispatch(t, q_disp_wind)
        dispatched_wind[t] = q_disp_wind
        solar_producer.record_dispatch(t, clearing.dispatched.get("solar", 0.0))
        for cp in conventional_producers:
            cp.record_dispatch(t, clearing.dispatched.get(cp.id, 0.0))

        # Steps 3.3–3.4 – Delivery, shortfall, and settlement
        settlement = market_operator.settle_interval(clearing, q_actual_wind, rho)
        wind_producer.record_settlement(t, settlement)

    # Step 3.5 – End-of-day aggregation
    E_d = float(np.sum(wind_producer.shortfall))
    system_regulator.record_day_outcomes(E_d, rho)

    return SimulationResult(
        daily_net_revenue=wind_producer.accumulated_net_revenue,
        interval_net_revenue=wind_producer.interval_net_revenue.copy(),
        interval_shortfall=wind_producer.shortfall.copy(),
        clearing_prices=clearing_prices,
        dispatched_wind=dispatched_wind,
        realised_wind=wind_producer.realised_output.copy(),
        demand_realisations=market_operator.demand_realisations.copy(),
        total_daily_shortfall=system_regulator.daily_shortfall,
        total_penalty_collected=system_regulator.daily_penalty_collected,
        rho=rho,
    )


# ---------------------------------------------------------------------------
# Interface function
# ---------------------------------------------------------------------------

def run_simulation(
        # ---- Techno-economic constants ----
        wind_capacity: float = 200.0,
        mu_wind: float = 100.0,
        sigma_wind: float = 20.0,
        solar_a: float = 60.0,
        solar_b: float = 80.0,
        conventional_capacities: tuple = (150.0, 150.0, 150.0),
        # ---- Demand parameters ----
        mu_demand: float = 400.0,
        sigma_demand: float = 30.0,
        # ---- Uncertain competitor bid-price parameters ----
        conv_mu_prices: tuple = (40.0, 55.0, 70.0),
        conv_sigma_prices: tuple = (5.0, 5.0, 5.0),
        solar_mu_price: float = 5.0,
        solar_sigma_price: float = 2.0,
        # ---- Perspective / modular controls ----
        perspective: str = "composed",
        rho_exogenous: float = 30.0,
        # ---- Regulator parameters (Perspectives 2 and composed) ----
        cost_of_shortfall: float = 150.0,
        min_producer_revenue: float = 500.0,
        rho_bounds: tuple = (0.0, 200.0),
        # ---- Simulation settings ----
        n_days: int = 1,
        n_mc_optimisation: int = 1000,
        seed: int = 42,
) -> dict:
    """
    Single interface function for the unified wind-market ABM.

    Instantiates all seven agents, resolves the active perspective(s),
    runs the required optimisation submodels, and simulates n_days of
    market operations.

    Parameters
    ----------
    wind_capacity : float
        Installed wind capacity Q_bar^W (MWh).
    mu_wind : float
        Mean wind output mu^W (MWh).
    sigma_wind : float
        Standard deviation of wind output sigma^W (MWh).
    solar_a : float
        Baseline offset in the solar diurnal profile (MWh).
    solar_b : float
        Amplitude of the solar diurnal profile (MWh).
    conventional_capacities : tuple of 3 floats
        Nameplate capacities (K_bar^c1, K_bar^c2, K_bar^c3) in MWh.
    mu_demand : float
        Mean market demand mu^D (MWh).
    sigma_demand : float
        Standard deviation of demand sigma^D (MWh).
    conv_mu_prices : tuple of 3 floats
        Mean bid prices (mu_p^c1, mu_p^c2, mu_p^c3) in $/MWh.
    conv_sigma_prices : tuple of 3 floats
        Standard deviations of conventional producer bid prices ($/MWh).
    solar_mu_price : float
        Mean bid price of the solar producer mu_p^S ($/MWh).
    solar_sigma_price : float
        Standard deviation of solar bid price sigma_p^S ($/MWh).
    perspective : str
        Modular evaluation mode.  One of:
            'perspective1'  – Perspective 1 only: optimise wind bids,
                              rho fixed at rho_exogenous.
            'perspective2'  – Perspective 2 only: optimise rho,
                              wind bids fixed at mu^W (no optimisation).
            'composed'      – Full bilevel model: regulator optimises rho
                              as Stackelberg leader; wind producer optimises
                              bids as follower.  (default)
    rho_exogenous : float
        Penalty rate used when perspective = 'perspective1' ($/MWh).
    cost_of_shortfall : float
        Social cost c_short ($/MWh); used in Perspectives 2 and composed.
    min_producer_revenue : float
        Participation constraint R_min ($); used in Perspectives 2 and
        composed.
    rho_bounds : tuple[float, float]
        Search bounds for rho in the regulator's optimisation.
    n_days : int
        Number of days to simulate after optimisation.
    n_mc_optimisation : int
        Monte Carlo sample size passed to bid and penalty optimisation.
    seed : int
        Master random seed for full reproducibility.

    Returns
    -------
    dict with keys:
        'perspective'         : str             active perspective label
        'optimal_rho'         : float           penalty rate used
        'wind_bid_quantities' : np.ndarray(24)  optimised bid quantities
        'results'             : list[SimulationResult]  one entry per day
        'summary'             : dict            mean/std of key daily outputs
        'agents'              : dict            all seven instantiated agents
    """
    if perspective not in {"perspective1", "perspective2", "composed"}:
        raise ValueError(
            "perspective must be 'perspective1', 'perspective2', or 'composed'."
        )

    # ---------------------------------------------------------------
    # Seed management: derive independent seeds for each agent
    # ---------------------------------------------------------------
    rng_master = default_rng(seed)
    agent_seeds = rng_master.integers(0, 1 << 31, size=10).tolist()

    # ---------------------------------------------------------------
    # Instantiate all seven agents
    # ---------------------------------------------------------------

    conventional_producers = [
        ConventionalProducer(
            producer_id=f"conv{i + 1}",
            capacity=conventional_capacities[i],
            mu_price=conv_mu_prices[i],
            sigma_price=conv_sigma_prices[i],
            seed=agent_seeds[i],
        )
        for i in range(3)
    ]

    solar_producer = SolarProducer(
        a=solar_a,
        b=solar_b,
        mu_price=solar_mu_price,
        sigma_price=solar_sigma_price,
        seed=agent_seeds[3],
    )

    wind_producer = WindProducer(
        capacity=wind_capacity,
        mu_wind=mu_wind,
        sigma_wind=sigma_wind,
        rho=rho_exogenous,
        seed=agent_seeds[4],
    )

    market_operator = MarketOperator(
        mu_demand=mu_demand,
        sigma_demand=sigma_demand,
        conventional_producers=conventional_producers,
        solar_producer=solar_producer,
        seed=agent_seeds[5],
    )

    system_regulator = SystemRegulator(
        cost_of_shortfall=cost_of_shortfall,
        min_producer_revenue=min_producer_revenue,
        rho_bounds=rho_bounds,
    )

    # ---------------------------------------------------------------
    # Phase 1 – Pre-market policy setting and bid optimisation
    # ---------------------------------------------------------------

    n_mc_inner = max(n_mc_optimisation // 10, 50)
    n_mc_outer = max(n_mc_optimisation // 5, 50)

    if perspective == "perspective1":
        # Perspective 1: rho is exogenous; optimise wind bids only
        rho = rho_exogenous
        wind_producer.rho = rho
        print(f"[Perspective 1] Optimising wind bids with rho = {rho:.2f} $/MWh ...")
        wind_producer.optimise_bids(market_operator, n_mc=n_mc_optimisation)
        print("[Perspective 1] Bid optimisation complete.")

    elif perspective == "perspective2":
        # Perspective 2: wind bids fixed at mu^W; optimise rho only
        wind_producer.bid_quantities = np.full(24, mu_wind)
        wind_producer.bid_prices = np.zeros(24)
        print("[Perspective 2] Optimising penalty rate rho (wind bids fixed at mu^W) ...")
        rho = system_regulator.optimise_penalty(
            wind_producer, market_operator,
            n_mc_outer=n_mc_outer,
            n_mc_inner=n_mc_inner,
        )
        print(f"[Perspective 2] Optimal rho* = {rho:.4f} $/MWh")

    else:
        # Composed: full bilevel – regulator leads, wind producer follows
        print("[Composed] Solving bilevel problem ...")
        print("  Step 1: Regulator optimises rho (with wind producer best response) ...")
        rho = system_regulator.optimise_penalty(
            wind_producer, market_operator,
            n_mc_outer=n_mc_outer,
            n_mc_inner=n_mc_inner,
        )
        print(f"  Step 1 complete. Optimal rho* = {rho:.4f} $/MWh")
        print("  Step 2: Re-optimising wind bids at rho* with full MC budget ...")
        wind_producer.rho = rho
        wind_producer.optimise_bids(market_operator, n_mc=n_mc_optimisation)
        print("  Step 2 complete.")

    # ---------------------------------------------------------------
    # Phases 2–3 – Simulate n_days days
    # ---------------------------------------------------------------

    print(f"\nSimulating {n_days} day(s) ...")
    results = []
    for day in range(n_days):
        result = _run_single_day(
            wind_producer=wind_producer,
            conventional_producers=conventional_producers,
            solar_producer=solar_producer,
            market_operator=market_operator,
            system_regulator=system_regulator,
            rho=rho,
        )
        results.append(result)

    # ---------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------

    daily_revenues = np.array([r.daily_net_revenue for r in results])
    daily_shortfalls = np.array([r.total_daily_shortfall for r in results])
    daily_penalties = np.array([r.total_penalty_collected for r in results])
    mean_clearing_price = float(np.mean([r.clearing_prices for r in results]))

    summary = {
        "mean_daily_net_revenue": float(np.mean(daily_revenues)),
        "std_daily_net_revenue": float(np.std(daily_revenues)),
        "mean_daily_shortfall_MWh": float(np.mean(daily_shortfalls)),
        "std_daily_shortfall_MWh": float(np.std(daily_shortfalls)),
        "mean_daily_penalty": float(np.mean(daily_penalties)),
        "mean_clearing_price": mean_clearing_price,
    }

    print("\n=== Simulation Summary ===")
    print(f"  Perspective              : {perspective}")
    print(f"  Penalty rate (rho)       : {rho:.4f} $/MWh")
    print(f"  Days simulated           : {n_days}")
    print(f"  Mean daily net revenue   : ${summary['mean_daily_net_revenue']:>10,.2f}")
    print(f"  Std  daily net revenue   : ${summary['std_daily_net_revenue']:>10,.2f}")
    print(f"  Mean daily shortfall     : {summary['mean_daily_shortfall_MWh']:>10.2f} MWh")
    print(f"  Mean daily penalty paid  : ${summary['mean_daily_penalty']:>10,.2f}")
    print(f"  Mean clearing price      : ${summary['mean_clearing_price']:>10.2f} /MWh")

    return {
        "perspective": perspective,
        "optimal_rho": rho,
        "wind_bid_quantities": wind_producer.bid_quantities.copy(),
        "results": results,
        "summary": summary,
        "agents": {
            "wind_producer": wind_producer,
            "conventional_producers": conventional_producers,
            "solar_producer": solar_producer,
            "market_operator": market_operator,
            "system_regulator": system_regulator,
        },
    }


# ---------------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv

    # ------------------------------------------------------------------
    # Table 4 parameters
    # ------------------------------------------------------------------
    WIND_CAPACITY = 3000.0  # not in Table 4; set above mu_G + 4*sigma_G
    MU_WIND = 275.0  # mu_G
    SIGMA_WIND = 50.0  # sigma_G
    SOLAR_A = 0.0  # a
    SOLAR_B = -400.0  # b
    SOLAR_MU_P = 35.0  # mu_ps
    SOLAR_SIGMA_P = 4.0  # sigma_ps
    CONV_CAPACITIES = (300.0, 250.0, 1000.0)  # b_i
    CONV_MU_P = (45.0, 50.0, 60.0)  # mu_pi
    CONV_SIGMA_P = (2.0, 2.0, 2.0)  # sigma_pi
    MU_DEMAND = 800.0  # mu_D
    SIGMA_DEMAND = 20.0  # sigma_D
    WIND_BID_QTY = 300.0  # b_wt (fixed for all t)
    WIND_BID_PRICE = 50.0  # p_wt (fixed for all t)
    RHO_VALUES = [20.0, 100.0, 180.0]  # q_u values
    N_RUNS = 10
    BASE_SEED = None

    # ------------------------------------------------------------------
    # Derive independent seeds for each (rho, run) combination
    # ------------------------------------------------------------------
    rng_master = default_rng(BASE_SEED)
    run_seeds = rng_master.integers(
        0, 1 << 31, size=(len(RHO_VALUES), N_RUNS)
    ).tolist()

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    records = []

    for rho_idx, rho in enumerate(RHO_VALUES):
        print(f"\n{'=' * 55}")
        print(f"  Penalty rate rho = {rho:.0f} $/MWh")
        print(f"{'=' * 55}")

        for run in range(N_RUNS):
            seed = run_seeds[rho_idx][run]
            rng_agents = default_rng(seed)
            agent_seeds = rng_agents.integers(0, 1 << 31, size=10).tolist()

            # Instantiate all seven agents
            conventional_producers = [
                ConventionalProducer(
                    producer_id=f"conv{i + 1}",
                    capacity=CONV_CAPACITIES[i],
                    mu_price=CONV_MU_P[i],
                    sigma_price=CONV_SIGMA_P[i],
                    seed=agent_seeds[i],
                )
                for i in range(3)
            ]
            solar_producer = SolarProducer(
                a=SOLAR_A,
                b=SOLAR_B,
                mu_price=SOLAR_MU_P,
                sigma_price=SOLAR_SIGMA_P,
                seed=agent_seeds[3],
            )
            wind_producer = WindProducer(
                capacity=WIND_CAPACITY,
                mu_wind=MU_WIND,
                sigma_wind=SIGMA_WIND,
                rho=rho,
                seed=agent_seeds[4],
            )
            market_operator = MarketOperator(
                mu_demand=MU_DEMAND,
                sigma_demand=SIGMA_DEMAND,
                conventional_producers=conventional_producers,
                solar_producer=solar_producer,
                seed=agent_seeds[5],
            )
            system_regulator = SystemRegulator(
                cost_of_shortfall=0.0,  # not used with fixed bids
                min_producer_revenue=0.0,
            )

            # Fix wind bid schedule at (b_wt, p_wt) = (300, 50) for all t
            wind_producer.bid_quantities = np.full(24, WIND_BID_QTY)
            wind_producer.bid_prices = np.full(24, WIND_BID_PRICE)

            # Run one simulated day
            result = _run_single_day(
                wind_producer=wind_producer,
                conventional_producers=conventional_producers,
                solar_producer=solar_producer,
                market_operator=market_operator,
                system_regulator=system_regulator,
                rho=rho,
            )

            # Store interval-level outputs
            for t in range(24):
                records.append({
                    "rho": rho,
                    "run": run + 1,
                    "interval": t + 1,  # 1-indexed per ODD spec
                    "dispatched_wind_MWh": result.dispatched_wind[t],
                    "net_revenue_USD": result.interval_net_revenue[t],
                })

            print(
                f"  Run {run + 1:2d}  |  "
                f"R_d^W = ${result.daily_net_revenue:>10,.2f}  |  "
                f"E_d^W = {result.total_daily_shortfall:>7.2f} MWh"
            )

    # ------------------------------------------------------------------
    # Save results to CSV
    # ------------------------------------------------------------------
    OUTPUT_FILE = "market1_results.csv"
    fieldnames = ["rho", "run", "interval", "dispatched_wind_MWh", "net_revenue_USD"]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nResults saved to '{OUTPUT_FILE}'")
    print(f"Total records: {len(records)} "
          f"({len(RHO_VALUES)} rho values × {N_RUNS} runs × 24 intervals)")
