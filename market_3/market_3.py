"""
wind_power_abm.py
=================
Agent-Based Model: Day-Ahead Wind-Power Revenue Estimation
Specification: ODD Protocol (Unified Model)

Agents
------
  ConventionalProducer  – stable bid quantity, stochastic bid price
  SolarProducer         – cosine bid quantity, stochastic bid price
  WindProducer          – user-specified bid strategy, stochastic output
  MarketOperator        – merit-order clearing with priority tie-breaking

Perspectives
------------
  WindProducerPerspective  – shortfall, net revenue, daily aggregation
  SystemRegulatorPerspective – system cost, renewable penetration, shortfall

Interface
---------
  run_simulation(...)  – single entry-point for all evaluation modes
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MarketOutcome:
    """Clearing results for a single interval."""
    clearing_price: float  # lambda_t  ($/MWh)
    dispatched_quantities: dict  # {agent_id: q_t}  (MWh)
    residual_demand: float  # D^res_t  (MWh)


@dataclass
class SimulationResult:
    """
    Outputs returned by run_simulation().

    Arrays are shaped (N, 24), where N is the number of replications
    and 24 is the number of hourly intervals.  Scalar summaries are
    shaped (N,).
    """
    # --- Wind-producer perspective -----------------------------------------
    clearing_prices: NDArray  # (N, 24)  lambda_t
    wind_dispatched: NDArray  # (N, 24)  q^W_t
    wind_realised: NDArray  # (N, 24)  Q^W_t (actual output)
    wind_shortfall: NDArray  # (N, 24)  delta^W_t
    wind_interval_revenue: NDArray  # (N, 24)  r^W_t
    wind_daily_revenue: NDArray  # (N,)     R^W

    # --- Regulator perspective ---------------------------------------------
    system_interval_cost: NDArray  # (N, 24)  E_t
    renewable_fraction: NDArray  # (N, 24)  F_t
    system_daily_cost: NDArray  # (N,)     E
    mean_renewable_frac: NDArray  # (N,)     F-bar
    total_wind_shortfall: NDArray  # (N,)     Delta^W

    # --- Summary statistics ------------------------------------------------
    def summary(self) -> dict:
        """Return expected values and standard deviations for key outputs."""
        return {
            # Wind-producer perspective
            "E[R^W]": float(np.mean(self.wind_daily_revenue)),
            "Std[R^W]": float(np.std(self.wind_daily_revenue)),
            "CVaR5[R^W]": float(_cvar(self.wind_daily_revenue, alpha=0.05)),
            "E[Delta^W]": float(np.mean(self.total_wind_shortfall)),
            # Regulator perspective
            "E[E]": float(np.mean(self.system_daily_cost)),
            "Std[E]": float(np.std(self.system_daily_cost)),
            "E[F-bar]": float(np.mean(self.mean_renewable_frac)),
            "E[Delta^W_reg]": float(np.mean(self.total_wind_shortfall)),
        }


def _cvar(values: NDArray, alpha: float = 0.05) -> float:
    """Conditional Value-at-Risk at level alpha (left tail)."""
    threshold = np.quantile(values, alpha)
    tail = values[values <= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else float(threshold)


# ---------------------------------------------------------------------------
# Agent: ConventionalProducer
# ---------------------------------------------------------------------------

class ConventionalProducer:
    """
    Conventional (coal) producer.

    Bid quantity  : fixed constant  v^{C_k}_t = Q_bar^{C_k}  for all t
    Bid price     : i.i.d. normal   b^{C_k}_t ~ N(mu_b, sigma2_b)
    """

    def __init__(
            self,
            agent_id: str,
            capacity: float,
            mu_b: float,
            sigma2_b: float,
            price_floor: float = 0.0,
            price_cap: float = float("inf"),
    ) -> None:
        """
        Parameters
        ----------
        agent_id   : unique label, e.g. 'C1'
        capacity   : installed capacity Q_bar^{C_k}  (MWh)
        mu_b       : mean of bid price distribution   ($/MWh)
        sigma2_b   : variance of bid price distribution
        price_floor: market bid floor  p_under  ($/MWh)
        price_cap  : market bid cap    p_bar    ($/MWh)
        """
        self.agent_id = agent_id
        self.capacity = capacity
        self.mu_b = mu_b
        self.sigma_b = np.sqrt(sigma2_b)
        self.price_floor = price_floor
        self.price_cap = price_cap

    def submit_bids(
            self, n_intervals: int, rng: np.random.Generator
    ) -> tuple[NDArray, NDArray]:
        """
        Draw bid prices and return (bid_quantities, bid_prices) arrays
        of shape (n_intervals,).
        """
        quantities = np.full(n_intervals, self.capacity)
        prices = rng.normal(self.mu_b, self.sigma_b, size=n_intervals)
        prices = np.clip(prices, self.price_floor, self.price_cap)
        return quantities, prices


# ---------------------------------------------------------------------------
# Agent: SolarProducer
# ---------------------------------------------------------------------------

class SolarProducer:
    """
    Solar PV producer.

    Bid quantity  : deterministic cosine profile (truncated at 0)
                    v^S_t = max(0, a + b * cos(2*pi*t / 24))
    Bid price     : i.i.d. normal  b^S_t ~ N(mu_b, sigma2_b)
    """

    def __init__(
            self,
            a: float,
            b: float,
            mu_b: float,
            sigma2_b: float,
            price_floor: float = 0.0,
            price_cap: float = float("inf"),
            agent_id: str = "S",
    ) -> None:
        """
        Parameters
        ----------
        a          : baseline offset of cosine profile  (MWh)
        b          : amplitude of cosine profile        (MWh)
        mu_b       : mean of bid price distribution     ($/MWh)
        sigma2_b   : variance of bid price distribution
        price_floor: market bid floor  ($/MWh)
        price_cap  : market bid cap    ($/MWh)
        """
        self.agent_id = agent_id
        self.a = a
        self.b = b
        self.mu_b = mu_b
        self.sigma_b = np.sqrt(sigma2_b)
        self.price_floor = price_floor
        self.price_cap = price_cap

    def _cosine_quantities(self, n_intervals: int) -> NDArray:
        """Compute deterministic bid quantities for intervals 1..T."""
        t = np.arange(1, n_intervals + 1, dtype=float)
        return np.maximum(0.0, self.a + self.b * np.cos(2 * np.pi * t / n_intervals))

    def submit_bids(
            self, n_intervals: int, rng: np.random.Generator
    ) -> tuple[NDArray, NDArray]:
        """
        Return (bid_quantities, bid_prices) arrays of shape (n_intervals,).
        Quantities are deterministic; prices are stochastic.
        """
        quantities = self._cosine_quantities(n_intervals)
        prices = rng.normal(self.mu_b, self.sigma_b, size=n_intervals)
        prices = np.clip(prices, self.price_floor, self.price_cap)
        return quantities, prices


# ---------------------------------------------------------------------------
# Agent: WindProducer
# ---------------------------------------------------------------------------

class WindProducer:
    """
    Wind producer.

    Bid quantity / price : user-specified exogenous inputs
    Realised output      : i.i.d. normal, truncated to [0, Q_bar^W]
                           Q^W_t ~ N(mu_W, sigma2_W)
    """

    def __init__(
            self,
            capacity: float,
            mu_W: float,
            sigma2_W: float,
            agent_id: str = "W",
    ) -> None:
        """
        Parameters
        ----------
        capacity  : nameplate capacity Q_bar^W  (MWh)
        mu_W      : mean of wind output distribution    (MWh)
        sigma2_W  : variance of wind output distribution
        """
        self.agent_id = agent_id
        self.capacity = capacity
        self.mu_W = mu_W
        self.sigma_W = np.sqrt(sigma2_W)

    def submit_bids(
            self,
            bid_quantities: NDArray,
            bid_prices: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Validate and return the user-supplied bid arrays.

        Parameters
        ----------
        bid_quantities : (T,) array  v^W_t in [0, Q_bar^W]
        bid_prices     : (T,) array  b^W_t

        Returns
        -------
        (bid_quantities, bid_prices) validated copies
        """
        q = np.asarray(bid_quantities, dtype=float)
        p = np.asarray(bid_prices, dtype=float)
        if q.shape != p.shape:
            raise ValueError("bid_quantities and bid_prices must have the same shape.")
        if np.any(q < 0) or np.any(q > self.capacity):
            raise ValueError(
                f"bid_quantities must lie in [0, {self.capacity}]. "
                f"Got min={q.min():.3f}, max={q.max():.3f}."
            )
        return q.copy(), p.copy()

    def realise_output(
            self, n_intervals: int, rng: np.random.Generator
    ) -> NDArray:
        """
        Draw realised wind output Q^W_t for each interval.
        Truncated to [0, capacity].
        """
        raw = rng.normal(self.mu_W, self.sigma_W, size=n_intervals)
        return np.clip(raw, 0.0, self.capacity)


# ---------------------------------------------------------------------------
# Agent: MarketOperator
# ---------------------------------------------------------------------------

class MarketOperator:
    """
    Passive market operator.

    Implements merit-order clearing with:
      - uniform clearing price (pay-as-cleared)
      - priority tie-breaking: S > W > C1 > C2 > C3
    """

    # Fixed dispatch priority order (lower index = higher priority)
    PRIORITY_ORDER: list[str] = ["S", "W", "C1", "C2", "C3"]

    def __init__(self, demand_floor: float = 1.0) -> None:
        """
        Parameters
        ----------
        demand_floor : minimum admissible demand D_under  (MWh)
        """
        self.demand_floor = demand_floor

    def _priority_rank(self, agent_id: str) -> int:
        """Return the dispatch priority rank of an agent (lower = higher priority)."""
        try:
            return self.PRIORITY_ORDER.index(agent_id)
        except ValueError:
            return len(self.PRIORITY_ORDER)  # unknown agents dispatched last

    def clear_interval(
            self,
            demand: float,
            bid_quantities: dict[str, float],
            bid_prices: dict[str, float],
    ) -> MarketOutcome:
        """
        Clear a single interval via the merit-order mechanism.

        Parameters
        ----------
        demand         : realised demand D_t  (MWh)
        bid_quantities : {agent_id: v^i_t}
        bid_prices     : {agent_id: b^i_t}

        Returns
        -------
        MarketOutcome with clearing price and dispatched quantities.
        """
        demand = max(demand, self.demand_floor)
        agent_ids = list(bid_quantities.keys())

        # --- Step 1: rank bids by price, break ties by priority -----------
        sorted_agents = sorted(
            agent_ids,
            key=lambda i: (bid_prices[i], self._priority_rank(i)),
        )

        # --- Step 2: find marginal price ----------------------------------
        cumulative = 0.0
        k_star = None
        for agent in sorted_agents:
            cumulative += bid_quantities[agent]
            if cumulative >= demand:
                k_star = agent
                break

        if k_star is None:
            # Supply cannot meet demand; dispatch everything at highest price
            k_star = sorted_agents[-1]

        clearing_price = bid_prices[k_star]

        # --- Step 3: resolve ties at the marginal price -------------------
        infra_marginal = [a for a in agent_ids if bid_prices[a] < clearing_price]
        marginal_set = [a for a in agent_ids if bid_prices[a] == clearing_price]
        # supra-marginal agents receive zero dispatch implicitly

        infra_supply = sum(bid_quantities[a] for a in infra_marginal)
        residual_demand = max(0.0, demand - infra_supply)

        # Sort marginal-set by priority order
        marginal_set_sorted = sorted(
            marginal_set, key=lambda i: self._priority_rank(i)
        )

        dispatched: dict[str, float] = {a: 0.0 for a in agent_ids}

        # Fully dispatch all infra-marginal producers
        for a in infra_marginal:
            dispatched[a] = bid_quantities[a]

        # Dispatch marginal producers in priority order
        remaining = residual_demand
        for a in marginal_set_sorted:
            if remaining <= 0.0:
                break
            allocation = min(bid_quantities[a], remaining)
            dispatched[a] = allocation
            remaining -= allocation

        return MarketOutcome(
            clearing_price=clearing_price,
            dispatched_quantities=dispatched,
            residual_demand=residual_demand,
        )

    def clear_day(
            self,
            demands: NDArray,
            bid_quantities: dict[str, NDArray],
            bid_prices: dict[str, NDArray],
    ) -> tuple[NDArray, dict[str, NDArray]]:
        """
        Clear all T intervals for a single simulated day.

        Parameters
        ----------
        demands        : (T,) array of realised demands
        bid_quantities : {agent_id: (T,) array}
        bid_prices     : {agent_id: (T,) array}

        Returns
        -------
        clearing_prices : (T,) array
        dispatched      : {agent_id: (T,) array}
        """
        n = len(demands)
        agent_ids = list(bid_quantities.keys())
        clearing_prices = np.zeros(n)
        dispatched = {a: np.zeros(n) for a in agent_ids}

        for t in range(n):
            q_t = {a: float(bid_quantities[a][t]) for a in agent_ids}
            p_t = {a: float(bid_prices[a][t]) for a in agent_ids}
            outcome = self.clear_interval(float(demands[t]), q_t, p_t)
            clearing_prices[t] = outcome.clearing_price
            for a in agent_ids:
                dispatched[a][t] = outcome.dispatched_quantities[a]

        return clearing_prices, dispatched


# ---------------------------------------------------------------------------
# Perspective: WindProducerPerspective
# ---------------------------------------------------------------------------

class WindProducerPerspective:
    """
    Wind-producer perspective (P_W).

    Computes shortfall, interval net revenue, and daily net revenue
    given market outcomes and realised wind output.

    Penalty mechanism
    -----------------
    delta^W_t = max(0, q^W_t - Q^W_t)
    r^W_t     = lambda_t * q^W_t - rho * delta^W_t
    R^W       = sum_t r^W_t
    """

    def __init__(self, penalty_rate: float) -> None:
        """
        Parameters
        ----------
        penalty_rate : rho, penalty cost per MWh of under-delivery  ($/MWh)
        """
        self.penalty_rate = penalty_rate

    def evaluate(
            self,
            clearing_prices: NDArray,
            dispatched_wind: NDArray,
            realised_wind: NDArray,
    ) -> dict[str, NDArray]:
        """
        Compute all P_W state variables for a single simulated day.

        Parameters
        ----------
        clearing_prices : (T,) lambda_t
        dispatched_wind : (T,) q^W_t
        realised_wind   : (T,) Q^W_t (actual output)

        Returns
        -------
        dict with keys: shortfall, interval_revenue, daily_revenue
        """
        shortfall = np.maximum(0.0, dispatched_wind - realised_wind)
        interval_revenue = clearing_prices * dispatched_wind - self.penalty_rate * shortfall
        daily_revenue = float(np.sum(interval_revenue))

        return {
            "shortfall": shortfall,
            "interval_revenue": interval_revenue,
            "daily_revenue": daily_revenue,
        }


# ---------------------------------------------------------------------------
# Perspective: SystemRegulatorPerspective
# ---------------------------------------------------------------------------

class SystemRegulatorPerspective:
    """
    System regulator perspective (P_R).

    Monitors system cost, renewable penetration, and wind shortfall.

    E_t   = lambda_t * D_t
    F_t   = (q^W_t + q^S_t) / D_t
    E     = sum_t E_t
    F-bar = mean_t F_t
    Delta^W = sum_t delta^W_t
    """

    def evaluate(
            self,
            clearing_prices: NDArray,
            demands: NDArray,
            dispatched_wind: NDArray,
            dispatched_solar: NDArray,
            wind_shortfall: NDArray,
    ) -> dict[str, NDArray | float]:
        """
        Compute all P_R state variables for a single simulated day.

        Parameters
        ----------
        clearing_prices  : (T,) lambda_t
        demands          : (T,) D_t
        dispatched_wind  : (T,) q^W_t
        dispatched_solar : (T,) q^S_t
        wind_shortfall   : (T,) delta^W_t

        Returns
        -------
        dict with keys: interval_cost, renewable_fraction,
                        daily_cost, mean_renewable_frac, total_shortfall
        """
        interval_cost = clearing_prices * demands
        renewable_fraction = np.where(
            demands > 0,
            (dispatched_wind + dispatched_solar) / demands,
            0.0,
        )

        return {
            "interval_cost": interval_cost,
            "renewable_fraction": renewable_fraction,
            "daily_cost": float(np.sum(interval_cost)),
            "mean_renewable_frac": float(np.mean(renewable_fraction)),
            "total_shortfall": float(np.sum(wind_shortfall)),
        }


# ---------------------------------------------------------------------------
# Interface function
# ---------------------------------------------------------------------------

def run_simulation(
        # --- Wind-producer inputs (P_W controls) ---
        wind_bid_quantities: NDArray,
        wind_bid_prices: NDArray,
        # --- Environment: wind agent ---
        wind_capacity: float,
        wind_mu: float,
        wind_sigma2: float,
        # --- Environment: solar agent ---
        solar_a: float,
        solar_b: float,
        solar_mu_b: float,
        solar_sigma2_b: float,
        # --- Environment: conventional agents ---
        conv_capacities: list[float],
        conv_mu_b: list[float],
        conv_sigma2_b: list[float],
        # --- Market constants ---
        demand_mu: float,
        demand_sigma2: float,
        demand_floor: float,
        price_floor: float,
        price_cap: float,
        penalty_rate: float,
        # --- Simulation settings ---
        n_replications: int = 10_000,
        n_intervals: int = 24,
        seed: Optional[int] = None,
        # --- Modular evaluation flags ---
        evaluate_wind_perspective: bool = True,
        evaluate_regulator_perspective: bool = True,
) -> SimulationResult:
    """
    Single entry-point for the unified day-ahead wind-power ABM.

    Parameters
    ----------
    wind_bid_quantities : (T,) array  v^W_t in [0, wind_capacity]
    wind_bid_prices     : (T,) array  b^W_t in [price_floor, price_cap]
    wind_capacity       : nameplate capacity Q_bar^W  (MWh)
    wind_mu             : mean of wind output  mu_W   (MWh)
    wind_sigma2         : variance of wind output sigma2_W
    solar_a             : cosine profile offset  a    (MWh)
    solar_b             : cosine profile amplitude b  (MWh)
    solar_mu_b          : mean of solar bid price    ($/MWh)
    solar_sigma2_b      : variance of solar bid price
    conv_capacities     : list of 3 conventional capacities  (MWh)
    conv_mu_b           : list of 3 conventional bid price means
    conv_sigma2_b       : list of 3 conventional bid price variances
    demand_mu           : mean of hourly demand  mu_D     (MWh)
    demand_sigma2       : variance of demand     sigma2_D
    demand_floor        : minimum admissible demand D_under (MWh)
    price_floor         : market bid floor  p_under  ($/MWh)
    price_cap           : market bid cap    p_bar    ($/MWh)
    penalty_rate        : rho, penalty per MWh of wind under-delivery ($/MWh)
    n_replications      : number of Monte Carlo replications N
    n_intervals         : number of trading intervals T (default 24)
    seed                : random seed for reproducibility
    evaluate_wind_perspective      : if False, skip P_W computations
    evaluate_regulator_perspective : if False, skip P_R computations

    Returns
    -------
    SimulationResult with per-interval and daily arrays for both
    perspectives, plus a summary() method for key statistics.

    Raises
    ------
    ValueError : if bid arrays are mis-shaped or out of bounds.
    """
    # --- Validate inputs ---------------------------------------------------
    wind_bid_quantities = np.asarray(wind_bid_quantities, dtype=float)
    wind_bid_prices = np.asarray(wind_bid_prices, dtype=float)

    if wind_bid_quantities.shape != (n_intervals,):
        raise ValueError(
            f"wind_bid_quantities must have shape ({n_intervals},), "
            f"got {wind_bid_quantities.shape}."
        )
    if wind_bid_prices.shape != (n_intervals,):
        raise ValueError(
            f"wind_bid_prices must have shape ({n_intervals},), "
            f"got {wind_bid_prices.shape}."
        )
    if len(conv_capacities) != 3 or len(conv_mu_b) != 3 or len(conv_sigma2_b) != 3:
        raise ValueError("conv_capacities, conv_mu_b, conv_sigma2_b must each have 3 elements.")

    # --- Instantiate agents ------------------------------------------------
    wind_agent = WindProducer(wind_capacity, wind_mu, wind_sigma2)
    solar_agent = SolarProducer(solar_a, solar_b, solar_mu_b, solar_sigma2_b,
                                price_floor, price_cap)
    conv_agents = [
        ConventionalProducer(f"C{k + 1}", conv_capacities[k],
                             conv_mu_b[k], conv_sigma2_b[k],
                             price_floor, price_cap)
        for k in range(3)
    ]
    market_operator = MarketOperator(demand_floor)

    # --- Instantiate perspectives ------------------------------------------
    wind_perspective = WindProducerPerspective(penalty_rate)
    reg_perspective = SystemRegulatorPerspective()

    # --- Validate wind bids against agent constraints ----------------------
    v_W, b_W = wind_agent.submit_bids(wind_bid_quantities, wind_bid_prices)

    # --- Pre-compute solar quantities (deterministic, shared across reps) --
    solar_qty, _ = solar_agent.submit_bids(n_intervals, np.random.default_rng(0))
    solar_qty, _ = solar_agent._cosine_quantities(n_intervals), None  # quantities only

    # --- Pre-allocate result arrays ----------------------------------------
    all_clearing_prices = np.zeros((n_replications, n_intervals))
    all_wind_dispatched = np.zeros((n_replications, n_intervals))
    all_wind_realised = np.zeros((n_replications, n_intervals))
    all_wind_shortfall = np.zeros((n_replications, n_intervals))
    all_wind_interval_revenue = np.zeros((n_replications, n_intervals))
    all_wind_daily_revenue = np.zeros(n_replications)
    all_system_interval_cost = np.zeros((n_replications, n_intervals))
    all_renewable_fraction = np.zeros((n_replications, n_intervals))
    all_system_daily_cost = np.zeros(n_replications)
    all_mean_renewable_frac = np.zeros(n_replications)
    all_total_wind_shortfall = np.zeros(n_replications)

    # --- Monte Carlo loop --------------------------------------------------
    rng = np.random.default_rng(seed)
    demand_sigma = np.sqrt(demand_sigma2)

    for n in range(n_replications):

        # Step 1 — Draw stochastic realisations
        demands = rng.normal(demand_mu, demand_sigma, size=n_intervals)
        demands = np.maximum(demands, demand_floor)
        realised_wind = wind_agent.realise_output(n_intervals, rng)
        solar_prices = rng.normal(solar_agent.mu_b, solar_agent.sigma_b,
                                  size=n_intervals)
        solar_prices = np.clip(solar_prices, price_floor, price_cap)

        conv_qtys = {}
        conv_pxs = {}
        for ca in conv_agents:
            cq, cp = ca.submit_bids(n_intervals, rng)
            conv_qtys[ca.agent_id] = cq
            conv_pxs[ca.agent_id] = cp

        # Step 2 — Assemble bid dictionaries
        bid_quantities = {
            "S": solar_agent._cosine_quantities(n_intervals),
            "W": v_W,
            **conv_qtys,
        }
        bid_prices = {
            "S": solar_prices,
            "W": b_W,
            **conv_pxs,
        }

        # Step 3 — Merit-order clearing
        clearing_prices, dispatched = market_operator.clear_day(
            demands, bid_quantities, bid_prices
        )

        all_clearing_prices[n] = clearing_prices
        all_wind_dispatched[n] = dispatched["W"]
        all_wind_realised[n] = realised_wind

        # Step 4 — Wind-producer perspective (P_W)
        if evaluate_wind_perspective:
            pw = wind_perspective.evaluate(
                clearing_prices, dispatched["W"], realised_wind
            )
            all_wind_shortfall[n] = pw["shortfall"]
            all_wind_interval_revenue[n] = pw["interval_revenue"]
            all_wind_daily_revenue[n] = pw["daily_revenue"]

        # Step 5 — System regulator perspective (P_R)
        if evaluate_regulator_perspective:
            shortfall_for_reg = (
                all_wind_shortfall[n] if evaluate_wind_perspective
                else np.maximum(0.0, dispatched["W"] - realised_wind)
            )
            pr = reg_perspective.evaluate(
                clearing_prices, demands,
                dispatched["W"], dispatched["S"],
                shortfall_for_reg,
            )
            all_system_interval_cost[n] = pr["interval_cost"]
            all_renewable_fraction[n] = pr["renewable_fraction"]
            all_system_daily_cost[n] = pr["daily_cost"]
            all_mean_renewable_frac[n] = pr["mean_renewable_frac"]
            all_total_wind_shortfall[n] = pr["total_shortfall"]

    return SimulationResult(
        clearing_prices=all_clearing_prices,
        wind_dispatched=all_wind_dispatched,
        wind_realised=all_wind_realised,
        wind_shortfall=all_wind_shortfall,
        wind_interval_revenue=all_wind_interval_revenue,
        wind_daily_revenue=all_wind_daily_revenue,
        system_interval_cost=all_system_interval_cost,
        renewable_fraction=all_renewable_fraction,
        system_daily_cost=all_system_daily_cost,
        mean_renewable_frac=all_mean_renewable_frac,
        total_wind_shortfall=all_total_wind_shortfall,
    )


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv

    T = 24
    PENALTY_RATES = [20, 100, 180]
    N_REPLICATIONS = 10

    # --- Table 4 parameter values ------------------------------------------
    WIND_BID_QTY = np.full(T, 300.0)  # b_wt = 300 MWh for all t
    WIND_BID_PX = np.full(T, 50.0)  # p_wt =  50 $/MWh for all t

    COMMON_PARAMS = dict(
        wind_bid_quantities=WIND_BID_QTY,
        wind_bid_prices=WIND_BID_PX,
        wind_capacity=3000.0,  # >= bid quantity and mu_G
        wind_mu=275.0,  # mu_G     [MWh]
        wind_sigma2=2500.0,  # sigma_G=50 => sigma2=2500
        solar_a=0.0,  # a        [MWh]
        solar_b=-400.0,  # b        [MWh]
        solar_mu_b=35.0,  # mu_ps    [$/MWh]
        solar_sigma2_b=16.0,  # sigma_ps=4 => sigma2=16
        conv_capacities=[300.0, 250.0, 1000.0],
        conv_mu_b=[45.0, 50.0, 60.0],
        conv_sigma2_b=[4.0, 4.0, 4.0],  # sigma_pi=2 => sigma2=4
        demand_mu=800.0,  # mu_D     [MWh]
        demand_sigma2=400.0,  # sigma_D=20 => sigma2=400
        demand_floor=1.0,
        price_floor=0.0,
        price_cap=500.0,
        n_replications=N_REPLICATIONS,
        n_intervals=T,
        seed=None,
        evaluate_regulator_perspective=False,
    )

    # --- Run simulation for each penalty rate and collect results ----------
    rows = []  # each row: (q_u, run, t, wind_dispatch, wind_revenue)

    for q_u in PENALTY_RATES:
        result = run_simulation(**COMMON_PARAMS, penalty_rate=float(q_u))

        for run_idx in range(N_REPLICATIONS):
            for t_idx in range(T):
                rows.append((
                    q_u,
                    run_idx + 1,
                    t_idx + 1,
                    result.wind_dispatched[run_idx, t_idx],
                    result.wind_interval_revenue[run_idx, t_idx],
                ))

    # --- Save to CSV -------------------------------------------------------
    output_path = "market3_results.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q_u", "run", "t", "wind_dispatch_MWh", "wind_revenue_USD"])
        writer.writerows(rows)

    print(f"Results saved to '{output_path}'.")
    print(f"  Rows: {len(rows)}  ({len(PENALTY_RATES)} penalty rates"
          f" x {N_REPLICATIONS} runs x {T} intervals)")
