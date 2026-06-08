# LLM-assisted Model Formalisation and Implementation in Agent-Based Modelling

Code and data accompanying the paper *"LLM-assisted Model Formalisation and Implementation in Agent-Based Modelling"* (Z. Pei, N. Lipovetzky, A. M. Rojas-Arevalo, F. J. de Haan, E. A. Moallemi), to appear in the proceedings of the Social Simulation Conference (SSC) 2026.

The full communication log between the LLM and the researchers is available at:
<https://claude.ai/share/b5d1cfc3-b054-4c3c-832f-6030cdbf8cb5>

## Model and experiment details

The tables below specify the tests used to verify the implementation, the target components of the electricity market case study, and the parameter settings used to generate the reported time series. They reproduce the appendix tables of the paper.

### Verification tests

Table 3: Tests used for verification in the Python implementation of the electricity market problem.

| Test | Details |
|---|---|
| Unit tests | They verified that the revenue and penalty calculations for the wind-power producer were correct and that total dispatched supply was equal to the market demand. |
| Property-based test | It verified the correctness of the implementation of the merit-order market-clearing mechanism. |
| Scenario test | It verified whether the program's performance was consistent with the target context in a specific scenario (described in Table 5). |

### Target components of the electricity market problem

Table 4: Target components of the electricity market problem (Section 4).

| Type | Parameters | Description |
|---|---|---|
| Agents | $C_1, C_2, C_3, W, S, O$ | Market participants. |
| Control variables | $(b_{wt}, p_{wt})$ | Bid quantity and price of the wind-power producer $w$ at interval $t$. |
| Clearing process | $h_t(\cdot)$ | Merit-order market-clearing mechanism and dispatch schedule. |
| Objective | $\Pi$ | Maximisation of the wind-power producer's expected daily revenue. |
| Deterministic parameters | $T$; $(\mu_D, \sigma_D)$; $b_i$; $(a,b)$; $(\mu_G, \sigma_G)$; $q_u$; $b_{st}$ | Simulation time scale; mean and standard deviation of market demand; bid quantity of conventional producer $i$; solar output profile parameters; mean and standard deviation of wind-power production; shortfall penalty coefficient; and bid quantity of solar-power producer $s$ at interval $t$. |
| Uncertainties | $(\mu_{pi}, \sigma_{pi})$; $(\mu_{ps}, \sigma_{ps})$ | Mean and standard deviation of bid prices for conventional producer $i$; mean and standard deviation of bid prices for solar-power producer $s$. |
| Stochastic variables | $D_t$; $p_{it}$; $p_{st}$; $G_t$ | Market demand; bid price of conventional producer $i$; bid price of solar-power producer $s$; and wind-power production at interval $t$. |

### Settings used to generate the time series

Table 5: Settings used to generate the time series in the electricity market problem, following the electricity market case in (Pei et al., 2026) with different parameter values.

| Parameter | Description | Value |
|---|---|---|
| $\mu_{D}$ | Mean of market demand | 800 (MWh) |
| $\sigma_{D}$ | Standard deviation of market demand | 20 (MWh) |
| $b_i$ | Bid quantity of conventional producer $i$ | [300, 250, 1000] (MWh) |
| $\mu_{pi}$ | Mean of bid price of conventional producer $i$ | [45, 50, 60] (\$/MWh) |
| $\sigma_{pi}$ | Standard deviation of bid price of conventional producer $i$ | [2, 2, 2] (\$/MWh) |
| $(a,b)$ | Solar output profile parameters | (0, -400) (MWh) |
| $\mu_{ps}$ | Mean of bid price of the solar-power producer $s$ | 35 (\$/MWh) |
| $\sigma_{ps}$ | Standard deviation of bid price of the solar-power producer $s$ | 4 (\$/MWh) |
| $\mu_G$ | Mean of wind-power production | 275 (MWh) |
| $\sigma_G$ | Standard deviation of wind-power production | 50 (MWh) |
| $(b_{wt}, p_{wt})$ | Bid quantity and price of the wind-power producer $w$ at interval $t$ | (300, 50) (MWh, \$/MWh) |
| $q_{u}$ | Shortfall penalty coefficient | [20, 100, 180] (\$/MWh) |
