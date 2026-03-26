# Commodity Portfolio Q-Learning Model

## What This Does

This model helps a country (or group of countries) figure out the **cheapest way to buy a commodity** — like oil, gas, or fertiliser — when supply can be disrupted at any time.

It does this by simulating thousands of scenarios and letting a learning algorithm discover which procurement strategies work best under uncertainty. The key question it answers is: **given the current market conditions, what mix of contracted suppliers, domestic production, spot purchases, and strategic storage minimises long-run costs?**

---

## The Core Problem

Imagine you're responsible for securing a nation's fuel supply. You have several options:

- **Contract with overseas suppliers** — cheaper per unit, but they can go offline (wars, sanctions, disasters)
- **Produce domestically** — more expensive, but reliable
- **Buy on the spot market** — prices spike when everyone else is also scrambling for supply
- **Build and draw from storage** — acts as insurance, but costs money to maintain

No single option is best all the time. The optimal mix depends on what's happening in the market right now *and* how the market will evolve in response to your decisions.

---

## How the Model Works

### 1. The Simulated World

Each simulation run plays out over a number of time periods (default: 36 months). In each period:

- **Suppliers can go offline or come back online.** Each supplier has a probability of disruption (`p_d`) and a probability of recovery (`p_r`). This is modelled as a simple two-state Markov chain — disrupted or operating — that flips randomly each period.

- **Spot prices are set by supply and demand.** When multiple countries are chasing scarce supply, the model runs a price-clearing mechanism that raises the spot price until demand meets available supply. Countries with more elastic demand (i.e. more price-sensitive) reduce their purchases more.

- **Suppliers and domestic producers adjust capacity over time.** This is the key feedback loop. If a supplier's output is being underutilised (because buyers are stockpiling or sourcing elsewhere), it gradually cuts capacity. If utilisation is high, it expands. This means that over-reliance on storage or domestic production can cause international suppliers to shrink — making the market tighter and crises worse when they do hit.

- **Domestic production works similarly.** It has both a fixed cost (for maintaining capacity) and a variable cost (for actual output). Capacity adjusts based on how much it's being used.

### 2. The Decision Agent (Q-Learning)

Each country has its own **Q-learning agent** — a simple reinforcement learning algorithm that learns from trial and error.

**What the agent sees (its "state"):**

The agent compresses the complex world into four discretised signals:

| Signal | What it captures | Bins |
|---|---|---|
| **Storage level** | How full are the reserves? | 5 levels (empty → full) |
| **Spot price** | Is the market cheap or expensive? | 4 levels (low → crisis) |
| **Disruption severity** | How many suppliers are offline? | 4 levels (none → widespread) |
| **Supply tightness** | Is total available capacity close to total demand? | 3 levels (surplus → deficit) |

These combine into 5 × 4 × 4 × 3 = **240 possible states**.

**What the agent chooses (its "action"):**

Each period, the agent picks one of **18 actions** — a combination of:

- **An allocation profile** (6 options): how to split purchases across contracted suppliers, spot market, and domestic production. Profiles range from "diversified" (spread evenly) to "heavy-spot" (buy mostly on the open market) to "min-cost" (favour the cheapest suppliers).

- **A storage decision** (3 options): store more, hold steady, or draw down reserves.

**How it learns:**

The agent maintains a table of 240 states × 18 actions = 4,320 entries. Each entry stores the agent's estimate of how good that action is in that state. After each period, it updates its estimate based on the cost it actually incurred (the "reward," which is the negative of cost-per-unit-of-demand). Over thousands of training episodes, the agent converges on a policy that minimises average procurement cost across a wide range of disruption scenarios.

The learning process uses:
- **Exploration vs. exploitation**: Early on, the agent tries random actions to explore the space. Over time, it increasingly follows its best-known strategy. The exploration rate decays from 100% to 3%.
- **Discounting**: The agent values near-term cost savings slightly more than far-future ones (discount factor 0.95), which prevents it from hoarding indefinitely.

### 3. Evaluation

After training, the model runs two sets of Monte Carlo simulations (default: 300 each):

- **Q-Learning policy**: the trained agent picks its best action in each state
- **Naïve baseline**: a fixed strategy (action #1 — "diversified allocation, hold storage") in every period

This lets you compare how much value the learned strategy adds over a simple passive approach.

---

## What Comes Out

The API returns a JSON object with three top-level sections:

### Training Curves (`tr`)
- **Spot price trajectory** over training (sampled every 30 episodes)
- **Average cost per country** over training — you should see these decline as the agents learn

### Naïve Baseline Results (`nv`) and Q-Learning Results (`ql`)

Both share the same structure. For each country, you get:

- **Cost time series** — average and 95th percentile cost per period
- **Storage time series** — how reserves evolve over the simulation horizon
- **Domestic capacity time series** — how domestic production capacity rises or falls
- **Supplier capacity time series** — total international supplier capacity over time
- **Spot price statistics** — mean, standard deviation, and 95th percentile
- **Supply volume breakdown** — how much comes from each source (each contracted supplier, spot, domestic, storage draws), both in absolute terms and as a share of total
- **Action frequencies** — how often each allocation profile and storage decision was chosen
- **Blended weights** — the effective portfolio weights implied by the agent's action mix
- **Cost breakdown** — per-unit costs split into contract variable costs, supplier fixed costs, spot purchases, domestic variable costs, domestic fixed costs, and storage holding costs

---

## Key Dynamics to Watch For

**The capacity feedback loop.** If the learned policy relies heavily on storage and domestic production, international suppliers lose customers, cut capacity, and the market becomes structurally tighter. This can create a vicious cycle where the "safe" strategy actually makes crises more severe. The supply tightness state dimension was added specifically to help the agent learn to account for this.

**Fixed vs. variable costs.** Suppliers charge a fixed cost proportional to their maintained capacity (shared across all their customers) plus a variable cost per unit actually delivered. This means that even idle contract relationships have a cost — the agent must weigh the insurance value of maintaining supplier relationships against the carrying cost.

**Spot price spikes.** When supply is disrupted and storage is low, the spot price can rise to 10–20× the base price. The agent learns to avoid these situations through a mix of diversification and proactive storage management.

---

## Running the API

The model is served as a FastAPI application.

```bash
pip install fastapi uvicorn numpy pydantic
uvicorn main:app --reload
```

### Endpoints

- `GET /api/health` — health check
- `POST /api/run` — run the full training and evaluation pipeline

### Example Request

```json
{
  "suppliers": [
    {
      "name": "Middle East",
      "max_capacity": 120,
      "init_capacity": 100,
      "fixed_cost": 2.0,
      "variable_cost": 35.0,
      "p_d": 0.05,
      "p_r": 0.6,
      "capacity_adjust_rate": 0.08
    },
    {
      "name": "West Africa",
      "max_capacity": 60,
      "init_capacity": 50,
      "fixed_cost": 3.0,
      "variable_cost": 45.0,
      "p_d": 0.10,
      "p_r": 0.4
    }
  ],
  "countries": [
    {
      "name": "Country A",
      "demand": 80,
      "elasticity": -0.3,
      "dom_variable_cost": 55.0,
      "dom_fixed_cost": 5.0,
      "dom_max_capacity": 30,
      "dom_init_capacity": 20,
      "max_dom": 0.4,
      "stor_max": 50,
      "stor_cost": 1.5,
      "stor_init": 10
    }
  ],
  "base_spot": 55.0,
  "train_episodes": 2000,
  "eval_sims": 300,
  "ep_length": 36
}
```

### Parameter Reference

**Supplier parameters:**

| Parameter | Description |
|---|---|
| `max_capacity` | Maximum possible production capacity |
| `init_capacity` | Starting capacity at the beginning of each episode |
| `fixed_cost` | Cost per unit of maintained capacity (paid regardless of output) |
| `variable_cost` | Cost per unit actually produced and delivered |
| `p_d` | Probability of disruption each period (when currently operating) |
| `p_r` | Probability of recovery each period (when currently disrupted) |
| `capacity_adjust_rate` | How fast capacity responds to utilisation changes (default 0.08) |

**Country parameters:**

| Parameter | Description |
|---|---|
| `demand` | Units of commodity needed per period |
| `elasticity` | Price elasticity of spot demand (negative; e.g. -0.3 means a 10% price rise cuts spot demand by 3%) |
| `dom_variable_cost` | Variable cost per unit of domestic production |
| `dom_fixed_cost` | Fixed cost per unit of maintained domestic capacity |
| `dom_max_capacity` | Maximum domestic production capacity |
| `dom_init_capacity` | Starting domestic capacity |
| `max_dom` | Maximum share of demand that can be met domestically (0 to 1) |
| `stor_max` | Maximum storage capacity (0 = no storage) |
| `stor_cost` | Holding cost per unit stored per period |
| `stor_init` | Starting storage level |

**Simulation parameters:**

| Parameter | Description | Default |
|---|---|---|
| `base_spot` | Reference spot price | 55.0 |
| `train_episodes` | Number of Q-learning training episodes (capped at 5,000) | 2,000 |
| `eval_sims` | Number of Monte Carlo evaluation simulations (capped at 1,000) | 300 |
| `ep_length` | Periods per episode (capped at 60) | 36 |

---

## Limitations

- **Tabular Q-learning** scales poorly. With 240 states and 18 actions per country, this works well for 1–3 countries. For larger problems, a neural network–based approach (e.g. DQN or PPO) would be needed.
- **Countries learn independently.** Each country's agent optimises for itself without considering how its actions affect other countries' costs. There is no game-theoretic equilibrium — just parallel single-agent optimisation. This may be reasonable for markets without concentration but is unrealistic for concentrated share where participants infer about strategic behavior (OPEC, many critical minerals or other commodities).
- **Lack of Strategic Behavior:** Outages and supply shocks are purely stochastic, but some shocks like recent Chinese rare earth export controls are strategic.
- **Strategic Behavior and Adversarial Behavior:** Some countries may see "pain" for other countries as a reward. Obviously this is an issue for modelling strategic competition. This could lead to more adversarial dumping, efforts to take market share and spike markets and the like. A potential extension. 
- **Demand is fixed.** Countries always need the same quantity each period. There is no demand seasonality or growth.
- **No transport costs or logistics.** Suppliers deliver costlessly. In practice, freight rates and shipping routes matter significantly. While there is a strong optimization based literature here from the likes of Jun Ukita Shepard and Lincoln Pratson and Gosens / Turnbull / Jotzo this is not implemented here. 
- **Discrete action space.** The six allocation profiles are hand-designed. A continuous action space would allow finer-grained portfolio tuning but would also increase computational complexity.

Architecture
```
frontend/          React + Recharts dashboard (Vite)
  src/App.jsx      Parameter editing + 8-tab results visualisation
api/               FastAPI + numpy backend
  main.py          Q-learning training + Monte Carlo evaluation
```
The frontend sends parameters via `POST /api/run` and receives the full
results JSON. The Python backend runs 3,000 training episodes and 400
evaluation simulations — much faster and more scalable than in-browser.
Local Development
Option A: Direct (recommended)
```bash
# Terminal 1 — API
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```
Open http://localhost:5173. The Vite dev server proxies `/api` to port 8000.
Option B: Docker Compose
```bash
docker compose up
```
Frontend at http://localhost:5173, API at http://localhost:8000.
Deploy to Vercel
Push to GitHub, then in the Vercel dashboard:
Import the repo
Vercel auto-detects `vercel.json` — the frontend builds as static,
the API deploys as a Python serverless function
Done — both served from the same domain
Note: Vercel serverless has a 60s timeout (configured in vercel.json).
With default parameters (3,000 episodes, 400 sims), the model runs in
~15–25 seconds on Vercel's infrastructure.
Deploy to Fly.io / Railway (for heavier workloads)
If you need longer training (5,000+ episodes) or more simulations:
```bash
# Deploy API to Fly.io
cd api
fly launch
fly deploy

# Build frontend with remote API URL
cd frontend
VITE_API_URL=https://your-app.fly.dev npm run build

# Deploy frontend/dist to Vercel, Netlify, or any static host
```
Deploy to AWS / GCP
For production at scale:
API: Container on ECS/Cloud Run, or Lambda behind API Gateway
Frontend: S3 + CloudFront (static files)
Set `VITE_API_URL` at build time to point to the API endpoint
Model Details
State space: 80 states per agent (5 storage bins × 4 price regimes × 4 disruption levels)
Action space: 18 actions (6 allocation profiles × 3 storage actions)
Training: Tabular Q-learning, lr=0.12, γ=0.95, ε: 1.0 → 0.03
Market coupling: Bisection clearing where all agents' spot demand competes
Outputs: Supply mix, strategy frequencies, blended allocation weights,
cost/storage/spot time series with confidence bands
