Commodity Offtake Portfolio Model — Multi-Agent Q-Learning
A multi-agent reinforcement learning model for commodity offtake portfolio
optimisation. Four country-agents learn adaptive contracting, spot-market,
and storage strategies through Q-learning in a shared market with Markov
supply disruptions and demand-elasticity price clearing.

Each supplier has a fixed cost (incurred per unit of capacity maintained) and a variable cost (incurred per unit actually produced). Each period, suppliers observe their contracted volumes and spot sales from the previous period and decide whether to maintain, expand, or shrink capacity. If countries build large storage buffers and reduce offtake, suppliers rationally shed capacity — which then makes future disruptions worse because there's less slack in the system. This creates a feedback loop: storage policy affects supplier capacity decisions, which affects the risk environment the agents are learning in.
Domestic production gets the same treatment — countries must pay fixed costs to maintain domestic capacity whether or not they use it. This makes the "heavy-domestic" strategy more expensive during calm periods but still valuable as insurance.

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
