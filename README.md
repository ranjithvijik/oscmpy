# OSCM Simulator for Streamlit

An interactive Python and Streamlit simulator for Operations and Supply Chain Management (OSCM). The app helps learners practice quantitative models, quality tools, process analysis, supply chain decisions, forecasting, inventory planning, MRP, scheduling, and exam review in a guided browser interface.

Live app:

```text
https://oscmsim.streamlit.app/
```

Repository:

```text
https://github.com/ranjithvijik/oscmpy
```

The Streamlit version is the Python application companion to the static `oscm` simulator. It keeps the learning experience interactive, but uses Python libraries for calculations, charts, tables, probability models, and Streamlit-native widgets.

## What This Project Contains

- `app.py`: the full Streamlit application, including theme styling, module registry, formulas, calculators, charts, navigation, and learning content.
- `requirements.txt`: Python dependencies used by Streamlit Community Cloud and local installs.
- `README.md`: project documentation, setup, deployment, module catalog, and QA notes.

This repository is intentionally small. The app deploys directly from `app.py` and `requirements.txt`; there is no separate backend service, database, build step, or frontend package manager.

## Quick Start

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app locally:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

If port `8501` is busy, use another port:

```bash
streamlit run app.py --server.port 8502
```

## Deployed App

The production app is deployed on Streamlit Community Cloud:

```text
https://oscmsim.streamlit.app/
```

The deployment reads:

- `app.py` as the app entry point.
- `requirements.txt` for Python dependencies.
- The `main` branch of this repository.

Streamlit Community Cloud automatically rebuilds the app when changes are pushed to the connected GitHub repository.

## Simulator Overview

The simulator is organized as a single Streamlit application with chapter-grouped navigation in the sidebar. Each topic is implemented as a Python function, and the module registry maps those functions into searchable, bookmarkable navigation entries.

Common module patterns:

- A module header with chapter, title, and topic summary.
- Theory sections that explain the operations concept.
- Formula cards rendered with Streamlit/KaTeX mathematical notation.
- Interactive widgets for parameters, rates, costs, probabilities, and constraints.
- Computed metric cards for immediate feedback.
- Plotly charts for visual interpretation.
- DataFrames and styled tables for tabular models.
- Practice sections and guided worked examples.
- Sidebar search, bookmarks, recent modules, progress tracking, and theme toggle.

The app covers material from operations strategy, project management, capacity planning, manufacturing, service systems, queuing, process analysis, Six Sigma, quality control, lean, logistics, sourcing, forecasting, aggregate planning, inventory, MRP, scheduling, and exam review.

## Technology Stack

- Python 3.12 compatible application code.
- Streamlit for the web application shell and widgets.
- Pandas for tabular calculations and display.
- NumPy for numerical operations.
- SciPy for statistics, distributions, optimization, and probability calculations.
- Plotly for interactive charts and visualizations.
- Matplotlib and Altair available for additional chart support.
- OpenPyXL and XlsxWriter for spreadsheet-compatible data workflows.

Dependencies are pinned by range in `requirements.txt` to stay compatible with Streamlit Community Cloud while allowing minor patch updates.

## Application Architecture

`app.py` contains the full application in a deliberately linear structure:

1. Imports and compatibility patches.
2. Streamlit page configuration.
3. Session state initialization.
4. Theme palette and dynamic CSS.
5. Plotly theme helpers.
6. Reusable UI components.
7. Shared OSCM calculation utilities.
8. Individual simulator modules.
9. Module registry and chapter navigation.
10. Sidebar search, bookmarks, recent modules, and progress tracking.
11. Main render flow.

The single-file structure keeps deployment simple for Streamlit Community Cloud. The reusable helpers reduce duplication inside the module implementations and keep visual styling consistent.

## User Experience Features

- Light and dark mode toggle.
- Searchable module navigation.
- Chapter-grouped sidebar.
- Module bookmarks.
- Recently visited modules.
- Progress tracking across visited modules.
- Practice-problem solved counters and streak tracking.
- Responsive layout for desktop, tablet, and mobile browser sizes.
- Improved light-mode contrast for readable text and buttons.
- Polished equation display with scroll-safe formula panels.
- Streamlit HTML rendering compatibility layer to prevent HTML fragments from showing as raw text.

## Module Catalog

### 1. Supply Chain Risk Assessment, Chapter 1

The supply chain risk module introduces OSCM strategy, risk exposure, and mitigation planning. Learners evaluate risks using probability-impact thinking and expected monetary value.

Core concepts:

- Supply, demand, process, and external risks.
- Probability-impact scoring.
- Expected monetary value.
- Risk mitigation and resilience.
- Triple bottom line awareness.

### 2. PERT Network, Chapter 4

The PERT module teaches probabilistic project analysis. It covers optimistic, most likely, and pessimistic activity estimates, expected time, activity variance, path variance, and completion probability.

Core concepts:

- PERT expected activity time.
- Activity standard deviation and variance.
- Critical path duration.
- Path variance aggregation.
- Z-score probability for deadlines.

### 3. Project Crashing, Chapter 4

The project crashing module explains how to reduce project duration by spending additional resources. Learners compare normal time, crash time, normal cost, crash cost, and cost per time unit saved.

Core concepts:

- Normal versus crash duration.
- Normal versus crash cost.
- Crash cost slope.
- Critical-path compression.
- Direct and indirect project cost tradeoffs.

### 4. Break-Even Analysis, Chapter 5

The break-even module covers cost-volume-profit analysis for capacity and process decisions. It helps learners understand how fixed cost, variable cost, price, and target profit determine required volume.

Core concepts:

- Break-even units.
- Break-even revenue.
- Contribution margin.
- Contribution margin ratio.
- Target profit volume.
- Indifference points between alternatives.

### 5. Decision Trees, Chapter 5

The decision tree module supports structured decisions under uncertainty. It uses probabilities and payoffs to compare alternatives using expected monetary value.

Core concepts:

- Decision nodes and chance nodes.
- Expected monetary value.
- Rollback analysis.
- Expected value of perfect information.
- Risk-aware decision selection.

### 6. Learning Curves, Chapter 6

The learning curves module shows how time or cost changes as cumulative production increases. It supports learning-rate interpretation and log-linear production estimates.

Core concepts:

- Unit time learning model.
- Learning exponent.
- Doubling rule.
- Cumulative time and average time.
- Cost and labor planning implications.

### 7. Customer Order Decoupling Point, Chapter 7

The decoupling point module explains where inventory separates forecast-driven operations from customer-order-driven operations.

Core concepts:

- Make-to-stock.
- Assemble-to-order.
- Make-to-order.
- Engineer-to-order.
- Forecast risk, customization, lead time, and inventory positioning.

### 8. Line Balancing, Chapter 8

The line balancing module covers assigning tasks to workstations while meeting cycle-time constraints. It helps learners evaluate efficiency, idle time, and bottlenecks.

Core concepts:

- Cycle time.
- Minimum workstations.
- Line efficiency.
- Balance delay.
- Task assignment and workstation utilization.

### 9. Service Design, Chapter 9

The service design module connects process choices to customer experience and service delivery. It covers customer contact, visibility, service blueprint thinking, and failure prevention.

Core concepts:

- Service process design.
- Customer contact.
- Service blueprinting.
- Front-stage and back-stage work.
- Service failure points.

### 10. Poka-yoke Database, Chapter 9

The poka-yoke module is a mistake-proofing reference and design aid. It helps learners connect human error, process design, detection, and prevention.

Core concepts:

- Error prevention.
- Detection devices.
- Control devices.
- Human factors.
- Service and manufacturing mistake-proofing.

### 11. Queuing Theory, Chapter 10

The queuing module models waiting line performance. Learners explore how arrival rates, service rates, and utilization affect waiting time and congestion.

Core concepts:

- Arrival rate.
- Service rate.
- Utilization.
- Average number in system.
- Average waiting time.
- Capacity and waiting-cost tradeoffs.

### 12. Distributions, Chapter 10

The distributions module supports probability models commonly used in OSCM, including arrivals, defects, service times, and demand uncertainty.

Core concepts:

- Normal distribution.
- Poisson distribution.
- Binomial distribution.
- Exponential distribution.
- Probability interpretation for operations.

### 13. Little's Law, Chapter 11

The Little's Law module explains the relationship among inventory, throughput, and flow time.

Core concepts:

- `L = lambda W`.
- Work-in-process.
- Throughput rate.
- Flow time.
- Process diagnostics and bottleneck awareness.

### 14. DPMO and DMAIC, Chapter 12

The DPMO module introduces Six Sigma performance measurement and structured process improvement.

Core concepts:

- Defects per million opportunities.
- Defects, units, and opportunities.
- Sigma-level interpretation.
- DMAIC improvement cycle.
- Quality performance measurement.

### 15. FMEA Risk, Chapter 12

The FMEA module teaches failure mode and effects analysis. Learners score risks using severity, occurrence, and detection to prioritize action.

Core concepts:

- Failure modes.
- Failure effects.
- Severity, occurrence, and detection.
- Risk priority number.
- Recommended controls and mitigation.

### 16. p and c Charts, Chapter 13

The statistical quality control module focuses on attribute control charts for monitoring process stability.

Core concepts:

- p-charts for fraction defective.
- c-charts for defect counts.
- Center lines.
- Upper and lower control limits.
- Out-of-control interpretation.

### 17. Process Capability, Chapter 13

The process capability module evaluates whether a stable process can meet specification limits.

Core concepts:

- Specification limits.
- Process mean and standard deviation.
- Cp.
- Cpk.
- Centering versus spread.

### 18. Acceptance Sampling, Chapter 13

The acceptance sampling module covers lot inspection decisions and sampling risk.

Core concepts:

- Sample size.
- Acceptance number.
- Lot acceptance and rejection.
- Operating characteristic logic.
- Producer and consumer risk.

### 19. Pareto Analysis, Chapter 13

The Pareto module helps learners prioritize defects, causes, or problems by contribution.

Core concepts:

- Frequency ranking.
- Cumulative contribution.
- The vital few versus trivial many.
- Defect prioritization.
- Quality improvement focus.

### 20. Fishbone Diagram, Chapter 13

The fishbone module supports cause-and-effect analysis for root-cause investigation.

Core concepts:

- Ishikawa diagrams.
- Cause categories.
- Structured brainstorming.
- Root-cause hypotheses.
- Quality diagnostics.

### 21. SQC Practice, Chapter 13

The SQC practice module provides additional statistical quality control review and applied questions.

Core concepts:

- Control chart interpretation.
- Quality metrics.
- Capability review.
- Defect calculations.
- Exam-style practice.

### 22. Lean Supply Chains, Chapter 14

The lean module covers waste reduction, flow improvement, pull systems, and continuous improvement in supply chains.

Core concepts:

- Waste identification.
- Pull and flow.
- Kanban and JIT thinking.
- Continuous improvement.
- Supplier and process coordination.

### 23. Centroid Method, Chapter 15

The centroid method module covers a location planning technique based on weighted coordinates.

Core concepts:

- Weighted x and y coordinates.
- Demand-weighted location.
- Facility siting.
- Logistics distance tradeoffs.
- Visual location interpretation.

### 24. Factor Rating, Chapter 15

The factor rating module supports facility or supplier location comparison using weighted criteria.

Core concepts:

- Decision criteria.
- Factor weights.
- Alternative ratings.
- Weighted scores.
- Qualitative and quantitative location tradeoffs.

### 25. Transportation Method, Chapter 15

The transportation module introduces allocation decisions across origins and destinations.

Core concepts:

- Supply constraints.
- Demand constraints.
- Shipping cost matrix.
- Allocation decisions.
- Logistics network planning.

### 26. Global Sourcing, Chapter 16

The global sourcing module covers international sourcing decisions and supplier tradeoffs.

Core concepts:

- Total landed cost.
- Global supplier risk.
- Lead time.
- Flexibility.
- Strategic sourcing decisions.

### 27. Enhanced Forecast, Chapter 18

The forecasting module covers demand forecasting and forecast accuracy measurement.

Core concepts:

- Moving averages.
- Exponential smoothing.
- Trend logic.
- Forecast error.
- MAD, MSE, RMSE, MAPE, and bias.

### 28. Regression+, Chapter 18

The regression module supports trend-line forecasting and least-squares estimation.

Core concepts:

- Linear regression.
- Slope and intercept.
- R-squared interpretation.
- Residual awareness.
- Forecasting with explanatory variables.

### 29. Aggregate Planning, Chapter 19

The aggregate planning module supports medium-term supply and demand planning.

Core concepts:

- Sales and operations planning.
- Chase strategy.
- Level strategy.
- Workforce and capacity decisions.
- Inventory and backlog tradeoffs.

### 30. EOQ Model, Chapter 20

The EOQ module teaches economic order quantity and inventory cost tradeoffs.

Core concepts:

- Economic order quantity.
- Ordering cost.
- Holding cost.
- Total annual cost.
- Reorder cycle logic.

### 31. Safety Stock, Chapter 20

The safety stock module explains inventory buffers for demand and lead-time uncertainty.

Core concepts:

- Service level.
- Demand variability.
- Lead-time uncertainty.
- Safety stock.
- Reorder point.

### 32. Newsvendor Model, Chapter 20

The newsvendor module covers single-period inventory decisions under uncertain demand.

Core concepts:

- Overage cost.
- Underage cost.
- Critical fractile.
- Demand uncertainty.
- Single-period order quantity.

### 33. MRP Matrix, Chapter 21

The MRP module covers material requirements planning in a time-phased matrix.

Core concepts:

- Gross requirements.
- Scheduled receipts.
- Projected on-hand inventory.
- Net requirements.
- Planned order receipts and releases.
- Lead-time offset.

### 34. MRP Lot Sizing, Chapter 21

The MRP lot sizing module compares order-sizing approaches used in dependent demand planning.

Core concepts:

- Lot-for-lot.
- Fixed order quantity.
- Periodic order quantity.
- EOQ-style lot sizing.
- Setup and inventory tradeoffs.

### 35. Job Scheduling, Chapter 22

The scheduling module covers job sequencing and dispatching rules.

Core concepts:

- Shortest processing time.
- Earliest due date.
- First come, first served.
- Makespan.
- Flow time and lateness.
- Schedule performance comparison.

### 36. Practice Problems, Exam Prep

The practice module provides broad exam-style review across OSCM topics.

Core concepts:

- Mixed-topic practice.
- Formula application.
- Self-check review.
- Quantitative problem solving.
- Exam readiness.

## Navigation and Learning Workflow

Recommended learning flow:

1. Start with `SC Risk Assessment` for strategy context.
2. Work through project and capacity modules: `PERT Network`, `Project Crashing`, `Break-Even Analysis`, and `Decision Trees`.
3. Continue into process and quality modules: `Line Balancing`, `Queuing Theory`, `Little's Law`, `DPMO & DMAIC`, `FMEA Risk`, and control charts.
4. Use logistics, sourcing, forecasting, inventory, MRP, and scheduling modules for later-course topics.
5. Finish with `SQC Practice` and `Practice Problems` for exam review.

The sidebar search can jump directly to topics such as `EOQ`, `PERT`, `Cpk`, `queue`, `forecast`, `MRP`, or `safety stock`.

## Local Verification and QA

This repository does not require a JavaScript build or browser bundle. The most important baseline check is Python syntax compilation:

```bash
python3 -m py_compile app.py
```

Run the app locally for manual smoke testing:

```bash
streamlit run app.py
```

Recommended smoke test checklist:

- The app launches without a Streamlit exception.
- The sidebar renders the module list.
- Light and dark mode toggle correctly.
- Module search returns expected results.
- A representative module such as `PERT Network` opens.
- Equations render as mathematical notation, not raw HTML or literal LaTeX.
- Plotly charts render where expected.
- Tables and input widgets are readable in light mode.
- Mobile-width browser layouts keep equations scrollable instead of clipped.

Useful one-command local check:

```bash
python3 -m py_compile app.py && streamlit run app.py
```

## Deployment

The app is designed for Streamlit Community Cloud.

Deployment settings:

- Repository: `ranjithvijik/oscmpy`
- Branch: `main`
- Main file path: `app.py`
- Python dependencies: `requirements.txt`
- Production URL: `https://oscmsim.streamlit.app/`

When deploying from Streamlit Community Cloud:

1. Connect the GitHub repository.
2. Select branch `main`.
3. Set the app entry point to `app.py`.
4. Allow Streamlit to install `requirements.txt`.
5. Deploy and verify the production URL.

After each push to `main`, Streamlit Community Cloud should rebuild automatically.

## Troubleshooting

### Raw HTML appears on the page

The app includes a compatibility layer that routes trusted HTML fragments through `st.html()` where available, with a fallback for older Streamlit versions. If raw HTML appears after a Streamlit upgrade, inspect the `_markdown_with_safe_html()` and `render_html_fragment()` helpers in `app.py`.

### Equations appear clipped

Equation blocks are styled to allow horizontal scrolling on smaller screens. If a formula is clipped, check whether the rendered element contains `.katex-display` and whether the CSS block for `div[data-testid="stMarkdownContainer"]:has(.katex-display)` is active.

### Text is hard to read in light mode

The theme palette defines high-contrast light-mode text, accent, button, and sidebar colors in `_get_palette_cached()`. If Streamlit changes its generated markup, new selectors may need explicit color overrides.

### Dependency installation fails

Install dependencies in a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Port already in use

Run Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

## Development Notes

- Keep the app deployable with only `app.py` and `requirements.txt`.
- Prefer shared UI helpers over one-off HTML/CSS blocks inside modules.
- Use `_get_palette()` and the theme CSS system for colors instead of hard-coded values.
- Use `display_formula_card()` and `display_equation()` for formulas so equation display remains consistent.
- Keep mathematical calculations in helper functions when reused by more than one module.
- Run `python3 -m py_compile app.py` before committing.
- Avoid committing generated cache directories such as `__pycache__/`.

## Relationship to the Static OSCM Simulator

The static `oscm` repository is a browser-only version built from HTML, CSS, JavaScript, and Playwright QA. This `oscmpy` repository is the Streamlit/Python version.

Key differences:

- `oscm` is static and can be served by any static file host.
- `oscmpy` runs as a Streamlit app and executes Python calculations server-side.
- `oscm` uses JavaScript and MathJax for browser behavior and equations.
- `oscmpy` uses Python, Streamlit widgets, SciPy, Pandas, Plotly, and KaTeX/Streamlit equation rendering.
- `oscm` has a Playwright QA system.
- `oscmpy` currently uses Python compile checks plus local/browser smoke testing.

## License

No license file is currently included in this repository. Add a `LICENSE` file before redistributing or reusing the project outside its current intended educational context.
