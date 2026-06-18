# BNB Hackathon — Index

This is the map of content (MOC) for my entry into the [[BNB Hack - AI Trading Agent Edition]].  It links to every core area of development and serves as the entry point for navigation and for agent workflow orienting itself at the start of a session.

## Timeline

- Submissions must be registered before live trading begins on June 22nd
- Track 1 proof of concept is required by June 16th
- Switch to Track 2 if Track 1 fails requirements in time
## Core Development Concepts

- [[Project Overview]] - Neutral scope, four-surface stack, objective, and build path
- [[Tech Stack]] - Project scaffolding, environment, build, SDK and API reference
- [[MCP Server]] - Isolated project MCP server: tool catalog, safety tiers, build phasing
- [[Agent Communication Contract]] - **Normative**: how agents/workflows/MCP stay pointed at the goal (the success-metric gate, North-Star Header, drift alarm)
- [[Market Conditions]] - Macro/micro trends, market sentiment
- [[Trading Strategies]] - Technical indicators, wallet monitoring, asset comparison
- [[Token Universe]] - The 20 tradeable tokens, selection methodology, and the theory behind them
- [[Strategy Logic]] - Visual map (flowcharts) of the universe pipeline, research loop, and runtime logic
- [[Simulated Market]] - Historical exchange data, back-testing environment
- [[Pool-Event Data Layer]] - On-chain PancakeSwap event collection (liquidity/flow/wallet panels) + its probes
- [[AI Training]] - RL/ML agent training, reward tuning, curriculum
- [[Security and Encryption]] - Wallet addresses, encryption methods, best practices
- [[Real-time Monitoring]] - Monitoring wallets, trade executions, fund transfers
- [[Social Media Scanner]] - X.com scanner for breaking news such as hacks or adoption
- [[Remote Capabilities]] - Training AI from host computer, results dashboards, CI/CD
- [[Apentic Data Contract]] - The static-JSON contract between training and the web frontend
- [[Live Forward-Run Harness]] - Running the trained RL champion (ef-s2) live in paper mode on EC2: weekly-replay reuse, live data feed, private weights store, the deployment
- [[Experiment Log]] - Rigid, reproducible record of every training iteration + the current champion
- [[Build Log]] - Chronological record of what's been built and the key decisions behind it

## Rules / Requirements

Refer to [[BNB Hack - AI Trading Agent Edition]] for contest rules, regulations, and requirements. 


