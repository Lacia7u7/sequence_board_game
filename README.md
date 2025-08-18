# Sequence Web Migration

This repository is being migrated from a Tkinter-based desktop application to a
modern web stack consisting of a React frontend, a Flask backend and a
reinforcement learning module powered by PyTorch.

## Monorepo layout

```
.
├── apps/
│   ├── web/                 # React + TypeScript frontend (Vite)
│   └── server/              # Flask API and game logic
├── packages/
│   └── game-rules-ts/       # Shared TypeScript rule utilities
├── model/                   # RL training code and weights
├── assets/
│   ├── layouts/             # Official board layout JSON
│   └── svg-cards/           # Vector card assets (placeholders)
├── infra/
│   ├── docker/              # Dockerfiles
│   ├── firebase/            # Firebase config and rules
│   └── github-actions/      # CI workflows
└── tests/                   # Python unit tests for game rules
```

## Getting started

### Backend

```bash
pip install -r apps/server/requirements.txt
python apps/server/app.py
```

### Frontend

```bash
npm --prefix apps/web install
npm --prefix apps/web run dev
```

### Run tests

```bash
pytest
npm --prefix packages/game-rules-ts test
npm --prefix apps/web test
```

## Board layout validation

A script is provided to verify the official Sequence board layout:

```bash
python assets/layouts/validate_sequence.py
```

## Status

This migration is in its early stages.  Additional functionality such as
Firebase authentication, real-time multiplayer, AI opponents and comprehensive
tests will be implemented in future commits.
