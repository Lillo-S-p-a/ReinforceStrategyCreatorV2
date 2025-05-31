graph TD
    A[run_main_pipeline.py] --> B(Load Configuration);
    B --> C{Orchestrator};
    C --> D[Stage 1];
    C --> E[Stage 2];
    C --> F[...];
    C --> G[Stage N];