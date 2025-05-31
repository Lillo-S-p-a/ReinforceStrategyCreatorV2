graph LR
    subgraph DataIngestionStage
        direction LR
        A[Data Sources (CSV, API)] --> B(Fetcher);
        B --> C(Cache);
        C --> D(Validator);
        D --> E[Output Data];
    end