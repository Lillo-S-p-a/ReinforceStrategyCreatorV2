graph LR
    subgraph FeatureEngineeringStage
        direction LR
        A[Input Data] --> B{Transformation Step 1};
        B --> C{Transformation Step 2};
        C --> D{...};
        D --> E{Transformation Step N};
        E --> F[Output Features];
    end