graph TD
    subgraph TrainingStage
        A[Input Data] --> B(Model Initialization);
        B --> C{Training Loop};
        C -- Optional HPO ---> D(Hyperparameter Optimization);
        D --> C; 
        C --> E(Save Checkpoints);
        E --> F[Trained Model];
    end