graph TD
    subgraph EvaluationStage
        A[Trained Model] --> C{Metric Calculation};
        B[Test Data] --> C;
        C --> D(Benchmarking);
        D --> E(Report Generation);
        D --> F(Plot Generation);
    end