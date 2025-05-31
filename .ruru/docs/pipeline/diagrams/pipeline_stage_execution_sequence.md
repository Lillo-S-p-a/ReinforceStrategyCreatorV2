graph TD
    A["ModelPipeline.run()"] --> B{"Loop through Stages"};
    B --> C("Stage 1: Execute");
    C --> D{"Output from Stage 1"};
    D --> E("Stage 2: Execute with Input from Stage 1");
    E --> F{"Output from Stage 2"};
    F --> G("...");
    G --> H("Stage N: Execute with Input from Stage N-1");
    H --> I{"Final Output"};