Deploying New Models
====================

This section explains the process of deploying a trained and validated model using the ReinforceStrategyCreator Pipeline. Deployment involves packaging the model and then promoting it to a target environment (e.g., paper trading, live trading).

Core Deployment Components
--------------------------

*   **Model Registry (``src.models.registry.ModelRegistry``)**: Stores metadata and references to trained model artifacts. Before a model can be deployed, it must be registered here, typically after a successful training and evaluation run.
*   **Artifact Store (``src.artifact_store.base.ArtifactStore``)**: The central storage for all pipeline artifacts, including trained models, datasets, and deployment packages.
*   **Model Packager (``src.deployment.packager.ModelPackager``)**: Responsible for creating self-contained deployment packages. These packages (``.tar.gz`` files) include:
    *   The model files (weights, configuration).
    *   A ``deployment_manifest.json`` with metadata about the model, package, and deployment settings.
    *   A ``requirements.txt`` listing dependencies.
    *   Helper scripts like ``run.py`` (a placeholder for serving the model) and ``health_check.py``.
    *   A ``README.md`` for the package.
*   **Deployment Manager (``src.deployment.manager.DeploymentManager``)**: Orchestrates the deployment process. It uses the ``ModelPackager`` to create a deployment bundle (if not already provided) and then deploys this bundle to a specified target environment using a chosen strategy (e.g., direct, rolling). It also tracks deployment status and history.
*   **Specialized Deployers (e.g., ``src.deployment.paper_trading.PaperTradingDeployer``)**: For specific deployment targets like paper trading, specialized deployer classes build upon the ``DeploymentManager`` to handle interactions with the target system (e.g., a paper trading brokerage API).

Deployment Workflow
-------------------

1.  **Ensure Model is Trained and Registered**:
    *   Your model should have completed the training and evaluation stages.
    *   The chosen model version should be registered in the ``ModelRegistry``. This typically happens at the end of a successful training run where the model, its metadata, and performance metrics are saved.

2.  **Package the Model (Optional, can be done by DeploymentManager)**:
    *   While the ``DeploymentManager`` can create a package on the fly, you can also pre-package a model using the ``ModelPackager``.
    *   This step creates the ``.tar.gz`` deployment artifact and stores it in the ``ArtifactStore``.

    .. code-block:: python

       from src.models.registry import ModelRegistry
       from src.artifact_store.local_adapter import LocalArtifactStore # Example
       from src.deployment.packager import ModelPackager

       artifact_store = LocalArtifactStore(base_path="./artifacts")
       model_registry = ModelRegistry(artifact_store=artifact_store)
       packager = ModelPackager(model_registry=model_registry, artifact_store=artifact_store)

       model_id_to_package = "my_trained_dqn_model"
       model_version_to_package = "1.2.0" # Or None for latest
       
       try:
           package_id = packager.package_model(
               model_id=model_id_to_package,
               model_version=model_version_to_package,
               # This deployment_config is stored in the package's manifest.json
               # and might contain general settings for how this package should be deployed.
               deployment_config={"default_target_type": "paper_trading", "required_resources": "medium"},
               tags=["paper_trading_candidate"],
               description=f"Package for {model_id_to_package} v{model_version_to_package} for paper trading"
           )
           print(f"Model packaged successfully. Package ID: {package_id}")
       except ValueError as e:
           print(f"Error packaging model: {e}")

3.  **Deploy the Model using DeploymentManager**:
    *   The ``DeploymentManager.deploy()`` method is used to deploy a model (either by referencing its ``model_id`` or a pre-existing ``package_id``) to a target environment.

    .. code-block:: python

       from src.deployment.manager import DeploymentManager, DeploymentStrategy
       # ... (artifact_store and model_registry initialized as above) ...

       deployment_manager = DeploymentManager(
           model_registry=model_registry,
           artifact_store=artifact_store,
           deployment_root="./my_deployments" # Directory to store deployment state and extracted packages
       )

       target_env = "paper_trading_simulation"
       
       try:
           deployment_id = deployment_manager.deploy(
               model_id=model_id_to_package, # From step 2
               model_version=model_version_to_package, # From step 2
               target_environment=target_env,
               # package_id=package_id, # Optionally provide if pre-packaged
               deployment_config={ # Specific settings for this deployment instance
                   "initial_capital": 50000, 
                   "max_active_trades": 5
               },
               strategy=DeploymentStrategy.DIRECT # Or ROLLING, CANARY (if implemented)
           )
           print(f"Deployment initiated. Deployment ID: {deployment_id}")
           status = deployment_manager.get_deployment_status(deployment_id)
           print(f"Deployment status: {status['status']}")
       except ValueError as e:
           print(f"Error deploying model: {e}")

    The ``DeploymentManager`` will:
    *   Create a package if ``package_id`` is not provided.
    *   Extract the package contents to a versioned directory within ``deployment_root/<target_environment>/<model_id>/<deployment_id>``.
    *   Update symlinks or perform other actions based on the chosen deployment strategy (e.g., for a "rolling" deployment, it might update a ``current`` symlink).
    *   Record the deployment in its state file.

4.  **Using Specialized Deployers (e.g., Paper Trading)**:
    *   For environments like paper trading, a specialized class like ``PaperTradingDeployer`` is used. It typically wraps the ``DeploymentManager`` and adds logic specific to the target system.
    *   Refer to ``examples/paper_trading_example.py`` for a detailed example. The ``PaperTradingDeployer.deploy_to_paper_trading()`` method would internally handle getting the model (possibly via ``DeploymentManager``) and setting up the simulation environment.

    .. code-block:: python

       # Simplified from examples/paper_trading_example.py
       from src.deployment.paper_trading import PaperTradingDeployer
       # ... (components initialized) ...

       paper_trading_deployer = PaperTradingDeployer(
           deployment_manager=deployment_manager,
           model_registry=model_registry,
           artifact_store=artifact_store,
           paper_trading_root="./paper_trading_simulations"
       )

       simulation_config = {
           "initial_capital": 100000.0,
           "symbols": ["AAPL", "GOOGL"],
           # ... other paper trading parameters
       }
       
       # This step might internally call DeploymentManager.deploy or use its own logic
       # to prepare the model for the paper trading environment.
       # The example directly loads a mock model, but in a real scenario,
       # it would fetch a registered and packaged model.
       simulation_id = paper_trading_deployer.deploy_to_paper_trading(
           model_id="my_registered_model_id", # Ensure this model is packaged or deployable
           model_version="1.0.0",
           simulation_config=simulation_config
       )
       print(f"Paper trading simulation deployed with ID: {simulation_id}")
       
       # Then start and manage the simulation
       # paper_trading_deployer.start_simulation(simulation_id)
       # ...

Deployment Strategies
---------------------

The ``DeploymentManager`` supports different deployment strategies (passed to the ``strategy`` parameter of the ``deploy`` method):

*   **``DIRECT``**: The new version directly replaces the old one. This might involve a brief downtime.
*   **``ROLLING``**: (Conceptual - implementation details in ``DeploymentManager``) The new version is rolled out incrementally. For simple file-based deployments, this might involve updating a symlink (e.g., ``current``) to point to the new version's directory after it's fully prepared. This aims for zero-downtime.
*   **``BLUE_GREEN``**: (Conceptual) Two identical environments are maintained. The new version is deployed to the inactive (green) environment. After testing, traffic is switched to the green environment, which then becomes blue.
*   **``CANARY``**: (Conceptual) The new version is rolled out to a small subset of users/traffic first. If it performs well, it's rolled out to the rest.

Consult the ``DeploymentManager`` source code (``src/deployment/manager.py``) for details on how these strategies are implemented for file-based deployments.

Rollback
--------
The ``DeploymentManager`` provides a ``rollback()`` method to revert to a previous deployment if issues arise with the current one.

.. code-block:: python

   try:
       # Assuming 'current_deployment_id' is the ID of the problematic deployment
       # and 'previous_good_deployment_id' is known.
       # Or, to rollback to the immediately previous version:
       rollback_deployment_id = deployment_manager.rollback(
           model_id=model_id_to_package,
           target_environment=target_env
           # to_deployment_id="specific_previous_deployment_id" # Optional
       )
       print(f"Rollback successful. Active deployment ID: {rollback_deployment_id}")
   except ValueError as e:
       print(f"Rollback failed: {e}")

This process ensures that models are packaged consistently and deployed in a managed way, with capabilities for tracking and rollback.