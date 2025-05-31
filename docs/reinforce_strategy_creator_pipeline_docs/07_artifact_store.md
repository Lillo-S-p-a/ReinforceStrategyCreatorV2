# 7. Artifact Store (`src/artifact_store/`)

The Artifact Store is a critical component of the MLOps pipeline, responsible for managing the lifecycle of various outputs generated during pipeline execution. This includes trained models, datasets, evaluation results, and potentially other intermediate files. The implementation is likely located in `reinforcestrategycreator_pipeline/src/artifact_store/`.

### 7.1. Purpose and Architecture
The primary purposes of the Artifact Store are:
*   **Persistence:** To reliably save important outputs from pipeline stages.
*   **Traceability:** To link artifacts back to the specific pipeline run, configuration, and code version that produced them.
*   **Versioning:** To manage multiple versions of artifacts, allowing for rollback or comparison.
*   **Accessibility:** To provide a centralized location for other pipeline stages or external processes to retrieve these artifacts.

The architecture typically involves an adapter-based design to support different storage backends, with a common interface for storing and retrieving artifacts.

### 7.2. Supported Backends
The `artifact_store.type` parameter in `pipeline.yaml` specifies the backend to use. The architect's outline and `pipeline.yaml` suggest support for:
*   **`local`:** Stores artifacts on the local filesystem. The `artifact_store.root_path` (e.g., `"./artifacts"`) defines the base directory for local storage. This is suitable for development and smaller-scale deployments.
*   **Other Potential Backends (as implied by common MLOps practices and the `type` option):**
    *   `s3`: Amazon S3 for scalable cloud storage.
    *   `gcs`: Google Cloud Storage.
    *   `azure`: Azure Blob Storage.
Configuration for these cloud backends would typically involve additional parameters like bucket names, credentials, and regions.

### 7.3. Versioning and Metadata
Effective artifact management relies on versioning and metadata:
*   **Versioning:** If `artifact_store.versioning_enabled` is `true` (as per `pipeline.yaml`), the store will keep track of different versions of an artifact (e.g., multiple versions of a trained model resulting from different runs or HPO trials). This is crucial for reproducibility and for comparing model performance over time. The exact versioning scheme (e.g., run ID, timestamp, semantic versioning) would be part of the store's implementation.
*   **Metadata:** Along with the artifact itself, the store manages associated metadata. The `artifact_store.metadata_backend` (e.g., `"json"`, `"sqlite"`, `"postgres"`) specifies how this metadata is stored. Metadata can include:
    *   Timestamp of creation.
    *   Pipeline run ID.
    *   Source code version (Git commit hash).
    *   Configuration parameters used.
    *   Performance metrics (for model artifacts).
    *   Custom tags or descriptions.

The architect's outline notes that the internal workings of versioning implementation are a potential area for deeper documentation.

### 7.4. Storing and Retrieving Artifacts
Pipeline stages interact with the Artifact Store to:
*   **Store Artifacts:** After a stage completes its task (e.g., TrainingStage produces a model, EvaluationStage generates a report), it uses the Artifact Store's interface to save the output. The store handles the actual writing to the configured backend and records relevant metadata.
*   **Retrieve Artifacts:** Subsequent stages or external processes can query the Artifact Store to retrieve specific artifacts, perhaps by name, version, or associated metadata (e.g., "get the latest version of model X" or "get model Y from run Z").

The `artifact_store.cleanup_policy` in `pipeline.yaml` (with sub-parameters `enabled`, `max_versions_per_artifact`, `max_age_days`) defines rules for automatically deleting old or superseded artifacts to manage storage space, though it's disabled by default in the provided base configuration.