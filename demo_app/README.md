# demo_app (SYMAD-ECNN)

## Run backend (Flask)
cd demo_app/backend
pip install -r requirements.txt
python api.py

Health:
http://localhost:5000/health

## Run frontend (Streamlit)
cd demo_app/frontend
pip install -r requirements.txt
streamlit run streamlit_app.py

## Notes
- Model path is auto-detected from repo:
  models/saved_models/ecnn_optimized_best.pth
- Two modes:
  - Raw MRI (preprocess ON)
  - Preprocessed validation slice (preprocess OFF)
- Default threshold uses optimal_threshold from metrics_ecnn_v3.json

## Backend OOAD structure
The backend has been split into service-oriented modules:

- `backend/domain_models.py`
  - `PredictOptions`, `InferenceMaps`, `AggregationResult`, `PredictionResponse`
- `backend/model_architecture.py`
  - `ECNNAutoencoderV3`
- `backend/model_loader_service.py`
  - `ModelLoaderService`
- `backend/preprocessing_service.py`
  - `PreprocessingService`
- `backend/inference_service.py`
  - `InferenceService`, `RiskScoringService`
- `backend/prediction_service.py`
  - `PredictionService` (orchestrates preprocessing + inference + aggregation)

`backend/api.py` now delegates prediction flow to `PredictionService` while keeping the existing API contract unchanged.
