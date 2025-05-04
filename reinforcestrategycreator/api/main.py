from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from .routers import runs, episodes

app = FastAPI(
    title="ReinforceStrategyCreator Metrics API",
    description="API for accessing training run performance metrics.",
    version="0.1.0",
)

# Configure CORS
# Adjust origins as needed for production deployment
origins = [
    "*", # Allow all origins for now - TODO: Restrict in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(runs.router, prefix="/api/v1")
app.include_router(episodes.router, prefix="/api/v1")


@app.get("/", tags=["Health Check"])
async def read_root():
    """
    Root endpoint for health check.
    """
    return {"status": "ok", "message": "Welcome to ReinforceStrategyCreator Metrics API"}

# Placeholder for running with uvicorn (for local testing)
# if __name__ == "__main__":
#     import uvicorn
#     # Remember to set the API_KEY environment variable before running
#     # Example: API_KEY=your_secret_key uvicorn reinforcestrategycreator.api.main:app --reload
#     uvicorn.run("reinforcestrategycreator.api.main:app", host="0.0.0.0", port=8000, reload=True)