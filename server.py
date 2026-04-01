"""
FastAPI server for the Trust and Safety Decision Engine.
Provides REST API endpoints for interacting with the environment.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import json

from environment import TrustAndSafetyEnv
from models import Action, EpisodeConfig, Observation, StepResult


# Initialize FastAPI app
app = FastAPI(
    title="Trust and Safety Decision Engine",
    description="OpenEnv-compatible RL environment for content moderation",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = TrustAndSafetyEnv()


# Request/Response models
class ResetRequest(BaseModel):
    """Request to reset the environment."""
    difficulty: str = Field(default="medium", description="Task difficulty: easy, medium, hard")
    task_id: Optional[str] = Field(default=None, description="Optional specific task ID")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class ResetResponse(BaseModel):
    """Response from reset endpoint."""
    episode_id: str
    observation: Observation
    info: Dict[str, Any]


class StepRequest(BaseModel):
    """Request to step the environment."""
    action: str = Field(..., description="Action string")


class StepResponse(BaseModel):
    """Response from step endpoint."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    """Response from state endpoint."""
    state: Dict[str, Any]


class InfoResponse(BaseModel):
    """Response from info endpoint."""
    info: Dict[str, Any]


# Endpoints

@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest):
    """
    Reset the environment and start a new episode.
    
    Args:
        request: ResetRequest with optional difficulty and task_id
    
    Returns:
        ResetResponse with initial observation and episode metadata
    
    Example:
        POST /reset
        {"difficulty": "medium"}
    """
    try:
        config = EpisodeConfig(
            difficulty=request.difficulty,
            task_id=request.task_id,
            seed=request.seed,
        )
        reset_result = env.reset(config)
        
        return ResetResponse(
            episode_id=reset_result.info["episode_id"],
            observation=reset_result.observation,
            info=reset_result.info,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """
    Take a step in the environment.
    
    Args:
        request: StepRequest with action string
    
    Returns:
        StepResponse with observation, reward, done flag, and metadata
    
    Example:
        POST /step
        {"action": "ANALYZE: toxicity=high, intent=malicious"}
    
    Valid action formats:
        - "ANALYZE: toxicity=<level>, intent=<intent>"
        - "CHECK_HISTORY"
        - "DECIDE: <action>" where action is ALLOW, DELETE, REDUCE_VISIBILITY, or ESCALATE
        - "ESCALATE"
    """
    try:
        if env.episode_id is None:
            raise HTTPException(
                status_code=400,
                detail="Must call /reset before /step"
            )
        
        action = Action(action=request.action)
        result = env.step(action)
        
        return StepResponse(
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            info=result.info,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=StateResponse)
async def state():
    """
    Get the current state of the environment.
    
    Returns:
        Current environment state including actions, rewards, and episode info
    """
    try:
        if env.episode_id is None:
            raise HTTPException(
                status_code=400,
                detail="No active episode. Call /reset first."
            )
        
        return StateResponse(state=env.state())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/info", response_model=InfoResponse)
async def info():
    """
    Get information about the environment.
    
    Returns:
        Environment metadata including action/observation spaces and reward range
    """
    return InfoResponse(info=env.get_info())


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Trust and Safety Decision Engine",
        "version": "1.0.0",
    }


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "service": "Trust and Safety Decision Engine",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /reset": "Start a new episode",
            "POST /step": "Take an action in the environment",
            "GET /state": "Get current environment state",
            "GET /info": "Get environment information",
            "GET /health": "Health check",
            "GET /docs": "API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)",
        },
        "example_workflow": {
            "1_reset": "POST /reset with {\"difficulty\": \"medium\"}",
            "2_step": "POST /step with {\"action\": \"ANALYZE: toxicity=high, intent=malicious\"}",
            "3_step": "POST /step with {\"action\": \"CHECK_HISTORY\"}",
            "4_step": "POST /step with {\"action\": \"DECIDE: DELETE\"}",
            "5_state": "GET /state to view final results",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
