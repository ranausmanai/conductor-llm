"""FastAPI server for Conductor."""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

from conductor import (
    llm,
    llm_dry_run,
    smart_classify,
    CostController,
    Period,
    SQLiteStorage,
)


def _get_api_key() -> Optional[str]:
    return os.getenv("CONDUCTOR_API_KEY")


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    api_key = _get_api_key()
    if api_key and x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _controller() -> CostController:
    db_path = os.getenv("CONDUCTOR_DB_PATH", "conductor.db")
    storage = SQLiteStorage(db_path=db_path)
    return CostController(storage=storage)


app = FastAPI(title="Conductor API", version="2.1.0")


class LlmRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    quality: str = Field("auto", pattern="^(auto|high|low)$")
    api_key: Optional[str] = None
    max_tokens: int = Field(1000, ge=1, le=4096)


class LlmResponse(BaseModel):
    text: str
    model: str
    cost: float
    saved: float
    baseline_cost: float
    prompt_tokens: int
    completion_tokens: int


class DryRunRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    quality: str = Field("auto", pattern="^(auto|high|low)$")


class ClassifyRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class BudgetRequest(BaseModel):
    user_id: str
    hourly_usd: Optional[float] = None
    daily_usd: Optional[float] = None
    weekly_usd: Optional[float] = None
    monthly_usd: Optional[float] = None
    hard_limit: bool = True


class RecordRequest(BaseModel):
    user_id: str
    cost_usd: float
    model: str
    task_type: str
    feature: Optional[str] = None


class BudgetCheckRequest(BaseModel):
    user_id: str
    estimated_cost: float = 0.0


class ReportRequest(BaseModel):
    group_by: str = Field("user", pattern="^(user|model|task_type)$")
    period: Period = Period.DAILY
    top_n: int = Field(10, ge=1, le=100)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/llm", response_model=LlmResponse, dependencies=[Depends(_require_api_key)])
def llm_call(req: LlmRequest) -> LlmResponse:
    try:
        response = llm(
            req.prompt,
            quality=req.quality,
            api_key=req.api_key,
            max_tokens=req.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return LlmResponse(
        text=response.text,
        model=response.model,
        cost=response.cost,
        saved=response.saved,
        baseline_cost=response.baseline_cost,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
    )


@app.post("/llm/dry-run", dependencies=[Depends(_require_api_key)])
def llm_dry(req: DryRunRequest) -> Dict[str, Any]:
    return llm_dry_run(req.prompt, quality=req.quality)


@app.post("/classify", dependencies=[Depends(_require_api_key)])
def classify(req: ClassifyRequest) -> Dict[str, Any]:
    result = smart_classify(req.prompt)
    return {
        "complexity_score": result.complexity_score,
        "raw_score": result.raw_score,
        "reasoning": result.reasoning,
        "from_cache": result.from_cache,
    }


@app.post("/budgets", dependencies=[Depends(_require_api_key)])
def set_budget(req: BudgetRequest) -> Dict[str, Any]:
    controller = _controller()
    budget = controller.set_user_budget(
        req.user_id,
        hourly_usd=req.hourly_usd,
        daily_usd=req.daily_usd,
        weekly_usd=req.weekly_usd,
        monthly_usd=req.monthly_usd,
        hard_limit=req.hard_limit,
    )
    return budget.__dict__


@app.get("/budgets/{user_id}", dependencies=[Depends(_require_api_key)])
def get_budget(user_id: str) -> Dict[str, Any]:
    controller = _controller()
    budget = controller.get_user_budget(user_id)
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    return budget.__dict__


@app.delete("/budgets/{user_id}", dependencies=[Depends(_require_api_key)])
def remove_budget(user_id: str) -> Dict[str, Any]:
    controller = _controller()
    removed = controller.remove_user_budget(user_id)
    return {"removed": removed}


@app.post("/budgets/check", dependencies=[Depends(_require_api_key)])
def check_budget(req: BudgetCheckRequest) -> Dict[str, Any]:
    controller = _controller()
    return controller.check_budget(req.user_id, estimated_cost=req.estimated_cost)


@app.post("/records", dependencies=[Depends(_require_api_key)])
def add_record(req: RecordRequest) -> Dict[str, Any]:
    controller = _controller()
    record = controller.record_cost(
        user_id=req.user_id,
        cost_usd=req.cost_usd,
        model=req.model,
        task_type=req.task_type,
        feature=req.feature,
    )
    return {
        "user_id": record.user_id,
        "cost_usd": record.cost_usd,
        "model": record.model,
        "task_type": record.task_type,
        "timestamp": record.timestamp.isoformat(),
        "feature": record.feature,
        "record_id": record.record_id,
    }


@app.post("/reports", dependencies=[Depends(_require_api_key)])
def reports(req: ReportRequest) -> Dict[str, Any]:
    controller = _controller()
    report = controller.get_cost_report(
        group_by=req.group_by,
        period=req.period,
        top_n=req.top_n,
    )
    return {
        "period": report.period.value,
        "total_cost_usd": report.total_cost_usd,
        "total_requests": report.total_requests,
        "breakdown": report.breakdown,
        "top_spenders": report.top_spenders,
    }
