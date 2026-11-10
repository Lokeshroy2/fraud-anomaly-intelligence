"""Pydantic request/response schemas for the scoring API.

Transactions have ~140 usable raw columns (and 339 raw V-columns, of which
only a fitted-model-dependent subset is ever used). Rather than hand-listing
every column as a top-level field, the well-known "core" columns are typed
explicitly, and the high-cardinality C/D/M/V groups are accepted as nested
dicts (e.g. `C={"C1": 1.0, "C2": 0.0}`) keyed by the original column name.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionIn(BaseModel):
    TransactionID: int
    TransactionDT: int = Field(description="Seconds since a reference timestamp")
    TransactionAmt: float = Field(gt=0)
    ProductCD: str | None = None
    card1: int
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None
    C: dict[str, float] = Field(default_factory=dict, description='e.g. {"C1": 1.0, "C2": 0.0}')
    D: dict[str, float] = Field(default_factory=dict, description='e.g. {"D1": 14.0}')
    M: dict[str, str] = Field(default_factory=dict, description='e.g. {"M1": "T", "M5": "F"}')
    V: dict[str, float] = Field(default_factory=dict, description='e.g. {"V1": 1.0, "V45": 0.0}')


class Reason(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: Literal["increases", "decreases"]


class ScoreOut(BaseModel):
    TransactionID: int
    fraud_probability: float
    decision: Literal["approve", "review", "block"]
    top_reasons: list[Reason]


class BatchScoreIn(BaseModel):
    transactions: list[TransactionIn]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    n_features: int
    review_threshold: float
    block_threshold: float
    review_cost: float
    block_fp_cost: float
