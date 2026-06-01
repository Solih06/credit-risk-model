from pydantic import BaseModel

class CreditScoringRequest(BaseModel):
    ProviderId: str
    ProductId: str
    ProductCategory: str
    TransactionHour: int
    DayOfWeek: int
    IsWeekend: int
    IsNightTransaction: int
    Amount: float

class CreditScoringResponse(BaseModel):
    risk_probability: float
    credit_decision: str