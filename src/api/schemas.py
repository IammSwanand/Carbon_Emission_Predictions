from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict


class CountryFeaturesInput(BaseModel):
    """Input payload for CO2 emissions prediction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cereal_yield": 2845.5,
                "fdi_perc_gdp": 2.15,
                "gni_per_cap": 8420.0,
                "en_per_cap": 2150.3,
                "pop_urb_aggl_perc": 34.2,
                "prot_area_perc": 15.6,
                "pop_growth_perc": 1.25,
                "urb_pop_growth_perc": 2.45,
            }
        }
    )

    cereal_yield: float = Field(
        ..., ge=0.0, description="Cereal yield in kg per hectare"
    )
    fdi_perc_gdp: Optional[float] = Field(
        default=0.0, description="Foreign direct investment, net inflows (% of GDP)"
    )
    gni_per_cap: float = Field(
        ..., ge=0.0, description="GNI per capita, Atlas method (current US$)"
    )
    en_per_cap: float = Field(
        ..., ge=0.0, description="Energy use (kg of oil equivalent per capita)"
    )
    pop_urb_aggl_perc: float = Field(
        ..., ge=0.0, le=100.0, description="Population in urban agglomerations > 1M (%)"
    )
    prot_area_perc: float = Field(
        ..., ge=0.0, le=100.0, description="Terrestrial and marine protected areas (%)"
    )
    pop_growth_perc: float = Field(
        ..., ge=-100.0, le=100.0, description="Annual population growth rate (%)"
    )
    urb_pop_growth_perc: float = Field(
        ..., ge=-100.0, le=100.0, description="Annual urban population growth rate (%)"
    )


class PredictionResponse(BaseModel):
    """Prediction output schema."""
    predicted_co2_per_cap: float = Field(
        ..., description="Predicted CO2 emissions in metric tons per capita"
    )
    unit: str = "metric tons per capita"
    model_name: str
    model_version: str
    is_out_of_distribution: bool = Field(
        ..., description="True if any feature value deviates beyond training bounds"
    )
    warnings: List[str] = Field(default_factory=list)


class BatchPredictionInput(BaseModel):
    records: List[CountryFeaturesInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    count: int
    model_version: str


class ScenarioProjectionInput(BaseModel):
    country: str = Field(..., description="ISO-3 Country Code (e.g., IND, USA)")
    base_features: CountryFeaturesInput
    start_year: int = Field(default=2024, ge=2000, le=2100)
    years_ahead: int = Field(default=10, ge=1, le=30)
    growth_rate_overrides: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional feature annual growth rates as decimals (e.g. {'en_per_cap': -0.02})"
    )


class ScenarioProjectionResponse(BaseModel):
    country: str
    timeline: List[int]
    projected_co2: List[float]
    assumed_growth_rates: Dict[str, float]
    methodology: str = (
        "Scenario simulation compounding base features and scoring against trained pipeline. "
        "Predictions bounded by training envelope."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    features: List[str]
    version: str

