from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_path: Path
    app_cache_dir: Path
    outputs_dir: Path
    timezone: str = "Europe/Paris"
    log_level: str = "INFO"
    # Agent
    llm_assisted: bool = False  # OFF par défaut (contrainte)

def load_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]  # src/utils -> project root
    return Settings(
        project_root=root,
        data_path=root / "data" / "bdd_natalité_mortalité_clean.xlsx",
        app_cache_dir=root / "app" / "cache",
        outputs_dir=root / "app" / "outputs",
    )

settings = load_settings()