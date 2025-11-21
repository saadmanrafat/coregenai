from typing import Literal
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenConfig(BaseSettings):
    """
    Configuration for CoreGenAI, loaded declaratively from environment variables.

    This class uses `pydantic-settings` to automatically read values
    from the environment. For example, the field `project_id` will be
    populated by the `PROJECT_ID` environment variable (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    google_project_id: str | None = None
    google_location: str | None = None
    google_api_key: str | None = None
    google_application_credentials: str | None = None
    backend: Literal["auto", "vertex", "studio"] = "auto"

    model_name: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 8192
    request_timeout: float = 30.0
    max_retries: int = 3

    mock_mode: bool = False
    mock_response_payload: str | None = None

    @computed_field
    @property
    def use_vertex(self) -> bool:
        """
        Determines if the Vertex AI backend should be used.

        This is a computed field that resolves the 'auto' backend setting
        into a concrete boolean decision during model instantiation, making
        the configuration snapshot consistent and predictable.
        """
        if self.backend == "vertex":
            return True
        if self.backend == "studio":
            return False

        # 'auto' mode logic: prefer Vertex if ADC or project_id is available.
        # This logic now relies on fields within the model itself, making it
        # more robust and easily testable.
        return bool(self.google_application_credentials) or bool(self.google_project_id)
