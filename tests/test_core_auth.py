import pytest
from unittest.mock import patch, MagicMock
from google.auth.exceptions import (
    DefaultCredentialsError,
)  # Import a specific exception
from coregenai.core import CoreGenAI
from coregenai.config import GenConfig


@patch("coregenai.core.genai.Client")
@patch("google.auth.default")
def test_vertex_auto_detect_project_id_success(mock_auth_default, mock_genai_client):
    """
    GIVEN CoreGenAI configures to use Vertex AI without an explicit project ID,
    WHEN google.auth.default successfully provides a project ID,
    THEN the genai.Client should be initialized with the auto-detected project ID.
    """
    # Arrange
    expected_project_id = "auto-detected-project-123"
    mock_auth_default.return_value = (MagicMock(), expected_project_id)

    config = GenConfig(
        backend="vertex",
        google_project_id=None,  # Crucially, project ID is not set
        google_location="us-central1",
    )

    # Act
    core_gen_ai = CoreGenAI(config=config)

    # Assert
    mock_auth_default.assert_called_once()  # Ensure auto-detection was attempted

    _, kwargs = mock_genai_client.call_args
    assert kwargs["project"] == expected_project_id
    assert kwargs["vertexai"] is True
    assert kwargs["location"] == "us-central1"
    assert kwargs["api_key"] is None  # Ensure API key is not used for Vertex


@patch("coregenai.core.genai.Client")
@patch("google.auth.default")
def test_vertex_explicit_project_id_precedence(mock_auth_default, mock_genai_client):
    """
    GIVEN CoreGenAI configures to use Vertex AI with an explicit project ID,
    WHEN initializing the client,
    THEN the explicit project ID should take precedence, and auto-detection should not occur.
    """
    # Arrange
    explicit_project_id = "my-explicit-project-456"

    config = GenConfig(
        backend="vertex",
        google_project_id=explicit_project_id,  # Explicit project ID is set
        google_location="us-central1",
    )

    # Act
    core_gen_ai = CoreGenAI(config=config)

    # Assert
    mock_auth_default.assert_not_called()  # Auto-detection should be skipped

    _, kwargs = mock_genai_client.call_args
    assert kwargs["project"] == explicit_project_id
    assert kwargs["vertexai"] is True
    assert kwargs["location"] == "us-central1"
    assert kwargs["api_key"] is None


@patch("coregenai.core.genai.Client")
@patch("google.auth.default")
def test_vertex_auto_detect_project_id_failure(mock_auth_default, mock_genai_client):
    """
    GIVEN CoreGenAI configures to use Vertex AI without an explicit project ID,
    WHEN google.auth.default fails to provide a project ID (e.g., due to missing credentials),
    THEN the genai.Client should be initialized with project=None,
    allowing the underlying SDK to handle the credential error if necessary.
    """
    # Arrange
    mock_auth_default.side_effect = DefaultCredentialsError(
        "Could not determine credentials."
    )

    config = GenConfig(
        backend="vertex",
        google_project_id=None,  # Project ID is not set
        google_location="us-central1",
    )

    # Act
    core_gen_ai = CoreGenAI(config=config)

    # Assert
    mock_auth_default.assert_called_once()  # Ensure auto-detection was attempted

    _, kwargs = mock_genai_client.call_args
    assert (
        kwargs["project"] is None
    )  # Project should remain None if auto-detection fails
    assert kwargs["vertexai"] is True
    assert kwargs["location"] == "us-central1"
    assert kwargs["api_key"] is None
