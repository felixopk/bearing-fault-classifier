"""
Basic API tests for CI/CD pipeline
"""
import pytest
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_main():
    """Test that main module can be imported"""
    from app import main
    assert hasattr(main, 'app')


def test_class_labels():
    """Test class labels are defined"""
    from app.main import CLASS_LABELS
    assert len(CLASS_LABELS) == 4
    assert 'Ball' in CLASS_LABELS
    assert 'Inner_Race' in CLASS_LABELS
    assert 'Normal' in CLASS_LABELS
    assert 'Outer_Race' in CLASS_LABELS


def test_pydantic_models():
    """Test Pydantic models"""
    from app.main import PredictionInput, PredictionOutput, HealthResponse
    
    # Test PredictionInput
    input_data = PredictionInput(features=[0.1] * 19)
    assert len(input_data.features) == 19
    
    # Test PredictionOutput
    output = PredictionOutput(
        prediction="Normal",
        confidence=0.95,
        probabilities={"Normal": 0.95},
        status="success"
    )
    assert output.prediction == "Normal"
    assert output.confidence == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
