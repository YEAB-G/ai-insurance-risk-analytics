from src.data.load_data import load_raw_data


def test_load_raw_data_signature():
    """Basic smoke test that the function can be called without crashing
    when the file is missing (it should raise FileNotFoundError)."""
    # We only check that the function exists here; actual data loading
    # is tested manually when the dataset is available.
    assert callable(load_raw_data)
