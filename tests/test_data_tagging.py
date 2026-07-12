from smart_data_hub.data_tagging import filter_tagged_data
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def property_df():
    # Create a sample DataFrame for testing
    data = {
        "ID": ["1fgh", "2hjh", "3kij", "4kop"],
        "property": [
            "permeability",
            "hydraulic_conductivity",
            "density",
            "porosity",
        ],
        "agency": [["BGR", "BGE"], ["NAGRA"], ["BfS/BGE"], ["GRS"]],
        "location": [
            ["Morsleben", "Germany"],
            ["Switzerland"],
            ["Germany"],
            ["Germany"],
        ],
        "simplified_lithology": [["Mudstone"], ["Limestone"], None, ["Rocksalt"]],
    }

    return pd.DataFrame(data)

@pytest.fixture
def test_tag_dict():
    tag_lithology = ["Mudstone", "Rocksalt"]
    tag_agency = ["BGR", "BGE", "NAGRA"]
    tag_location = ["Germany"]
    tag_id = ["1fgh"]
    return {
        "agency": tag_agency,
        "location": tag_location,
        "simplified_lithology": tag_lithology,
        "ID": tag_id,
    }


def test_data_tagging(property_df, test_tag_dict):
    
    # Apply the filter_tagged_data function
    filtered_df = filter_tagged_data(property_df, test_tag_dict)

    # Expected output DataFrame after filtering
    expected_data = {
        "ID": ["1fgh"],
        "property": ["permeability"],
        "agency": ["BGR, BGE"],
        "location": ["Morsleben, Germany"],
        "simplified_lithology": ["Mudstone"],
    }
    expected_df = pd.DataFrame(expected_data)

    assert_frame_equal(
        filtered_df.reset_index(drop=True),
        expected_df,
        check_dtype=False,
    )
