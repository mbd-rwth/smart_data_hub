from src.data_tagging import filter_tagged_data
import pandas as pd


def test_data_tagging():
    tag_lithology = ["Mudstone", "Rocksalt"]
    tag_agency = ["BGR", "BGE", "NAGRA"]
    tag_location = ["Germany"]
    tag_id = ["1fgh"]
    test_tag_dict = {
        "agency": tag_agency,
        "location": tag_location,
        "simplified_lithology": tag_lithology,
        "ID": tag_id,
    }

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
    property_df = pd.DataFrame(data)

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

    diff_ID = set(filtered_df["ID"]) - set(expected_df["ID"])
    # Compare the differnce in IDs
    assert not diff_ID
    print("Data tagging test passed!")
