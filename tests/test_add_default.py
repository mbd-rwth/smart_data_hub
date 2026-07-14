from pathlib import Path
import os
from smart_data_hub.add_default import load_rock_property, add_default_df



def test_add_default():

    property_df = load_rock_property(
        [os.path.join(Path(__file__).resolve().parent.parent, "dataset", "rock_property", "DE_South_Claystone", "Muschelkalk_Middle.yaml")]
    )
    lithologies = ["Rocksalt", "Mudstone"]
    add_default_property_df = add_default_df(property_df, lithologies)

    # check if there are still missing properties
    missing = add_default_property_df.loc[add_default_property_df["ID"].isna()]
    assert missing.empty, (
        f"add_default_df did not fill all missing IDs. "
        f"Rows with missing ID:\n{missing}"
    )
