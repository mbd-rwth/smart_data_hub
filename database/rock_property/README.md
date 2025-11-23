## Structure of the YAML file 
This folder provides rock properties for each stratigraphic layer, including properties such as density, porosity, hydraulic conductivity, etc.  

The layout of the YAML files within these basis folders follows a specific structure:

If no value is provided for a specific entry field, then enter null.

```
field:
  source: STR                       # String with BibTeX key of data source.
  type: STR                         # String out of [scalar, dictionary, expression].
  value: VAL                        # A value of type float, integer, string, or dictionary.
  value_min: VAL                    # Minimum value.
  value_max: VAL                    # Maximum value.
  value_std: VAL                    # Standard deviation of the value.
  probability_distribution:
    sample_size: INT                # Number of samples to be drawn.
    sampled_data: LIST[VAL] or STR  # List of sampled data or a scipy.stats.rv_continuous().rvs() function.
  unit_str: STR                     # Standard string to indicate unit.
  unit_base: LIST(INT)              # An array of the form [ kg m s K A mol cd ] that gives the unit as the exponent of the SI basis units, e.g., m/s^2 is [0, 1, -2, 0, 0, 0, 0].
  variable_name: STR                # Function argument (e.g., temperature) (must be used if type is dictionary or expression).
  variable_unit_str: STR            # Standard string to indicate variable_unit.
  variable_unit_base: LIST(INT)     # See above (must be used if type is tabulated or expression).
  description: STR                  # Free text metadata.
  tag:
    agency: LIST[STR]              # List of agencies that provided the data.
    location: LIST[STR]            # List of locations where the data applies.
    simplified_lithology: LIST[STR]# List of simplified lithologies the data applies to.
    ID: STR                        # A unique ID for the data entry. It can be generated via the helper function: ../../src/generate_id.py.           
``` 

