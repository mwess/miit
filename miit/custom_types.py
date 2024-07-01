from typing import Tuple, Union, Dict, List

import pandas
import pyimzml


PdDataframe = pandas.core.frame.DataFrame
ImzmlParserType = pyimzml.ImzMLParser.ImzMLParser
IntensityDict = Dict[Union[int, str], List[float]]