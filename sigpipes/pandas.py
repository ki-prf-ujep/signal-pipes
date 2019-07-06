from sigpipes.sigoperator import MaybeConsumerOperator
from sigpipes.sigcontainer import SigContainer, TimeUnit
from typing import Union

import pandas as pd


class DataFrame(MaybeConsumerOperator):
    """
    Transformation signal part of container into pandas dataframe.
    """
    def __init__(self, file: str = None, *, time_unit: TimeUnit = TimeUnit.SECOND):
        """
        Args:
            file:  name of file to which the dataframe is stored
            (in native pandas pickle format), if it is None the dataframe is returned by apply
            method as main result (i.e operator is final consumer in pipeline)
            time_unit: row index units (natural integer numbering of samples or real time units)
        """
        self.to_file = file
        self.time_unit = time_unit

    def apply(self, container: SigContainer) -> Union[SigContainer, pd.DataFrame]:
        df = pd.DataFrame(data=container.d["signals/data"].transpose(),
                          columns=container.d["signals/channels"],
                          index=container.x_index(self.time_unit, container.d["signals/fs"]))
        if self.to_file is None:
            return df
        else:
            file = self.to_file.format(container)
            df.to_pickle(file)
            return container
