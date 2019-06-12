from sigpipes.sigoperator import SigOperator
from sigpipes.sigcontainer import SigContainer, TimeUnit
from typing import Union, Iterable

import pandas as pd


class DataFrame(SigOperator):
    def __init__(self, file: str = None, time_unit: TimeUnit = TimeUnit.SECOND):
        self.to_file = file
        self.time_unit = time_unit

    def apply(self, container: SigContainer) -> Union[SigContainer, pd.DataFrame]:
        df = pd.DataFrame(data=container.d["signals/data"].transpose(),
                          index=container.x_index(self.time_unit, container.d["signals/fs"]))
        if self.to_file is None:
            return df
        else:
            file = self.to_file.format(container)
            df.to_pickle(file)
            return container
