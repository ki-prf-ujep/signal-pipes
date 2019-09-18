from sigpipes.sigcontainer import SigContainer
from sigpipes.sigoperator import SimpleBranching
from multiprocessing import Pool


def fmap(branch, container):
    container | branch


_pool = Pool()


class ParTee(SimpleBranching):
    """
    Tee parallel branching operator. For each parameters of constructor the container is duplicated
    and processed in parallel by pipeline passed by this parameter
    (i.e. all pipelines have the same source,
    but they are independent). Only original container is returned (i.e. only one streamm continues)
    """
    def __init__(self, *branches):
        """
        Args:
            *branches:  one or more parameters in the form of signals operators (including whole
            pipelines in the form of compound operator)
        """
        super().__init__(*branches)

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        _pool.starmap(fmap, ((branch, container) for branch in self.branches))
        return container

    def log(self):
        return "#PTEE"