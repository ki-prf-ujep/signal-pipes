from sigpipes.auxtools import TimeUnit
from sigpipes.sigcontainer import SigContainer
from sigpipes.sigoperator import MaybeConsumerOperator
from sigpipes.auxtools import CyclicList
from sigpipes.auxtools import common_value

from dataclasses import dataclass
from typing import Sequence, Union, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib


@dataclass
class GraphOpts:
    """
    Simple box for parameters of graphs in plot operators.
    Some parameters have simple global value which has to be shared
    among all graphs, others are configurable on per graph basis by list
    of values. If you want create list with repeating values (or event with
    repeating single value) use cyclic list.
    """
    columns: int = 1
    title: CyclicList = CyclicList(["{i}. {signals}"])
    graph_width: float = 14
    graph_height: float = 2.5
    grid: CyclicList = CyclicList([True])
    time_unit: TimeUnit = TimeUnit.SECOND
    legend_loc: str = 'upper right'


@dataclass
class SignalOpts:
    """
    Simple box for signal (channels) plotted by plot operators.
    Some parameters have simple global value which has to be shared
    among all signals, others are configurable on per signal basis by list
    of values. If you want create list with repeating values (or event with
    repeating single value) use cyclic list.
    """
    colors: CyclicList = CyclicList(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    styles: CyclicList = CyclicList(["-"])
    legend: bool = True


@dataclass
class AnnotOpts:
    """
    Simple box for annotation points plotted by plot operators.
    Some parameters have simple global value which has to be shared
    among all annotations, others are configurable on per annotation basis by list
    of values. If you want create list with repeating values (or event with
    repeating single value) use cyclic list.
    """
    colors: CyclicList = CyclicList(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    styles: CyclicList = CyclicList("ovx^+")
    vertical_shift : float = 0.1
    legend: bool = True
    vertical_line = True


class GraphOrganizer:
    """
    Auxiliary class for organizing subgraphs in M×N matrix (where N = fixed count of columns]
    """
    def __init__(self, size: int, columns: int):
        assert size >= columns, f"Number of subplots ({size}) is smaller then number of columns ({columns})"
        self.size = size
        self.rows = (size - 1) // columns + 1
        self.columns = columns
        self.fig, self.axes = plt.subplots(self.rows, self.columns, sharex="all")
        for i in range(self.size, self.rows * self.columns):
            self._get_axes(i).remove()
        self.fig.subplots_adjust(left=0.1/self.columns, right=1-0.1/self.columns)

    @property
    def dim(self) -> int:
        if self.rows == 1 and self.columns == 1:
            return 0
        if self.rows == 1 or self.columns == 1:
            return 1
        return 2

    def _get_axes(self, item: int) -> Axes:
        if self.dim == 0:
            assert item == 0
            return self.axes
        if self.dim == 1:
            return self.axes[item]
        return self.axes[self.row_of(item)][self.column_of(item)]

    def __getitem__(self, item: int) -> Axes:
        assert 0 <= item < self.size, "Invalid index"
        return self._get_axes(item)

    def row_of(self, item: int) -> int:
        return item % self.rows

    def column_of(self, item: int) -> int:
        return item // self.rows


class BasePlot(MaybeConsumerOperator):
    """
    Abstract base class for plotting operators)
    """
    def __init__(self,
                 graph_specs: Optional[Iterable[Union[int, Iterable[int]]]] = None,
                 *,
                 file: str = None,
                 graph_opts: GraphOpts = GraphOpts(),
                 signal_opts: SignalOpts = SignalOpts()) -> None:
        self.graph_signals = graph_specs
        self.graph_option = graph_opts
        self.signal_option = signal_opts
        self.to_file = file

    def apply(self, container: SigContainer) -> Union[SigContainer, Figure]:
        fig = self.plot(container)
        if self.to_file is None:
            return fig
        else:
            file = self.to_file.format(container)
            fig.savefig(file)
            plt.close(fig)
            return container

    def _fix_graph_signals(self, container: SigContainer) -> None:
        fcol = []
        for item in self.graph_signals:
            if isinstance(item, int):
                fcol.append((item,))
            elif len(item) == 0:
                fcol.append(range(container.channel_count))
            else:
                fcol.append(item)
        self.graph_signals = fcol

    def _plot_signals(self, axes: GraphOrganizer, container: SigContainer) -> None:
        raise NotImplementedError("Abstract method")

    def plot(self, container: SigContainer) -> Figure:
        if self.graph_signals is None:
            self.graph_signals = range(container.channel_count)
        size = len(self.graph_signals)

        ax = GraphOrganizer(size, self.graph_option.columns)
        ax.fig.set_size_inches(ax.columns * self.graph_option.graph_width,
                               ax.rows * self.graph_option.graph_height)
        self._fix_graph_signals(container)
        self._plot_signals(ax, container)
        return ax.fig


class FftPlot(BasePlot):
    """
    Plot spectrum of signal (i.e. output of Fft operator)
    """
    def __init__(self,
                 source: str = "fft",
                 graph_specs: Optional[Iterable[Union[int, Iterable[int]]]] = None,
                 *,
                 file: str = None,
                 graph_opts: GraphOpts = GraphOpts(),
                 signal_opts: SignalOpts = SignalOpts()) -> None:
        super().__init__(graph_specs, file=file, graph_opts=graph_opts,
                         signal_opts=signal_opts)
        self.source = "meta/" + source

    def _plot_signals(self, axes: GraphOrganizer, container: SigContainer) -> None:
        for i, group in enumerate(self.graph_signals):
            signals = []
            for index in group:
                y, signal_name = container.get_fft_tuple(index, self.source)
                signal_name = signal_name.strip(",.:")
                x = container.x_index(self.graph_option.time_unit.fix(),
                                      container.d[f"{self.source}/fs"])
                axes[i].plot(x, y, self.signal_option.styles[index],
                             color=self.signal_option.colors[index],
                             label=signal_name if self.signal_option.legend else None)
                signals.append(signal_name)
            axes[i].grid(self.graph_option.grid[i])
            axes[i].set_ylabel("")
            xlabel = f"frequency"
            if axes.row_of(i) == axes.rows - 1:
                axes[i].set_xlabel(xlabel)
            axes[i].set_title(self.graph_option.title[i].format(i=i+1, signals=", ".join(signals)))


class Plot(BasePlot):
    def __init__(self,
                 graph_specs: Optional[Iterable[Union[int, Iterable[int]]]] = None,
                 annot_specs: Optional[Sequence[Optional[Union[str, Iterable[str]]]]] = None,
                 *,
                 file: str = None,
                 graph_opts: GraphOpts = GraphOpts(),
                 signal_opts: SignalOpts = SignalOpts(),
                 annot_opts: AnnotOpts = AnnotOpts()) -> None:
        """
        Args:
            graph_specs: specification of signals for (sub)graphs
                - None (optional): one subgraph per each signals = [0, 2, ..., n-1]
                - [()]: one graph with all signals = [(0, 2, ..., n-1)]
                - [0,1,(3,4)]: three subgraphs, the first graph with signal 0,
                  the second graph with signal 1,
                  the third graph with two signals (3 and 4)
                - [(), 1]: two subgraphs, first with all signals, second with second (=1) signal
            annot_specs: specification of added annotations
                - None (optional): none annotation in any subgraph
                - ["x"]: annotation "x" in the first subgraph
                - [("x", "y")]: annotations "a" and "y" in the first subgraph
                - ["x", ("y", "z")]: annotation "x" in the first subgraph,
                   annotations "y" and "z" in the second subgraph
                - [None, None, "x"]: annotation "x" in the third subgraph
            file: file name of target image file or None (application return directly
                  matplotlib figure)
        """
        super().__init__(graph_specs, file=file, graph_opts=graph_opts,
                         signal_opts=signal_opts)
        self.graph_annotations = annot_specs
        self.annotation_option = annot_opts

    def _plot_signals(self, axes: GraphOrganizer, container: SigContainer) -> None:
        if self.graph_annotations is None:
            self.graph_annotations = [()] * len(self.graph_signals)
        for i, group in enumerate(self.graph_signals):
            signals = []
            units = []
            for index in group:
                y, signal_name, signal_unit = container.get_channel_triple(index)
                signal_name = signal_name.strip(",.:")
                x = container.x_index(self.graph_option.time_unit.fix(), container.d["signals/fs"])
                axes[i].plot(x, y, self.signal_option.styles[index],
                             color=self.signal_option.colors[index],
                             label=signal_name if self.signal_option.legend else None)
                signals.append(signal_name)
                units.append(signal_unit)
            common_unit = common_value(units)
            axes[i].grid(self.graph_option.grid[i])
            axes[i].set_ylabel(common_unit)
            xlabel = f"time/{'s' if self.graph_option.time_unit != TimeUnit.SAMPLE else 'samples'}"
            if axes.row_of(i) == axes.rows - 1:
                axes[i].set_xlabel(xlabel)
            axes[i].set_title(self.graph_option.title[i].format(i=i+1, signals=", ".join(signals)))

            if i < len(self.graph_annotations):
                self._plot_annotations(container, axes[i], self.graph_annotations[i])

    def _plot_annotations(self, container: SigContainer, axis: matplotlib.axes.Axes,
                          annot: Optional[Union[str, Iterable[str]]]) -> None:
        if annot is None:
            return
        if isinstance(annot, str):
            annot = (annot,)
        bottom, top = axis.get_ylim()
        for i, a in enumerate(annot):
            x = container.get_annotation_positions(a, self.graph_option.time_unit.fix(),
                                                   container.d["signals/fs"])
            y = np.empty_like(x)
            y.fill(top + i*(top-bottom) * self.annotation_option.vertical_shift)
            axis.plot(x, y, self.annotation_option.styles[i],
                      color=self.annotation_option.colors[i],
                      label=a if self.annotation_option.legend else None)
            if self.annotation_option.vertical_line:
                axis.vlines(x, bottom,
                            top + i * (top - bottom) * self.annotation_option.vertical_shift)

