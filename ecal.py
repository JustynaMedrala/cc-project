from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math

beamAngle = 3.601e-3
EcalYShift = -0.19
EcalStart = 12520.0
EcalZSize = 825.0
EcalPosXY = EcalStart + EcalZSize / 2.0
EcalInnModLength = 690.0
EcalMidModLength = 705.0
EcalOutModLength = 705.0
EcalInnFrontCoverLength = 23.0
EcalStackLength = 432.0
EcalMidFrontCoverLength = 23.0
EcalOutFrontCoverLength = 23.0
EcalOutStackOffset = (
    -0.5 * EcalOutModLength + EcalOutFrontCoverLength + 0.5 * EcalStackLength
)
EcalInnStackOffset = (
    -0.5 * EcalInnModLength + EcalInnFrontCoverLength + 0.5 * EcalStackLength
)
EcalMidStackOffset = (
    -0.5 * EcalMidModLength + EcalMidFrontCoverLength + 0.5 * EcalStackLength
)

EcalInnOffset = 7.5
EcalMidOffset = 0.0
EcalOutOffset = 0.0

EcalInnStackStart = (
    EcalPosXY + EcalInnOffset + EcalInnStackOffset - 0.5 * EcalStackLength
)
EcalMidStackStart = (
    EcalPosXY + EcalMidOffset + EcalMidStackOffset - 0.5 * EcalStackLength
)
EcalOutStackStart = (
    EcalPosXY + EcalOutOffset + EcalOutStackOffset - 0.5 * EcalStackLength
)

EcalSteelOffset = -215.50
EcalPlasticOffset = -211.5
EcalScOffset = -205.88
EcalPbOffset = -202.76
EcalOffset = -215.50

EcalSteelThick = 1.0
EcalPlasticThick = 7.0
EcalPbThick = 2.0
EcalScThick = 4.0
EcalPaperThick = 0.12
EcalSandwichThick = EcalScThick + EcalPbThick + 2 * EcalPaperThick

# EcalInnSteelStart = EcalInnStackStart + EcalStackLength / 2.0 + EcalSteelOffset - EcalSteelThick / 2.0
# EcalPlasticStart = EcalInnStackStart + EcalStackLength / 2.0 + EcalPlasticOffset - EcalPlasticThick / 2.0
# EcalPbStart = EcalInnStackStart + EcalStackLength / 2.0 + EcalPbOffset - EcalPbThick / 2.0
EcalScStart = {
    "inner": EcalInnStackStart
    + EcalStackLength / 2.0
    + EcalScOffset
    - EcalScThick / 2.0,
    "middle": EcalMidStackStart
    + EcalStackLength / 2.0
    + EcalScOffset
    - EcalScThick / 2.0,
    "outer": EcalOutStackStart
    + EcalStackLength / 2.0
    + EcalScOffset
    - EcalScThick / 2.0,
}
EcalModXYSize = 121.9
BOUNDARIES = {
    "middle": {
        "x": {"min": -32 * EcalModXYSize / 2.0, "max": 32 * EcalModXYSize / 2.0},
        "y": {"min": -20 * EcalModXYSize / 2.0, "max": 20 * EcalModXYSize / 2.0},
        "z": {"min": EcalMidStackStart, "max": EcalMidStackStart + EcalStackLength},
    },
    "inner": {
        "x": {"min": -16 * EcalModXYSize / 2.0, "max": 16 * EcalModXYSize / 2.0},
        "y": {"min": -12 * EcalModXYSize / 2.0, "max": 12 * EcalModXYSize / 2.0},
        "z": {"min": EcalInnStackStart, "max": EcalInnStackStart + EcalStackLength},
    },
    "beam": {
        "x": {
            "min": -6 * EcalModXYSize / 2.0 + EcalModXYSize / 3.0,
            "max": 6 * EcalModXYSize / 2.0 - EcalModXYSize / 3.0,
        },
        "y": {"min": -4 * EcalModXYSize / 2.0, "max": 4 * EcalModXYSize / 2.0},
    },
    "outer": {
        "x": {"min": -64 * EcalModXYSize / 2.0, "max": 64 * EcalModXYSize / 2.0},
        "y": {"min": -52 * EcalModXYSize / 2.0, "max": 52 * EcalModXYSize / 2.0},
        "z": {"min": EcalOutStackStart, "max": EcalOutStackStart + EcalStackLength},
    },
}

UNIT_CELL_SIZE = EcalModXYSize / 6.0


class EcalImage:

    @staticmethod
    def prepare_dimension_columns(dimension, val):
        dim_cols = []
        for i in range(val * 6):
            dim_cols.append(BOUNDARIES["outer"][dimension]["min"] + i * UNIT_CELL_SIZE)
        return dim_cols

    @staticmethod
    def to_pxl(cell, axis):
        return (cell - BOUNDARIES["outer"][axis]["min"]) / (
            BOUNDARIES["outer"][axis]["max"] - BOUNDARIES["outer"][axis]["min"]
        )

    def fill_cells_with_energy(self):
        # FIXME: don't like this zipping of colums, but it seems to be the fastest for now
        for cell_x, cell_y, cell_size, energy in zip(
            self.hits_pickle["Cell_X"],
            self.hits_pickle["Cell_Y"],
            self.hits_pickle["Cell_Size"],
            self.hits_pickle[self.value_column],
        ):

            pxl_x = round(self.to_pxl(cell_x - cell_size / 2.0, "x") * 384.0)
            pxl_y = round(self.to_pxl(cell_y - cell_size / 2.0, "y") * 312.0)
            pxl_size = int(cell_size / EcalModXYSize * 6)
            self.E[pxl_y : pxl_y + pxl_size, pxl_x : pxl_x + pxl_size] += energy

    def __init__(self, hits_pickle, value_column="Active_Energy"):
        self.hits_pickle = hits_pickle
        self.value_column = value_column
        if isinstance(hits_pickle, str):
            self.hits_pickle = pd.read_pickle(hits_pickle)
        x = self.prepare_dimension_columns("x", 64)
        y = self.prepare_dimension_columns("y", 52)
        self.X, self.Y = np.meshgrid(x, y)
        self.E = np.zeros((len(y), len(x)))
        self.fill_cells_with_energy()
        self.E[self.E <= 0.0] = np.nan

    def plot_links(self, ax, df, **kwargs):
        tdf = (
            df[["Particle_Index", self.value_column]]
            .groupby(by="Particle_Index")
            .max()
            .copy()
        )
        tdf = df.merge(
            tdf,
            left_on="Particle_Index",
            right_on="Particle_Index",
            suffixes=("", "_Max"),
        )
        tdf = tdf[tdf[self.value_column] == tdf["Active_Energy_Max"]]
        for x_part, y_part, x_cell, y_cell in zip(
            tdf["Entry_X"], tdf["Entry_Y"], tdf["Cell_X"], tdf["Cell_Y"]
        ):

            ax.plot([x_part, x_cell], [y_part, y_cell], **kwargs)

    def plot_points(self, ax, df, position_type="", size=20, **kwargs):
        for x_pos, y_pos in zip(
            df["Position X" + position_type], df["Position Y" + position_type]
        ):
            point = plt.Circle((x_pos, y_pos), size, **kwargs)
            ax.add_artist(point)

    def plot_grid(
        self,
        ax,
        cmap=plt.get_cmap("viridis"),
        module_boundary_color="grey",
        rasterized=False,
        nan_color=None,
        **kwargs
    ):
        ax.set_xlim(BOUNDARIES["outer"]["x"]["min"], BOUNDARIES["outer"]["x"]["max"])
        ax.set_ylim(BOUNDARIES["outer"]["y"]["min"], BOUNDARIES["outer"]["y"]["max"])
        for module in ["middle", "inner", "beam"]:
            ax.add_artist(
                plt.Rectangle(
                    (BOUNDARIES[module]["x"]["min"], BOUNDARIES[module]["y"]["min"]),
                    height=(
                        BOUNDARIES[module]["y"]["max"] - BOUNDARIES[module]["y"]["min"]
                    ),
                    width=(
                        BOUNDARIES[module]["x"]["max"] - BOUNDARIES[module]["x"]["min"]
                    ),
                    color=module_boundary_color,
                    linewidth=1,
                    fill=False,
                )
            )
        if nan_color:
            cmap.set_bad(nan_color, 1.0)
        return ax.pcolormesh(
            self.X, self.Y, self.E, cmap=cmap, rasterized=rasterized, **kwargs
        )






