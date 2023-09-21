from analyser import Reader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import tqdm
from ovito import modifiers

matplotlib.use("Agg")


class DensityField(Reader):
    """Project density and coarse grain over length dL"""

    def __init__(self):
        description = self.__doc__
        super().__init__(description)
        self.parser.add_argument("folder", type=str)
        self.parser.add_argument("--dl", type=float, default=1.0)
        self.parser.add_argument("--selection", default="None", type=str)
        self.parser.add_argument("--nohdf5", action="store_true")
        self.parser.add_argument("--map2d", default=-1, type=int)
        self.parser.add_argument("--average", action="store_true")
        super().open_pipe()

    def compute(self, axis=2):
        start = self.args.start
        end = self.args.end
        stride = self.args.stride
        ax = [0, 1, 2]
        ax.pop(axis)
        x = ax[0]
        y = ax[1]
        z = axis
        data = self.pipe.compute(start)
        self.cell = data.cell[:]
        ox, Lx = self.cell[x, -1], self.cell[x, x]
        oy, Ly = self.cell[y, -1], self.cell[y, y]
        oz, Lz = self.cell[z, -1], self.cell[z, z]
        dl = self.args.dl
        binningx = np.arange(ox, ox + Lx + dl, dl)
        binningy = np.arange(oy, oy + Ly + dl, dl)
        binningz = np.arange(oz, oz + Lz + dl, dl)

        prop = data.particles.particle_types

        if self.args.selection != "None":
            pid = prop.type_by_name(self.args.selection).id
            self.pipe.modifiers.append(
                modifiers.ExpressionSelectionModifier(expression=f"ParticleType=={pid}")
            )
            self.pipe.modifiers.append(modifiers.InvertSelectionModifier())
            self.pipe.modifiers.append(modifiers.DeleteSelectedModifier())
        # fg, ax = plt.subplots(figsize=(10,10))
        if self.args.nohdf5 == True:
            pass
        else:
            h5f = h5py.File(self.args.folder + f"/hist-data-dl{dl}.h5", "w")
            h5f.create_dataset("Ls", data=[Lx, Ly, Lz])
            h5f.create_dataset("binning_x", data=binningx)
            h5f.create_dataset("binning_y", data=binningy)
            h5f.create_dataset("binning_z", data=binningz)

        for frame in tqdm.tqdm(range(start, end, stride)):
            data = self.pipe.compute(frame)
            pos = data.particles.positions.array
            H, edges = np.histogramdd(pos, bins=[binningx, binningy, binningz])

            # write to file if requested
            if self.args.nohdf5 == True:
                pass
            else:
                h5f.create_dataset("frame_%d" % frame, data=H.astype(np.ubyte))

            # save 2d map if requested along the requeste daxis
            if self.args.map2d >= 0 and self.args.map2d < 3:
                plt.matshow(H.mean(axis=self.args.map2d))
                plt.colorbar()
                plt.savefig(self.args.folder + f"/frame_{frame:06d}.png")
                plt.close()

        if self.args.nohdf5 == True:
            pass
        else:
            h5f.close()


D = DensityField()
D.compute()
