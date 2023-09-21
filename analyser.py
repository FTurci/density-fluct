import os

os.environ["DISPLAY"] = ""
import ovito
import numpy as np
import argparse

# Note: on the cluster, you may need to set the variable DISPLAY="" for Ovito to work


def asymmetry(rho, x, normalisation):
    p = x > 0
    n = x < 0
    b = x[1] - x[0]
    return (
        np.absolute(np.trapz(rho[p], x[p], dx=b) - np.trapz(rho[n], x[n], b))
        / normalisation
    )


class Reader:
    """Base class to read and process density profiles."""

    def __init__(self, description):
        parser = argparse.ArgumentParser(description)
        parser.add_argument("path", type=str)
        parser.add_argument("--start", default=0, type=int)
        parser.add_argument("--end", default=-10, type=int)
        parser.add_argument("--stride", default=1, type=int)
        parser.add_argument("-v", "--verbose", action="store_true")
        parser.add_argument("--unzip", action="store_true")
        parser.add_argument("--zcat", action="store_true")
        self.parser = parser

    def open_pipe(self):
        self.args = self.parser.parse_args()

        if self.args.zcat == True and self.args.path.endswith(".gz"):
            print("Zcatting...", end="", flush=True)
            os.system("zcat " + self.args.path + f" > {self.args.path[:-3]}")
            # os.system("gzip -c dummy.atom > "+self.args.path)
            # print("done.")
            self.args.path = self.args.path[:-3]
        if self.args.unzip == True and self.args.path.endswith(".gz"):
            print("Unzipping...", end="", flush=True)
            os.system("gunzip -k " + self.args.path)
            self.path = self.args.path[:-3]
            print("done.")
        else:
            self.path = self.args.path

        self.pipe = ovito.io.import_file(self.path, multiple_frames=True)

        nframes = self.pipe.source.num_frames
        if self.args.end == -10:
            self.args.end = nframes

    def vprint(self, *args, **kwargs):
        if self.args.verbose == True:
            print(":v:", *args, **kwargs)

    def __del__(self):
        try:
            if self.args.unzip == True and self.args.path.endswith(".gz"):
                try:
                    os.remove(self.path)
                except OSError:
                    pass
        except Exception as e:
            print("Quitting with exception:", e)


class Quadrant(Reader):
    def __init__(self):
        description = "Analyse the density difference in the four quadrants of the plane orthogonal to z"
        super().__init__(description)
        self.parser.add_argument("-tf", "--tofile", type=str, default=None)
        super().open_pipe()

    def compute(self):
        start = self.args.start
        end = self.args.end
        stride = self.args.stride
        if self.args.tofile != None:
            fopen = open(self.args.tofile, "w")

        for frame in range(start, end, stride):
            data = self.pipe.compute(frame)
            # only get x-y
            pos = data.particles.positions.array[:, :2]
            N = pos.shape[0]
            # count in each quadrant the number of particles
            quadrant_num = []
            for i in [-1, 1]:
                for j in [-1, 1]:
                    num = sum((i * pos[:, 0] > 0) * (j * pos[:, 1] > 0))
                    quadrant_num.append(num)
            # print(quadrant_num)
            quadrant_num = np.array(quadrant_num)
            quadrant_frac = quadrant_num / N

            self.vprint(frame, quadrant_frac.ptp())
            if "fopen" in locals():
                fopen.write(
                    f"{frame} {str(quadrant_frac)[1:-1]} {quadrant_frac.ptp()}\n"
                )


class LateralProfile(Reader):
    def __init__(self):
        description = "Compute lateral density profile"
        super().__init__(description)
        self.parser.add_argument("--bin", type=float, default=0.5)
        self.parser.add_argument("-ax", "--axis", type=int, default=0)
        super().open_pipe()
        # store indices of other two dimensions
        self.others = [0, 1, 2]
        self.others.remove(self.args.axis)
        self.axis = self.args.axis

    def compute(self):
        """Accumulate density profiles"""
        start = self.args.start
        end = self.args.end
        stride = self.args.stride
        axis = self.axis

        data = self.pipe.compute(0)
        self.cell = data.cell[:]
        self.volume = self.cell[0, 0] * self.cell[1, 1] * self.cell[2, 2]
        self.num_part = data.particles.positions.array.shape[0]
        self.density = self.num_part / self.volume
        binning = np.arange(
            self.cell[axis, -1],
            self.cell[axis, axis] + self.cell[axis, -1] + self.args.bin,
            self.args.bin,
        )
        profiles = []
        for frame in range(start, end, stride):
            data = self.pipe.compute(frame)
            pos = data.particles.positions.array[:, axis]

            profile, edges = np.histogram(pos, bins=binning)
            profiles.append(profile)

        profiles = np.array(profiles)

        self.profiles = profiles
        self.x = binning[:-1] + self.args.bin / 2

    def stats(self, start=None, end=None, stride=None, normalisation_density=None):
        """Compute statistics for the profile"""
        if hasattr(self, "profiles") == False:
            self.vprint(
                "ERROR! No profile has been accumulate. Run the `compute()` method first."
            )
        else:
            if start == None:
                start = 0
            if end == None:
                end = len(self.x)
            if stride == None:
                stride = self.args.stride

            self.avg_profile = self.profiles[start:end:stride].mean(axis=0)
            self.std_profile = self.profiles[start:end:stride].std(axis=0)

            self.avg_rho_profile = self.avg_profile / (
                self.args.bin
                * self.cell[self.others[0], self.others[0]]
                * self.cell[self.others[1], self.others[1]]
            )
            if normalisation_density != None:
                normalisation = (self.density - normalisation_density) * self.cell[
                    self.axis, self.axis
                ]
            else:
                normalisation = 1

            self.asymmetry = asymmetry(self.avg_rho_profile, self.x, normalisation)
            # print("asymmetry", self.asymmetry)
