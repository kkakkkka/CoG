from dataclasses import dataclass
import re


@dataclass
class Interval:
    minTime: float
    maxTime: float
    mark: str


class TextGrid(list):
    @classmethod
    def fromFile(cls, path):
        tiers = []
        current = None
        xmin = None
        xmax = None
        text = None
        in_interval = False

        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if re.match(r"item \[\d+\]:", line):
                    if current is not None:
                        tiers.append(current)
                    current = []
                    continue
                if re.match(r"intervals \[\d+\]:", line):
                    in_interval = True
                    xmin = xmax = text = None
                    continue
                if not in_interval:
                    continue
                if line.startswith("xmin ="):
                    xmin = float(line.split("=", 1)[1].strip())
                elif line.startswith("xmax ="):
                    xmax = float(line.split("=", 1)[1].strip())
                elif line.startswith("text ="):
                    value = line.split("=", 1)[1].strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    text = value
                    if current is not None and xmin is not None and xmax is not None:
                        current.append(Interval(xmin, xmax, text))
                    in_interval = False

        if current is not None:
            tiers.append(current)

        grid = cls()
        grid.extend(tiers)
        return grid
