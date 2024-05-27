from dataclasses import dataclass
from enum import IntEnum


@dataclass
class WearType(IntEnum):
    accessoire = 1
    innerWear = 2
    outerWear = 3
    bottomWear = 4
    shoes = 5
