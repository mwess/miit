from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import math

unit_to_factor = {
    'Qm': 30,
    'Rm': 27,
    'Ym': 24,
    'Zm': 21,
    'Em': 18,
    'Pm': 15,
    'Tm': 12,
    'Gm': 9,
    'Mm': 6,
    'km': 3,
    'hm': 2,
    'dam': 1,
    'm': 0,
    'dm': -1,
    'cm': -2,
    'mm': -3,
    'um': -6, 'μm': -6, 'µm': -6,
    'nm': -9,
    'pm': -12,
    'fm': -15,
    'am': -18,
    'zm': -21,
    'ym': -24,
    'rm': -27,
    'qm': -30,
    'px': None
}


@dataclass
class DUnit:
    """Class to keep track of images resolution. Used to denote a space resolution per pixel.

    Attributes
    ----------
    
    value (Decimal): Prefix of distance unit.
    
    symbol (str): String description of resolution.
    
    factor (int | None): Unit factor to denote resolution.
    """

    value: Decimal
    symbol: str = field(init=False, default='px')
    factor: int | None = field(init=False, default=None)

    @property
    def symbol(self) -> str:
        return self._symbol
    
    @symbol.setter
    def symbol(self, symbol: str):
        self._symbol = symbol
        self._factor = unit_to_factor[symbol]

    @property
    def factor(self) -> int:
        return self._factor
    
    @factor.setter
    def factor(self, factor: int):
        if factor not in unit_to_factor.values():
            raise Exception(f'Factor {factor} unkown.')
        self._factor = factor
        factor_to_unit = {value: key for (key, value) in unit_to_factor.items()}
        self._symbol = factor_to_unit[factor]

    def __init__(self, value: str | float | Decimal, symbol: str = 'px'):
        value = DUnit.to_decimal(value)
        self.value = value
        self.symbol = symbol
        self.factor = unit_to_factor[symbol]

    def __str__(self):
        return f'{float(self.value)}{self.symbol}'
    
    def equal_instance(self, other: 'DUnit') -> bool:
        return self.value == other.value and self.symbol == other.symbol

    def __eq__(self, other: 'DUnit') -> bool:
        if self.factor == other.factor:
            return self.value == other.value
        return self.to_dec() == other.to_dec()

    def __gt__(self, other: 'DUnit') -> bool:
        return self.to_dec().__gt__(other.to_dec)

    def __ge__(self, other: 'DUnit') -> bool:
        if self.__eq__(other):
            return True
        return self.__gt__(other)
    
    def __lt__(self, other: 'DUnit') -> bool:
        return self.to_dec().__lt__(other.to_dec())
    
    def __le__(self, other: 'DUnit') -> bool:
        return self.to_dec().__le__(other.to_dec())

    @staticmethod
    def to_decimal(value: float | str | Decimal) -> Decimal:
        """Converts value to a decimal.

        Args:
            value (float | str | Decimal):

        Returns:
            Decimal:
        """
        if isinstance(value, float):
            value = str(value)
        if isinstance(value, str):
            value = Decimal(value)
        return value

    def scale(self, scale_factor: float | Decimal, inplace: bool = True) -> 'DUnit' | None:
        """Scale a dunit by a given factor.

        Returns:
            DUnit | None: Returns a scaled DUnit if `inplace` == False.
        """
        scale_factor = DUnit.to_decimal(scale_factor)
        if inplace:
            self.value = self.value * scale_factor
        else:
            return DUnit(self.value * scale_factor, self.symbol)

    def to_dec(self) -> Decimal:
        """Computes resolution as one decimal.

        Returns:
            Decimal:
        """
        return (self.value * Decimal(math.pow(10, self.factor))).quantize(Decimal('0.00000000001'))
    
    def to_float(self) -> float:
        """Computes resolution as float.

        Returns:
            float:
        """
        return float(self.to_dec())
    
    @staticmethod
    def __verify_symbol(symbol: str):
        """Checks whether symbol is a SI unit.

        Args:
            symbol (str):

        Raises:
            Exception: Throws exception if symbol is pixel.
        """
        if symbol == 'px':
            raise Exception(f'px cannot be used in coordination with other symbols. Change symbol first to SI unit.')

    def convert_to_unit(self, symbol: str) -> 'DUnit':
        """Convert from one SI unit to another.

        Args:
            symbol (str):

        Returns:
            DUnit:
        """
        self.__verify_symbol(symbol)
        self.__verify_symbol(self.symbol)
        new_factor = unit_to_factor[symbol]
        rate = Decimal(math.pow(10, self.factor)) / Decimal(math.pow(10, new_factor))
        rate = rate.quantize(Decimal('0.00000000001'))
        new_value = self.value * rate
        return DUnit(new_value, symbol)

    def get_conversion_factor(self, target: 'DUnit') -> Decimal:
        """Compute a conversion factor between two DUnits.

        Args:
            target (DUnit):

        Returns:
            Decimal:
        """
        self.__verify_symbol(target.symbol)
        self.__verify_symbol(self.symbol)        
        src_flt = self.to_dec()
        target_flt = target.to_dec()
        conv_rate = target_flt / src_flt
        conv_rate = conv_rate.quantize(Decimal('0.00000000001'))
        return conv_rate

    @classmethod
    def default_dunit(cls):
        return cls(value = 1, symbol = 'px')
    
    def to_json(self) -> dict[str, float | str]:
        return {
            'value': str(self.value),
            'symbol': self.symbol
        }
    
    @classmethod
    def from_dict(cls, dct: dict[str, float | str]) -> 'DUnit':
        value = dct['value']
        if isinstance(value, str):
            value = Decimal(value)
        return cls(
            value = value,
            symbol= dct['symbol']
        )
