from __future__ import annotations

from dataclasses import dataclass, field
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
    'um': -6, 'μm': -6,
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

    value: float
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

    def __init__(self, value: float, symbol: str = 'px'):
        self.value = value
        self.symbol = symbol
        self.factor = unit_to_factor[symbol]

    def __str__(self):
        return f'{self.value}{self.symbol}'
    
    def equal_instance(self, other: 'DUnit') -> bool:
        return self.value == other.value and self.symbol == other.symbol

    def __eq__(self, other: 'DUnit') -> bool:
        if self.factor == other.factor:
            return self.value == other.value
        return self.to_float() == other.to_float()

    def __gt__(self, other: 'DUnit') -> bool:
        return self.to_float().__gt__(other.to_float)

    def __ge__(self, other: 'DUnit') -> bool:
        if self.__eq__(other):
            return True
        return self.__gt__(other)
    
    def __lt__(self, other: 'DUnit') -> bool:
        return self.to_float().__lt__(other.to_float())
    
    def __le__(self, other: 'DUnit') -> bool:
        return self.to_float().__le__(other.to_float())

    def scale(self, scale_factor: float, inplace: bool = True) -> 'DUnit' | None:
        if inplace:
            self.value = self.value * scale_factor
        else:
            return DUnit(self.value * scale_factor, self.symbol)

    def to_float(self) -> float:
        return self.value * math.pow(10, self.factor)
    
    @staticmethod
    def __verify_symbol(symbol: str):
        if symbol == 'px':
            raise Exception(f'px cannot be used in coordination with other symbols. Change symbol first to SI unit.')

    def convert_symbol(self, symbol: str) -> 'DUnit':
        self.__verify_symbol(symbol)
        self.__verify_symbol(self.symbol)
        new_factor = unit_to_factor[symbol]
        rate = math.pow(10, new_factor) / math.pow(10, self.factor)
        new_value = self.value * rate
        return DUnit(new_value, symbol)

    def get_conversion_factor(self, target: 'DUnit') -> float:
        self.__verify_symbol(target.symbol)
        self.__verify_symbol(self.symbol)        
        src_flt = self.to_float()
        target_flt = target.to_float()
        conv_rate = target_flt / src_flt
        return conv_rate

    @classmethod
    def default_dunit(cls):
        return cls(value = 1, symbol = 'px')
    
    def to_json(self) -> dict[str, float | str]:
        return {
            'value': self.value,
            'symbol': self.symbol
        }
    
    @classmethod
    def from_dict(cls, dct: dict[str, float | str]) -> 'DUnit':
        return cls(
            value = dct['value'],
            symbol= dct['symbol']
        )
