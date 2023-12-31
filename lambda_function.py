from strategies_maker import StrategiesMaker
from datetime import datetime

strategy_maker = StrategiesMaker('PRIO3.SA', datetime(2024, 5, 31), 0.115, 0.385)

print(strategy_maker.calculate_short_strangle_greeks())