from time import time
from pathlib import Path
import pandas as pd


def create_range_bars(tick_df, range_size):
    """
    Создает Range бары из тикового датафрейма.

    Parameters:
        tick_df (pd.DataFrame): Датафрейм с тиковыми данными
        (колонки: 'datetime', 'last', 'volume').
        range_size (float): Размер диапазона для Range баров.

    Returns:
        pd.DataFrame: Датафрейм с Range барами
        (колонки: 'datetime', 'open', 'high', 'low', 'close', 'volume').
    """
    # Инициализация переменных
    range_bars = []
    open_price = None
    high_price = None
    low_price = None
    vol = 0
    bar_start_time = None  # Время открытия текущего бара

    for _, row in tick_df.iterrows():
        price = row['last']
        volume = row['volume']
        datetime = row['datetime']

        # Если новый бар, инициализируем его
        if open_price is None:
            open_price = price
            high_price = price
            low_price = price
            bar_start_time = datetime  # Устанавливаем время первого тика

        # Обновляем high, low и объем
        high_price = max(high_price, price)
        low_price = min(low_price, price)
        vol += volume

        # Проверяем, превышен ли range_size
        if high_price - low_price >= range_size:
            # Закрываем текущий бар
            range_bars.append({
                'datetime': bar_start_time,  # Используем время первого тика в баре
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': price,
                'volume': vol
            })
            # Инициализируем следующий бар
            open_price = price
            high_price = price
            low_price = price
            vol = 0
            bar_start_time = datetime  # Обновляем время начала нового бара

    # Добавляем последний бар, если он не завершен
    if open_price is not None:
        range_bars.append({
            'datetime': bar_start_time,  # Используем время первого тика в баре
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': price,
            'volume': vol
        })

    return pd.DataFrame(range_bars)


if __name__ == "__main__":
    start_time = time()  # Время начала запуска скрипта

    directory = Path(r'c:\data_quote\data_finam_RTS_tick_zip')
    range_size_lst = [100, 150, 200, 250, 300, 350, 400, 450, 500]

    files = list(directory.glob('*.zip'))

    for file_1, file_2 in zip(files, files[1:]):
        df_tick_1 = pd.read_csv(file_1, compression='zip', parse_dates=['datetime'])
        range_size_dic = {}
        for range_size in range_size_lst:
            df_range_1 = create_range_bars(df_tick_1, range_size)
            range_size_dic[range_size] = len(df_range_1)
        print(file_1, file_2)
        print(range_size_dic)

    # print(files)

    # print(df)
    print(f'Скрипт выполнен за {(time() - start_time):.2f} с')