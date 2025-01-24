from time import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import finplot as fplt

fplt.display_timezone = timezone.utc  # Настройка тайм зоны, чтобы не было смещения времени


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


def adaptive_laguerre_filter(df, alpha=0.4):
    """
    Добавляет колонку 'alf' в DataFrame с расчетом Adaptive Laguerre Filter (ALF).

    :param df: DataFrame с колонками ['datetime', 'open', 'close', 'high', 'low', 'volume']
    :param alpha: Коэффициент сглаживания (от 0 до 1)
    :return: DataFrame с добавленной колонкой 'alf'
    """
    # Проверка наличия необходимых колонок
    required_columns = {'datetime', 'open', 'close', 'high', 'low', 'volume'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame должен содержать колонки: {required_columns}")

    # Создаем массив для хранения значений ALF
    n = len(df)
    alf_values = np.zeros(n)

    # Используем 'close' как входные данные для ALF
    prices = df['close'].values

    # Инициализация значений Laguerre
    L0 = L1 = L2 = L3 = 0

    for i in range(n):
        price = prices[i]

        L0_old = L0
        L1_old = L1
        L2_old = L2

        L0 = alpha * price + (1 - alpha) * L0_old
        L1 = -(1 - alpha) * L0 + L0_old + (1 - alpha) * L1_old
        L2 = -(1 - alpha) * L1 + L1_old + (1 - alpha) * L2_old
        L3 = -(1 - alpha) * L2 + L2_old + (1 - alpha) * L3

        alf_values[i] = (L0 + 2 * L1 + 2 * L2 + L3) / 6

    # Добавляем колонку 'alf' в DataFrame
    df['alf'] = alf_values
    df['alf'] = df['alf'].fillna(method='ffill')  # Заполнение пропусков предыдущими значениями
    return df


def volume_stops(df):
    # Инициализация новых колонок значением None
    df['long1'] = None
    df['short1'] = None
    df['long2'] = None
    df['short2'] = None

    # Проверка условий для каждой строки
    for i in range(2, len(df)):
        if (df['volume'][i - 2] < df['volume'][i - 1] < df['volume'][i] and
                df['open'][i - 2] >= df['close'][i - 2] and df['open'][i - 1] >= df['close'][i - 1] and
                df['open'][i] <= df['close'][i]):
            df.at[i, 'long1'] = df['low'][i] - 40

        if (df['volume'][i - 2] < df['volume'][i - 1] < df['volume'][i] and
                df['open'][i - 2] <= df['close'][i - 2] and df['open'][i - 1] <= df['close'][i - 1] and
                df['open'][i] >= df['close'][i]):
            df.at[i, 'short1'] = df['high'][i] + 40

        if (df['volume'][i - 2] > df['volume'][i - 1] > df['volume'][i] and
                df['open'][i - 2] >= df['close'][i - 2] and df['open'][i - 1] >= df['close'][i - 1] and
                df['open'][i] <= df['close'][i]):
            df.at[i, 'long2'] = df['low'][i] - 40

        if (df['volume'][i - 2] > df['volume'][i - 1] > df['volume'][i] and
                df['open'][i - 2] <= df['close'][i - 2] and df['open'][i - 1] <= df['close'][i - 1] and
                df['open'][i] >= df['close'][i]):
            df.at[i, 'short2'] = df['high'][i] + 40

    return df


if __name__ == "__main__":
    # start_time = time()  # Время начала запуска скрипта

    directory = Path(r'c:\data_quote\data_finam_RTS_tick_zip')
    range_bar_in_day = 300
    range_size_lst = [150, 200, 250, 300, 350, 400, 450]
    alpha = 0.4

    files = list(directory.glob('*.zip'))

    # Перебираем файлы zip перекрывающимися парами файлов
    for file_1, file_2 in zip(files, files[1:]):
        start_time = time()  # Время начала запуска скрипта
        # Чтение тиков в DF'ы из zip файлов
        df_tick_1 = pd.read_csv(file_1, compression='zip', parse_dates=['datetime'])
        df_tick_2 = pd.read_csv(file_2, compression='zip', parse_dates=['datetime'])

        range_size_dic = {}  # Словарь зависимости количества баров от размера range
        # Перебираем размерности range баров
        for range_size in range_size_lst:
            df_range_1 = create_range_bars(df_tick_1, range_size)
            # Заполняем словарь размерности, количеством баров
            range_size_dic[range_size] = len(df_range_1)

        # Вычисление размерности range баров к заданному количеству
        closest_key = min(range_size_dic, key=lambda k: abs(range_size_dic[k] - 300))
        # print(closest_key)

        # Создание DF'ов c range барами
        df_range_1 = create_range_bars(df_tick_1, closest_key)
        df_range_2 = create_range_bars(df_tick_2, closest_key)

        # Слияние датафреймов
        df = pd.concat([df_range_1, df_range_2])
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Сброс индекса (переиндексация)
        df = df.reset_index(drop=True)

        # Добавление индикатора ALF
        df = adaptive_laguerre_filter(df, alpha=alpha)
        # print(df)

        # Добавление индикатора Volume Stops
        df = volume_stops(df)

        # Извлечение имени файла без расширения (дата для выборки).
        file_name = file_2.stem

        # Выбор строк, соответствующих заданной дате
        df = df.loc[df['datetime'].dt.date == pd.to_datetime(file_name).date()]
        # print(df)
        # break

        # print(df.dtypes)  # Вывод типов данных
        # print(df['alf'].head())  # Посмотрите на первые значения в колонке alf

        df = df.set_index('datetime')

        # create two axes
        ax = fplt.create_plot(f'RTS range{closest_key}', rows=1)
        ax.set_visible(xgrid=True, ygrid=True)

        # plot candle sticks
        candles = df[['open', 'close', 'high', 'low']]
        # candles = df[['datetime', 'open', 'close', 'high', 'low']]
        fplt.candlestick_ochl(candles, ax=ax)

        # overlay volume on the top plot
        volumes = df[['open','close','volume']]
        # volumes = df[['datetime', 'open', 'close', 'volume']]
        fplt.volume_ocv(volumes, ax=ax.overlay())

        # put an ALF on the close price
        fplt.plot(df['alf'], ax=ax, legend=f'ALF-{alpha}')  # ax=ax,
        # fplt.plot(df['datetime'], df['alf'], legend=f'ALF-{alpha}')  # ax=ax,

        # Volume Stops
        fplt.plot(df['long1'], legend='Long 1 Max volume', style='o', color='#00f')
        # fplt.plot(df['datetime'], df['long1'], legend='Long 1 Max volume', style='o', color='#00f')
        # fplt.plot(df['datetime'], df['long2'], legend='Long 2 Min Volume', style='o', color='#f00')
        # fplt.plot(df['datetime'], df['long2'], legend='Long 2 Min Volume', style='o', color='#dc143c')
        fplt.plot(df['long2'], legend='Long 2 Min Volume', style='o', color='#006400')
        # fplt.plot(df['datetime'], df['long2'], legend='Long 2 Min Volume', style='o', color='#006400')
        fplt.plot(df['short1'], legend='Short 1 Max volume', style='o', color='#00f')
        # fplt.plot(df['datetime'], df['short1'], legend='Short 1 Max volume', style='o', color='#00f')
        fplt.plot(df['short2'], legend='Short 2 Min Volume', style='o', color='#006400')
        # fplt.plot(df['datetime'], df['short2'], legend='Short 2 Min Volume', style='o', color='#006400')

        fplt.show()

        # # Сохранение графика в файл
        fplt.screenshot(open(fr'chart\{file_name}.png', 'wb'))
        # ax.vb.win.grab().save(fr'chart\{file_name}.png')

        print(fr'chart\{file_name}.png')
        print(f'Скрипт выполнен за {(time() - start_time):.2f} с')

    # print(files)

    # print(df)
    # print(f'Скрипт выполнен за {(time() - start_time):.2f} с')