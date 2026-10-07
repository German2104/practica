"""Illustrative waterfall; values are not results of the portfolio model."""
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


class WaterfallChart:
    def __init__(self, *, data: pd.DataFrame) -> None:
        self.data = data

    def save(self, destination: pathlib.Path) -> None:
        data = self.data.copy()
        initial = 100.0
        data['end'] = initial + data['contribution'].cumsum()
        data['start'] = data['end'] - data['contribution']
        final = float(data['end'].iloc[-1])
        assert abs(final - initial - data['contribution'].sum()) < 1e-9
        navy = '#15365B'
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 16})
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.bar(0, initial, width=.68, color='#5598E8', edgecolor=navy)
        level = initial
        for index, row in enumerate(data.itertuples(index=False), start=1):
            color = '#B9D9F6' if row.contribution >= 0 else '#F5EACB'
            if row.factor == 'Остаток':
                color = '#D5D8DD'
            ax.plot([index - 1 + .34, index - .34], [level, level],
                    color=navy, linestyle='--', linewidth=1.2)
            ax.bar(index, abs(row.contribution), bottom=min(row.start, row.end),
                   width=.68, color=color, edgecolor=navy, linewidth=1)
            level = row.end
        ax.plot([6.34, 6.66], [final, final], color=navy, linestyle='--', linewidth=1.2)
        ax.bar(7, final, width=.68, color='#5598E8', edgecolor=navy)
        ax.set_xticks(range(8), ['NPV в T0', *data['factor'].tolist(), 'NPV в T1'])
        ax.set_title('Схема разложения', fontsize=25, weight='bold', color=navy, pad=22)
        ax.set_ylim(0, 163)
        ax.set_xlim(-.6, 7.6)
        ax.set_yticks([])
        ax.tick_params(axis='x', colors=navy, length=0, pad=15)
        for name, spine in ax.spines.items():
            spine.set_visible(name == 'bottom')
            spine.set_color(navy)
        ax.set_facecolor('none')
        fig.patch.set_alpha(0)
        fig.text(.5, .035, 'Иллюстративная схема — без численных результатов',
                 ha='center', color=navy, fontsize=13)
        fig.subplots_adjust(bottom=.23, top=.84, left=.025, right=.99)
        fig.savefig(destination.with_suffix('.png'), dpi=200, transparent=True)
        fig.savefig(destination.with_suffix('.svg'), transparent=True)
        plt.close(fig)
        data.to_csv(destination.with_suffix('.csv'), index=False)


if __name__ == '__main__':
    frame = pd.DataFrame({
        'factor': ['Объём', 'Ставка', 'PD', 'LTV', 'Взаимодействия', 'Остаток'],
        'contribution': [30., 15., -20., -10., 10., -5.],
    })
    WaterfallChart(data=frame).save(pathlib.Path(__file__).with_name('npv-waterfall-pandas'))
