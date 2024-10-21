import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import lfilter, freqz
import parselmouth
import matplotlib
from matplotlib.gridspec import GridSpec

matplotlib.rcParams["font.family"] = "MS Gothic"


class FormantAnalyzer:
    def __init__(self):
        # 母音ごとの理想フォルマント値
        self.vowel_ranges = {
            "あ": {"color": "pink", "F1": (750, 950), "F2": (1100, 1300)},
            "い": {"color": "lightblue", "F1": (250, 350), "F2": (2100, 2300)},
            "う": {"color": "lightgreen", "F1": (300, 400), "F2": (1200, 1400)},
            "え": {"color": "yellow", "F1": (450, 550), "F2": (1800, 2000)},
            "お": {"color": "orange", "F1": (450, 550), "F2": (800, 1000)},
        }

        # 平均値バッファ
        self.f1_buffer = []
        self.f2_buffer = []
        self.buffer_size = 10  # 平均を取るサンプル数

        # PyAudioの設定
        self.setup_audio()

        # グラフの設定
        self.setup_plots()

    def setup_audio(self):
        self.p = pyaudio.PyAudio()
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100

        print("\n利用可能なオーディオ入力デバイス:")
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if dev.get("maxInputChannels") > 0:
                print(f"デバイス {i}: {dev.get('name')}")

        device_index = int(input("\n使用するデバイス番号を入力してください: "))
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.CHUNK,
        )

    def setup_plots(self):
        self.fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, width_ratios=[3, 1])

        # 波形プロット
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        (self.line,) = self.ax1.plot([], [])
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_xlim(0, self.CHUNK)
        self.ax1.set_title("音声波形")
        self.ax1.set_ylabel("振幅")

        # リアルタイムフォルマントプロット（母音ごとの良好範囲付き）
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.bar_width = 0.35
        self.ax2.set_ylim(0, 3500)
        self.ax2.set_title("リアルタイムフォルマント")
        self.ax2.set_ylabel("周波数 (Hz)")

        # 母音ごとの良好範囲を表示
        for vowel, data in self.vowel_ranges.items():
            # F1の範囲
            self.ax2.add_patch(
                plt.Rectangle(
                    (1 - self.bar_width, data["F1"][0]),
                    self.bar_width * 2,
                    data["F1"][1] - data["F1"][0],
                    alpha=0.2,
                    color=data["color"],
                    label=f"{vowel}",
                )
            )
            # F2の範囲
            self.ax2.add_patch(
                plt.Rectangle(
                    (2 - self.bar_width, data["F2"][0]),
                    self.bar_width * 2,
                    data["F2"][1] - data["F2"][0],
                    alpha=0.2,
                    color=data["color"],
                )
            )

        self.current_bars = self.ax2.bar(
            [1 - self.bar_width / 2, 2 - self.bar_width / 2],
            [0, 0],
            self.bar_width,
            label="現在の声",
        )

        self.ax2.set_xticks([1, 2])
        self.ax2.set_xticklabels(["F1", "F2"])
        self.ax2.legend()

        # 1秒平均フォルマントプロット
        self.ax3 = self.fig.add_subplot(gs[0, 1])
        self.ax3.set_ylim(0, 3500)
        self.ax3.set_title("1秒平均フォルマント")
        self.ax3.set_ylabel("周波数 (Hz)")

        # 母音ごとの良好範囲を表示（平均値グラフ）
        for vowel, data in self.vowel_ranges.items():
            self.ax3.add_patch(
                plt.Rectangle(
                    (1 - 0.25, data["F1"][0]),
                    0.5,
                    data["F1"][1] - data["F1"][0],
                    alpha=0.2,
                    color=data["color"],
                )
            )
            self.ax3.add_patch(
                plt.Rectangle(
                    (2 - 0.25, data["F2"][0]),
                    0.5,
                    data["F2"][1] - data["F2"][0],
                    alpha=0.2,
                    color=data["color"],
                )
            )

        self.avg_bars = self.ax3.bar([1, 2], [0, 0], 0.5)

        self.ax3.set_xticks([1, 2])
        self.ax3.set_xticklabels(["F1", "F2"])

        # 母音フォルマント一覧
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        self.ax4.axis("off")
        vowel_text = "母音ごとの良好範囲:\n\n"
        for vowel, data in self.vowel_ranges.items():
            vowel_text += f"{vowel}: F1={data['F1'][0]}-{data['F1'][1]}Hz\n    F2={data['F2'][0]}-{data['F2'][1]}Hz\n"
        self.ax4.text(0.1, 0.9, vowel_text, va="top")

        # 差分表示用テキスト
        self.text = self.ax2.text(0.02, 0.95, "", transform=self.ax2.transAxes)

        plt.tight_layout()

    def update(self, frame):
        try:
            data = np.frombuffer(
                self.stream.read(self.CHUNK, exception_on_overflow=False),
                dtype=np.float32,
            )

            self.line.set_data(range(len(data)), data)

            if np.max(np.abs(data)) < 0.01:
                for bar in self.current_bars + self.avg_bars:
                    bar.set_height(0)
                    bar.set_color("gray")
                    if hasattr(bar, "text_label"):
                        bar.text_label.remove()
                        delattr(bar, "text_label")
                self.text.set_text("音声を入力してください")
                self.ax2.set_title("リアルタイムフォルマント - 待機中")
                return self.line, *self.current_bars, *self.avg_bars, self.text

            sound = parselmouth.Sound(data, sampling_frequency=self.RATE)
            formants = sound.to_formant_burg()

            f1 = formants.get_value_at_time(1, formants.duration / 2)
            f2 = formants.get_value_at_time(2, formants.duration / 2)

            if f1 and f2:
                # 現在の母音を判定
                current_vowel = None
                for vowel, data in self.vowel_ranges.items():
                    if (
                        data["F1"][0] <= f1 <= data["F1"][1]
                        and data["F2"][0] <= f2 <= data["F2"][1]
                    ):
                        current_vowel = vowel
                        break

                # リアルタイム値の更新
                for i, (bar, value) in enumerate(zip(self.current_bars, [f1, f2])):
                    bar.set_height(value)
                    if current_vowel:
                        bar.set_color(self.vowel_ranges[current_vowel]["color"])
                    else:
                        bar.set_color("gray")

                    if hasattr(bar, "text_label"):
                        bar.text_label.remove()
                    bar.text_label = self.ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{value:.0f}",
                        ha="center",
                        va="bottom",
                    )

                self.f1_buffer.append(f1)
                self.f2_buffer.append(f2)
                if len(self.f1_buffer) > self.buffer_size:
                    self.f1_buffer.pop(0)
                    self.f2_buffer.pop(0)

                f1_avg = np.mean(self.f1_buffer)
                f2_avg = np.mean(self.f2_buffer)

                # 平均値の母音判定
                avg_vowel = None
                for vowel, data in self.vowel_ranges.items():
                    if (
                        data["F1"][0] <= f1_avg <= data["F1"][1]
                        and data["F2"][0] <= f2_avg <= data["F2"][1]
                    ):
                        avg_vowel = vowel
                        break

                for i, (bar, value) in enumerate(zip(self.avg_bars, [f1_avg, f2_avg])):
                    bar.set_height(value)
                    if avg_vowel:
                        bar.set_color(self.vowel_ranges[avg_vowel]["color"])
                    else:
                        bar.set_color("gray")

                    if hasattr(bar, "text_label"):
                        bar.text_label.remove()
                    bar.text_label = self.ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{value:.0f}",
                        ha="center",
                        va="bottom",
                    )

                status = f'現在の母音: {current_vowel if current_vowel else "範囲外"}\n'
                status += f'平均母音: {avg_vowel if avg_vowel else "範囲外"}'
                self.text.set_text(status)

                self.ax2.set_title(
                    f'リアルタイムフォルマント - {"範囲内" if current_vowel else "範囲外"}'
                )

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        except Exception as e:
            print(f"更新エラー: {str(e)}")

        return self.line, *self.current_bars, *self.avg_bars, self.text

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self.update, interval=50, cache_frame_data=False, save_count=100
        )
        plt.show(block=True)
        return self.anim

    def cleanup(self):
        if hasattr(self, "stream"):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, "p"):
            self.p.terminate()

    def __del__(self):
        self.cleanup()


if __name__ == "__main__":
    try:
        analyzer = FormantAnalyzer()
        anim = analyzer.run()
        plt.show()
    except KeyboardInterrupt:
        print("\nプログラムを終了します")
    finally:
        if "analyzer" in locals():
            analyzer.cleanup()
