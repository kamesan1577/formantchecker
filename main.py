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
        self.vowel_formants = {
            "あ": {"F1": 850, "F2": 1200},
            "い": {"F1": 300, "F2": 2200},
            "う": {"F1": 350, "F2": 1300},
            "え": {"F1": 500, "F2": 1900},
            "お": {"F1": 500, "F2": 900},
        }

        # 現在の目標フォルマント
        self.target_formants = {"F1": 550, "F2": 2500}

        # 平均値バッファ
        self.f1_buffer = []
        self.f2_buffer = []
        self.freq_buffer = []
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

        # リアルタイムフォルマントプロット（良好範囲付き）
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.bar_width = 0.35
        self.ax2.set_ylim(0, 3500)
        self.ax2.set_title("リアルタイムフォルマント")
        self.ax2.set_ylabel("周波数 (Hz)")

        # 良好範囲の表示
        self.good_range = {
            "F1": {"min": 500, "max": 600},  # F1の良好範囲
            "F2": {"min": 2400, "max": 2600},  # F2の良好範囲
        }

        # 良好範囲の背景表示
        self.ax2.add_patch(
            plt.Rectangle(
                (1 - self.bar_width, self.good_range["F1"]["min"]),
                self.bar_width * 2,
                self.good_range["F1"]["max"] - self.good_range["F1"]["min"],
                alpha=0.2,
                color="green",
            )
        )
        self.ax2.add_patch(
            plt.Rectangle(
                (2 - self.bar_width, self.good_range["F2"]["min"]),
                self.bar_width * 2,
                self.good_range["F2"]["max"] - self.good_range["F2"]["min"],
                alpha=0.2,
                color="green",
            )
        )

        self.current_bars = self.ax2.bar(
            [1 - self.bar_width / 2, 2 - self.bar_width / 2],
            [0, 0],
            self.bar_width,
            label="現在の声",
        )

        self.target_bars = self.ax2.bar(
            [1 + self.bar_width / 2, 2 + self.bar_width / 2],
            [self.target_formants["F1"], self.target_formants["F2"]],
            self.bar_width,
            label="目標値",
        )

        self.ax2.set_xticks([1, 2])
        self.ax2.set_xticklabels(["F1", "F2"])
        self.ax2.legend()

        # 1秒平均フォルマントプロット（良好範囲付き）
        self.ax3 = self.fig.add_subplot(gs[0, 1])
        self.ax3.set_ylim(0, 3500)
        self.ax3.set_title("1秒平均フォルマント")
        self.ax3.set_ylabel("周波数 (Hz)")

        # 良好範囲の背景表示（平均値グラフ）
        self.ax3.add_patch(
            plt.Rectangle(
                (1 - 0.25, self.good_range["F1"]["min"]),
                0.5,
                self.good_range["F1"]["max"] - self.good_range["F1"]["min"],
                alpha=0.2,
                color="green",
            )
        )
        self.ax3.add_patch(
            plt.Rectangle(
                (2 - 0.25, self.good_range["F2"]["min"]),
                0.5,
                self.good_range["F2"]["max"] - self.good_range["F2"]["min"],
                alpha=0.2,
                color="green",
            )
        )

        self.avg_bars = self.ax3.bar([1, 2], [0, 0], 0.5)

        self.ax3.set_xticks([1, 2])
        self.ax3.set_xticklabels(["F1", "F2"])

        # 母音フォルマント一覧
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        self.ax4.axis("off")
        vowel_text = "理想フォルマント値:\n\n"
        for vowel, formants in self.vowel_formants.items():
            vowel_text += f"{vowel}: F1={formants['F1']}Hz, F2={formants['F2']}Hz\n"
        vowel_text += "\n良好範囲:\n"
        vowel_text += (
            f"F1: {self.good_range['F1']['min']}-{self.good_range['F1']['max']}Hz\n"
        )
        vowel_text += (
            f"F2: {self.good_range['F2']['min']}-{self.good_range['F2']['max']}Hz"
        )
        self.ax4.text(0.1, 0.9, vowel_text, va="top")

        # 差分表示用テキスト
        self.text = self.ax2.text(0.02, 0.95, "", transform=self.ax2.transAxes)

        plt.tight_layout()

    def calculate_mean_frequency(self, data):
        # 音声信号のゼロクロス数をカウント
        zero_crossings = np.where(np.diff(np.signbit(data)))[0]
        if len(zero_crossings) > 1:
            # ゼロクロスの間隔から周波数を計算
            mean_interval = np.mean(np.diff(zero_crossings))
            frequency = self.RATE / (2 * mean_interval)
            return frequency
        return 0

    def get_improvement_advice(self, f1_diff, f2_diff):
        if abs(f1_diff) < 50 and abs(f2_diff) < 100:
            return "良好な音声です"
        elif f1_diff > 50:
            return "より口を狭く"
        elif f1_diff < -50:
            return "より口を開いて"
        elif f2_diff > 100:
            return "舌の位置を下げて"
        elif f2_diff < -100:
            return "舌の位置を上げて"
        return "調整を続けてください"

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
                    bar.set_color("blue")  # デフォルト色に戻す
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
                # リアルタイム値の更新と数値表示
                for i, (bar, value) in enumerate(zip(self.current_bars, [f1, f2])):
                    bar.set_height(value)
                    # しきい値チェックと色の設定
                    formant_type = "F1" if i == 0 else "F2"
                    if (
                        self.good_range[formant_type]["min"]
                        <= value
                        <= self.good_range[formant_type]["max"]
                    ):
                        bar.set_color("red")
                    else:
                        bar.set_color("blue")

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
                for i, (bar, value) in enumerate(zip(self.avg_bars, [f1_avg, f2_avg])):
                    bar.set_height(value)
                    # 平均値バーの色も設定
                    formant_type = "F1" if i == 0 else "F2"
                    if (
                        self.good_range[formant_type]["min"]
                        <= value
                        <= self.good_range[formant_type]["max"]
                    ):
                        bar.set_color("red")
                    else:
                        bar.set_color("blue")

                    if hasattr(bar, "text_label"):
                        bar.text_label.remove()
                    bar.text_label = self.ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{value:.0f}",
                        ha="center",
                        va="bottom",
                    )

                f1_diff = f1 - self.target_formants["F1"]
                f2_diff = f2 - self.target_formants["F2"]
                diff_text = f"差分:\nF1: {f1_diff:+.0f}Hz\nF2: {f2_diff:+.0f}Hz"
                self.text.set_text(diff_text)

                advice = self.get_improvement_advice(f1_diff, f2_diff)
                self.ax2.set_title(f"リアルタイムフォルマント - {advice}")

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
