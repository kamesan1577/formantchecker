# Formant Checker
マイクからの音声をリアルタイムで受け取り、理想の女声とのフォルマントの差を視覚的に確認できるスクリプトです
Windows11 Python3.11.0 環境でのみテスト済み
# 実行方法
```
pip install -r requirements.txt
python ./main.py

//ターミナルで入力デバイスを選ぶとウィンドウが立ち上がる
```
# 特筆事項
PyAudioは音声デバイスのアクセスに必要なportaudioに依存します

Ubuntu/Debian: sudo apt-get install portaudio19-dev
macOS: brew install portaudio
Windows: Visual C++ build toolsが必要な場合があります
