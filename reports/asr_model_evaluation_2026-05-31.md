# Whisper / Qwen ASR 評価レポート

作成日: 2026-05-31  
対象: OpenVINO Whisper IR モデル、Qwen3-ASR OpenVINO IR モデル  
注記: 依頼文の「qwern」は、リポジトリ内の実装に合わせて Qwen3-ASR として評価した。

## 結論

今回の 2 本の日本語合成音声では、精度は Qwen3-ASR 1.7B 系が最も安定し、推論速度は Whisper large-v3-turbo 系が最も高速だった。

- 精度優先: `qwen3-asr-1.7b-fp16-ov`
- 速度優先: `whisper-large-v3-turbo-int4-ov` または `whisper-large-v3-turbo-int8-ov`
- バランス重視: `whisper-large-v3-turbo-int4-ov`
- 軽量モデル比較では、Whisper tiny/base は速いが誤認識が増え、Qwen 0.6B は精度が比較的良い代わりに推論が約 3 倍遅い

## 評価条件

| 項目 | 内容 |
|---|---|
| 実行日 | 2026-05-31 |
| 実行環境 | Windows / OpenVINO `GPU` |
| 入力音声 | Windows 日本語 TTS で生成した 2 本の WAV |
| サンプル | `artifacts/audio/ja_eval_ichiro.wav`, `artifacts/audio/ja_eval_haruka.wav` |
| 計測範囲 | `record`, `load`, `warmup` を除外したモデル処理時間 |
| Warmup | 各モデル 1 回 |
| 精度指標 | CER: Character Error Rate。空白・句読点・長音記号等は正規化して除外 |
| 生データ | `artifacts/results/asr_eval_2026-05-31.jsonl` |

制約として、今回は自然発話ではなく TTS 合成音声を使っている。モデル間の同条件比較には使えるが、実運用のマイク音声、雑音、話者差、話速、専門用語への強さは別途評価が必要。

## モデル別結果

平均 CER は低いほど良く、平均処理時間は短いほど良い。

| モデル | 系統 | 平均 CER | 平均処理時間 | 平均 infer | 平均 load |
|---|---:|---:|---:|---:|---:|
| `qwen3-asr-1.7b-fp16-ov` | Qwen | 5.43% | 0.991s | 0.991s | 9.140s |
| `qwen3-asr-1.7b-int4-ov` | Qwen | 6.52% | 1.040s | 1.040s | 7.401s |
| `qwen3-asr-1.7b-int8-ov` | Qwen | 6.52% | 1.062s | 1.062s | 8.042s |
| `qwen3-asr-0.6b-int8-ov` | Qwen | 8.70% | 1.064s | 1.064s | 5.748s |
| `whisper-large-v3-turbo-int4-ov` | Whisper | 10.87% | 0.354s | 0.350s | 1.855s |
| `whisper-small-int8-ov` | Whisper | 12.03% | 0.500s | 0.496s | 1.053s |
| `whisper-large-v3-turbo-fp16-ov` | Whisper | 13.04% | 0.353s | 0.347s | 2.932s |
| `whisper-large-v3-turbo-int8-ov` | Whisper | 13.04% | 0.354s | 0.351s | 1.851s |
| `qwen3-asr-0.6b-fp16-ov` | Qwen | 15.44% | 0.935s | 0.935s | 5.551s |
| `qwen3-asr-0.6b-int4-ov` | Qwen | 16.53% | 1.028s | 1.028s | 5.511s |
| `whisper-base-int8-ov` | Whisper | 17.69% | 0.361s | 0.357s | 0.739s |
| `whisper-tiny-int8-ov` | Whisper | 26.62% | 0.341s | 0.334s | 0.806s |

## 系統別傾向

| 系統 | サンプル数 | 平均 CER | 平均処理時間 | 平均 load |
|---|---:|---:|---:|---:|
| Whisper | 12 | 15.55% | 0.377s | 1.539s |
| Qwen3-ASR | 12 | 9.86% | 1.020s | 6.899s |

Whisper はモデル処理が速く、特に `large-v3-turbo` は tiny/base より高精度でありながら処理時間が短かった。短文のリアルタイム用途、低遅延 UI、繰り返し推論では Whisper が扱いやすい。

Qwen3-ASR はロード時間が長く、単発実行では重い。ただし 1.7B 系は CER が低く、固有語に近い「ウィスパー」「キューウェン」でも Whisper より意味を保つケースが多かった。常駐プロセスでロード済みモデルを使い回す運用なら、精度面の利点が出やすい。

## 誤認識の例

参照文:

> 録音した音声を使って、ウィスパーとキューウェンの推論時間を測定します。誤認識が少ないモデルを選びます。

代表的な出力:

| モデル | CER | 出力 |
|---|---:|---|
| `whisper-tiny-int8-ov` | 36.96% | ログを下折せを使って、ウイスパート球員のスイロン時間を測定します。ご認識が少ないモデルを選びます。 |
| `whisper-large-v3-turbo-int4-ov` | 21.74% | 録音した音声を使って、VスパーとQNの推論時間を測定します。5人式が少ないモデルを選びます。 |
| `qwen3-asr-1.7b-fp16-ov` | 10.87% | 録音した音声を使ってウィスパーと吸煙の推論時間を測定します。誤認識が少ないモデルを選びます。 |

このサンプルでは、「キューウェン」が各モデルで崩れやすかった。実運用で製品名や人名が重要なら、語彙を含む評価音声を追加して確認する必要がある。

## 推奨

1. 低遅延・逐次処理を優先するなら `whisper-large-v3-turbo-int4-ov` を第一候補にする。今回の条件では処理時間が約 0.354s と速く、1 本目の音声では完全一致した。
2. 文字起こし精度を優先し、モデルを常駐させられるなら `qwen3-asr-1.7b-fp16-ov` を候補にする。平均 CER は 5.43% で最良だったが、平均 load は約 9.14s と長い。
3. Qwen 0.6B は 1.7B より軽いが、今回の結果では精度面の優位は限定的だった。特に `qwen3-asr-0.6b-int8-ov` は 0.6B 内では最も良い。
4. Whisper tiny/base は速度差が小さい割に精度が落ちるため、今回の日本語用途では積極採用しにくい。
5. 本番判断前に、実マイク音声、雑音あり音声、複数話者、長尺音声、業務固有語を含む 20-50 サンプルで再評価する。

## 再現メモ

評価時は、各音声を 16 kHz mono float32 に変換し、既存の `_run_model` と `_run_qwen_model` を直接呼び出した。`total` は録音・ロード・ウォームアップを除外し、Whisper は `preprocess + infer + decode`、Qwen は helper 実装上 `infer` のみとして記録した。
