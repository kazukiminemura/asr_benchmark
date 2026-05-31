# Whisper / Kotoba / Qwen ASR 評価レポート

作成日: 2026-05-31  
対象: OpenVINO Whisper IR モデル、Kotoba Whisper v2.2 OpenVINO IR モデル、Qwen3-ASR OpenVINO IR モデル  
注記: `RoachLin/kotoba-whisper-v2.2-faster` は CTranslate2/faster-whisper 形式のため、OpenVINO IR は model card の `base_model` である `kotoba-tech/kotoba-whisper-v2.2` から export した。`kotoba-whisper-v2.2-faster-fp16-ov` と `kotoba-whisper-v2.2-fp16-ov` は同じ IR を別名で保存した比較用エントリ。

## 結論

今回の 2 本の日本語合成音声では、精度は `qwen3-asr-1.7b-fp16-ov` が最良、モデル処理時間は `whisper-tiny-int8-ov` が最速だった。Kotoba v2.2 は Whisper large-v3-turbo より少し遅く、今回の短文セットでは精度も turbo の最良ケースを上回らなかった。

- 精度優先: `qwen3-asr-1.7b-fp16-ov`
- 速度優先: `whisper-tiny-int8-ov`
- Whisper/Kotoba 系で精度優先: `whisper-large-v3-turbo-int4-ov`
- Kotoba v2.2 は `今日は` を `共和` と誤認識し、固有語サンプルでも `キューウェン` が崩れたため、この評価セットでは採用優先度は中位
- `kotoba-tech/kotoba-whisper-v2.2` と `RoachLin/kotoba-whisper-v2.2-faster` 由来の OpenVINO IR は同一重みなので、差はロード順や実行ばらつき程度

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
| `qwen3-asr-1.7b-fp16-ov` | Qwen3-ASR | 5.43% | 0.991s | 0.991s | 9.140s |
| `qwen3-asr-1.7b-int4-ov` | Qwen3-ASR | 6.52% | 1.040s | 1.040s | 7.401s |
| `qwen3-asr-1.7b-int8-ov` | Qwen3-ASR | 6.52% | 1.062s | 1.062s | 8.042s |
| `qwen3-asr-0.6b-int8-ov` | Qwen3-ASR | 8.70% | 1.064s | 1.064s | 5.748s |
| `whisper-large-v3-turbo-int4-ov` | Whisper | 10.87% | 0.354s | 0.350s | 1.855s |
| `whisper-small-int8-ov` | Whisper | 12.03% | 0.500s | 0.496s | 1.053s |
| `whisper-large-v3-turbo-fp16-ov` | Whisper | 13.04% | 0.352s | 0.347s | 2.932s |
| `whisper-large-v3-turbo-int8-ov` | Whisper | 13.04% | 0.354s | 0.350s | 1.850s |
| `kotoba-whisper-v2.2-fp16-ov` | Kotoba Whisper | 14.36% | 0.421s | 0.415s | 2.486s |
| `kotoba-whisper-v2.2-faster-fp16-ov` | Kotoba Whisper | 14.36% | 0.431s | 0.425s | 2.423s |
| `qwen3-asr-0.6b-fp16-ov` | Qwen3-ASR | 15.44% | 0.935s | 0.935s | 5.551s |
| `qwen3-asr-0.6b-int4-ov` | Qwen3-ASR | 16.53% | 1.028s | 1.028s | 5.511s |
| `whisper-base-int8-ov` | Whisper | 17.69% | 0.361s | 0.357s | 0.739s |
| `whisper-tiny-int8-ov` | Whisper | 26.62% | 0.341s | 0.334s | 0.806s |

## 系統別傾向

| 系統 | サンプル数 | 平均 CER | 平均処理時間 | 平均 load |
|---|---:|---:|---:|---:|
| Qwen3-ASR | 12 | 9.86% | 1.020s | 6.899s |
| Kotoba Whisper | 4 | 14.36% | 0.426s | 2.454s |
| Whisper | 12 | 15.55% | 0.377s | 1.539s |

Whisper はモデル処理が速く、特に `large-v3-turbo` は tiny/base より高精度でありながら処理時間が短かった。短文のリアルタイム用途、低遅延 UI、繰り返し推論では Whisper が扱いやすい。

Kotoba Whisper v2.2 は `large-v3-turbo` よりやや大きく、平均処理時間は約 0.426s。今回の 2 サンプルでは `whisper-small-int8-ov` と近い平均 CER で、`large-v3-turbo-int4-ov` よりは悪かった。ただし評価セットが短く、Kotoba が得意なドメイン語彙や話し言葉では別途確認する価値がある。

Qwen3-ASR はロード時間が長く、単発実行では重い。ただし 1.7B 系は CER が低く、固有語に近い「ウィスパー」「キューウェン」でも Whisper/Kotoba より意味を保つケースが多かった。常駐プロセスでロード済みモデルを使い回す運用なら、精度面の利点が出やすい。

## 誤認識の例

参照文:

> 録音した音声を使って、ウィスパーとキューウェンの推論時間を測定します。誤認識が少ないモデルを選びます。

代表的な出力:

| モデル | CER | 出力 |
|---|---:|---|
| `whisper-tiny-int8-ov` | 36.96% | ログを下折せを使って、ウイスパート球員のスイロン時間を測定します。ご認識が少ないモデルを選びます。 |
| `whisper-large-v3-turbo-int4-ov` | 21.74% | 録音した音声を使って、VスパーとQNの推論時間を測定します。5人式が少ないモデルを選びます。 |
| `kotoba-whisper-v2.2-fp16-ov` | 21.74% | 録音した音声を使ってVスパーと9円の推論時間を測定します。5人式が少ないモデルを選びます。 |
| `qwen3-asr-1.7b-fp16-ov` | 10.87% | 録音した音声を使ってウィスパーと吸煙の推論時間を測定します。誤認識が少ないモデルを選びます。 |

このサンプルでは、「キューウェン」が各モデルで崩れやすかった。Kotoba は `Vスパー` と `9円`、Qwen 1.7B は `吸煙` という誤認識になったが、文全体の意味は Qwen 1.7B が最も保っていた。

## 推奨

1. 低遅延・逐次処理を優先するなら `whisper-large-v3-turbo-int4-ov` を第一候補にする。今回の条件では処理時間が約 0.354s と速く、平均 CER も Whisper/Kotoba 系で最良だった。
2. 文字起こし精度を優先し、モデルを常駐させられるなら `qwen3-asr-1.7b-fp16-ov` を候補にする。平均 CER は 5.43% で最良だったが、平均 load は約 9.14s と長い。
3. `kotoba-tech/kotoba-whisper-v2.2` は今回の短文 TTS では平均 CER 14.36%、平均処理時間 0.421s。日本語特化モデルとして追加評価の価値はあるが、この 2 サンプルだけなら `large-v3-turbo-int4-ov` を優先する。
4. `RoachLin/kotoba-whisper-v2.2-faster` は CTranslate2 版そのものを OpenVINO が読むわけではなく、同じ元モデルから IR export したものとして比較する。OpenVINO 運用では `kotoba-whisper-v2.2-fp16-ov` 名を使う方が由来が明確。
5. 本番判断前に、実マイク音声、雑音あり音声、複数話者、長尺音声、業務固有語を含む 20-50 サンプルで再評価する。

## 再現メモ

評価時は、各音声を 16 kHz mono float32 に変換し、既存の `_run_model` と `_run_qwen_model` を直接呼び出した。`total` は録音・ロード・ウォームアップを除外し、Whisper/Kotoba は `preprocess + infer + decode`、Qwen は helper 実装上 `infer` のみとして記録した。
