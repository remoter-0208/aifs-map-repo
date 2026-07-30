# AIFS-single 500hPa gh & 850hPa T auto chart

GitHub Actions で ECMWF Open Data の **AIFS-single**（データ駆動型モデル）から
500hPa 高度場（gh）と 850hPa 気温（t）を定期取得し、
weather-models.info 風の重ね合わせ図を自動生成・公開します。

## 仕組み

1. `.github/workflows/update.yml` が UTC 06:20 / 12:20 / 18:20 / 00:20（各サイクル+約6時間後）に起動
   （AIFS-single のデータは各サイクルの5〜6時間後に公開されるため）
2. `scripts/plot_500z_850t.py` が `ecmwf-opendata` パッケージ経由で最新サイクルの
   GRIB2 を取得し、cartopy で作図
3. 生成した PNG を `docs/output/latest_XXX.png`（XXX=予報時間）として上書きコミット
4. `docs/index.html`（GitHub Pages）がプルダウンで予報時間を選んで最新図を表示

## セットアップ

1. このリポジトリを GitHub に push
2. Settings → Pages → Source を `main` ブランチの `/docs` に設定
3. Actions タブで `Update AIFS-single 500Z/850T chart` を一度 **Run workflow** で手動実行し、
   正常に動くことを確認（初回は依存インストールで数分かかります）
4. 以降は cron で自動更新されます

## ローカルでの試し方

```bash
pip install -r requirements.txt
python scripts/plot_500z_850t.py --steps 24 48 72 --domain japan --out-dir docs/output
```

`--date`/`--time` を省略すると `ecmwf-opendata` が自動的に「取得可能な最新サイクル」を
選んでくれるので、実行タイミングを気にする必要はありません。

## カスタマイズ

- `scripts/plot_500z_850t.py` の `DOMAINS` に緯度経度範囲を追加すれば表示領域を変更可能
- `T_LEVELS` / `T_COLORS` で850hPa気温の配色を調整
- `Z_INTERVAL` で500hPa高度の等値線間隔（既定60m）を変更
- 予報時間の一覧は workflow の `steps` 入力、および `docs/index.html` の `<select>` を編集

## 注意

- ECMWF Open Data の利用規約に従ってください（商用転載制限など、配布元により条件が異なります）
- AIFS-single はモデル初期化から公開まで遅延があるため、直近サイクルが未公開の場合
  `ecmwf-opendata` は自動的に一つ前の利用可能サイクルを返します
