＜内容＞

Signate主催コンペ「日本取引所グループ ファンダメンタルズ分析チャレンジ」で9位入賞したソースコード



＜注意点＞

コンペで提供されたデータがないと実行できないが、データはコンペルールにより共有不可でかつ削除済みである。

必要なデータ仕様については、コンペのチュートリアルページを参照いただきたい。

上記データをもとに作成した中間ファイル（m_forecast_confidence.csv、m_qsales.csv）があるが、その中身も共有することは許可されていないため、ヘッダだけ記載された空のファイルを格納している。



＜実行手順＞

①m_forecast_confidence.csvファイルを作成し、dataフォルダに格納する

②m_qsales.csvファイルを作成し、dataフォルダに格納する

③JPX_train.ipynbを実行し、２つのモデルファイルを生成する

④上記２つのモデルファイルをインプットに、predictor.pyを実行する



＜①m_forecast_confidence.csvファイルの作成方法＞

銘柄ごと、会計年度ごとの売上の予想と実績の乖離率を計算する。対象期間は2016年～2020年。



乖離率 = ( 売上実績 ー 売上予測 ) / 売上実績



予測信頼度フラグを作成する。銘柄ごとの乖離率の平均が、0.12以上なら1、-0.12以下なら-1、それ以外なら0とする。

銘柄、予測信頼度フラグをファイル内容とする。



＜②m_qsales.csvファイルの作成方法＞

銘柄ごと、会計年度&四半期ごとの売上実績の平均を計算する。対象期間は2016年Q1～2020年Q4。

銘柄ごとの各四半期売上を合計し、年間における四半期ごとの比率を計算する。

上記を累積値に換算する。ただし、数値が不足していて計算ができない銘柄は、以下とする。

Q1	0.25

Q2	0.5

Q3	0.75

Q4	1



銘柄、第1四半期売上比率、第2四半期売上比率、第3四半期売上比率、第4四半期売上比率をファイル内容とする。

