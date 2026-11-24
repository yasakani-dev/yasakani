YASAKANI quickstart 0.3.5
=========================

日本語
------
Windows 11 と NVIDIA の RTX 1 枚のパソコンで、YASAKANI をすぐに試すための起動ファイルです。動作を確かめたのは RTX 5090 です。

使い方
  1. この zip を、C:\yasakani のような英数字だけの場所に展開します。
  2. START.cmd をダブルクリックします。
  3. 画面の質問に番号で答えます（モデルは「品質モデル」「最速モデル」「最高品質モデル」「Inkling-2 Mini（試験的）」のどれか）。
  4. 最後にメニューが出ます。Enter を押すと、ブラウザーの画面が開きます。
  5. 点検まで済むと、次からは START.cmd をダブルクリックするだけで、ブラウザーの画面が直接開きます
     （モデルの質問とメニューは出ません）。メニューを出す時は、このフォルダーでコマンドプロンプトを開いて
     START.cmd --menu と打ちます。

START.cmd がすること
  - このパソコンの確認（Python 3.10 以上、GPU、NVIDIA のドライバー、空き容量）
  - エンジンの zip を公開ページからダウンロードして、SHA-256 を確かめる
  - このフォルダーの中に Python の環境を作って、パッケージを入れる
  - モデル（約 13〜18 GB。Inkling-2 Mini は約 15.9 GB）を Hugging Face からダウンロードする
  - 点検（yasakani info・verify・selfcheck）
  - メニューを出す：1 番（Enter だけでも同じ）がブラウザーの画面です。ほかに、速さを測る／先読みあり・なしを比べる／
    黒い画面でチャットする／サーバーを起動する、を選べます

知っておくこと
  - Python 3.10 以上が要ります（https://www.python.org/downloads/windows/）。
  - 空き容量は 20 GB ほど要ります。
  - NVIDIA のドライバーは 580 以上が要ります（エンジンが CUDA 13 で作ってあるため）。古いときは、ダウンロードの前に
    止まって案内します（https://www.nvidia.com/ja-jp/drivers/）。
  - GPU：確かめたのは RTX 5090 だけです。0.1.3 から、RTX 30・40・50 シリーズで GPU のメモリが 16 GB 以上のカードでも
    動くように作ってあります。ただし最適化も動作の確認もしていません。24 GB 未満のカードでは最速モデルだけを使い（品質モデルと最高品質モデルは入りません）、文脈の長さは空きメモリから
    決めます。AMD・Intel の GPU と Mac には対応していません。
  - 「先読みあり・なしを比べる」（lookahead_check.py）は、同じ 4 つの課題を、先読みなしと出荷時の設定で書かせます。
    1 回の計算の時間、1 回で確定したトークン数、速さ、書かれた文章が同じかどうかを表示します。
    技術の解説（docs/TECHNOLOGY.ja.md）の 1.3 の表と比べられます。
  - ブラウザーの画面（chat.html）は、このパソコンの中だけで動きます（127.0.0.1）。会話はどこにも送られません。
    タブは「チャット」「スピード測定」「API 接続」「モデル・状態」の 4 つです。「API 接続」には、ほかのアプリから使う時の
    アドレスが出ます。「モデル・状態」では、ダウンロード済みのモデルが 2 つ以上ある時に切り替えられます。
    終わる時は、画面の「終了」ボタンを押すか、黒い画面で Enter を押します。メニューから開いた時の Enter は、
    メニューに戻るだけです。START.cmd が正常に終わると、黒い画面は自動で閉じます。エラーで止まった時は、
    キーを押すまで開いたままです。
  - 入れたものは、すべてこのフォルダーの中に置きます。パソコン全体には何も入れません。
    やめるときは、このフォルダーを消すだけです。
  - 配布ファイルにはコード署名がありません。Windows が警告を出したら、発行元と入手先を確かめたうえで実行してください。
  - 途中で止めた時も、もう一度 START.cmd をダブルクリックすれば、済んだ手順は飛ばして続きから始まります。
  - 新しい版の確認：起動のたびに github.com に問い合わせます（1 時間に 1 回まで）。問い合わせるのは、メニューを
    出す前か、ブラウザーの画面を直接開く前です（問い合わせる前に表示します。送るのは最新の版を聞くリクエスト 1 回だけで、このパソコンの情報は送りません）。
    新しい版がある時、メニューの前では更新するかを聞きます。ブラウザーの画面を直接開く時は、黒い画面では聞きません。
    画面に「更新する」ボタンが出ます。「y」と答えるかボタンを押すまで、何も変えません。
    画面の「モデル・状態」の「更新を確認」を押すと、その場で 1 回だけ問い合わせます
    （押した時だけ。--no-update-check の時も使えます）。
    更新すると今の起動ファイルを backup フォルダーに控え、エンジンとパッケージを入れ直します（モデルはそのまま）。
    今すぐ確かめる：メニューの 7 か START.cmd --action update
    前の版に戻す：START.cmd --action rollback
    確認を止める：START.cmd --no-update-check（戻すときは --update-check）
  - Inkling-2 Mini（0.3.1 から試験的に対応。依頼は 1 件ずつ・24 GB 以上のカード）は、0.3.2 からモデルの質問の
    4 番で選べます（START.cmd --model inkling2 でも同じ）。変換済みのコピー（約 15.9 GB）を Hugging Face から取得して、
    ほかのモデルと同じく点検します。24 GB 未満のカードでは選べません。
    0.3.5 から画像も読めます（ブラウザーの画面の「画像」ボタン）。0.3.4 までに取得したフォルダーには、画像用のファイル
    （約 0.64 GB）を足すかを 1 回だけ聞きます。Enter だけで取得します。n なら文章だけのままです。後から足す時は
    START.cmd --image-file です。自分で取得や変換をしたフォルダーは、これまで通り START.cmd --model-dir <フォルダー> で
    開けます。手順はリポジトリの README.ja.md の「Inkling-2 Mini」にあります。
  - 手で入れる手順と、くわしい説明は https://github.com/yasakani-dev/yasakani にあります。

English
-------
A starter for trying YASAKANI on a Windows 11 computer with one NVIDIA RTX card (tested on the RTX 5090).

How to use
  1. Unpack this zip to a plain path such as C:\yasakani
  2. Double-click START.cmd
  3. Answer the questions with a number (the model is the "quality model", the "fastest model", the "max-quality model"
     or Inkling-2 Mini, experimental).
  4. A menu comes last. Press Enter and the page opens in your browser.
  5. Once the checks have passed, a double-click on START.cmd opens the browser page directly (no model question, no menu).
     For the menu, open a command prompt in this folder and type START.cmd --menu

What START.cmd does
  - checks this computer (Python 3.10 or newer, the GPU, the NVIDIA driver, free disk space)
  - downloads the engine zip from the release page and checks its SHA-256
  - makes a Python environment inside this folder and installs the package
  - downloads the model (about 13 to 18 GB; Inkling-2 Mini about 15.9 GB) from Hugging Face
  - runs the checks (yasakani info, verify, selfcheck)
  - shows a menu: 1 (or Enter alone) is the page in your browser; the other items measure the speed, compare look-ahead
    on and off, chat in the terminal, or start the server

Good to know
  - Python 3.10 or newer is needed (https://www.python.org/downloads/windows/).
  - About 20 GB of free disk space is needed.
  - NVIDIA driver 580 or newer is needed (the engine is built with CUDA 13). With an older driver START.cmd stops before
    the downloads and says so (https://www.nvidia.com/en-us/drivers/).
  - GPU: only the RTX 5090 has been tested. Since 0.1.3 YASAKANI is also built to run on RTX 30, 40 and 50 series cards with
    16 GB or more, but it is not optimized or tested on them. On cards with less than 24 GB only the fastest model is used, and
    the context length is chosen from the free GPU memory. AMD and Intel GPUs and Macs are not supported.
  - "Compare look-ahead on and off" (lookahead_check.py) writes the same four tasks without look-ahead and with the
    shipping settings. It prints the time of one pass, the tokens confirmed per pass, the speed, and whether the text is
    the same. Compare the result with table 1.3 of the technical notes (docs/TECHNOLOGY.md).
  - The browser page (chat.html) runs on this computer only (127.0.0.1). The conversation is not sent anywhere.
    It has four tabs: Chat, Speed, API and Model. The API tab shows the address for other apps. On the Model tab you can
    switch between models when two or more are downloaded. To stop, press the Quit button of the page or press Enter in
    the black window (if the page was opened from the menu, Enter goes back to the menu). When START.cmd ends without
    an error, the black window closes by itself; after an error it waits for a key, so that the messages can be read.
  - Everything is kept inside this folder; nothing is installed system-wide. To remove it, delete the folder.
  - The files are not code-signed. If Windows shows a warning, check the publisher and the source before you run them.
  - If a run was stopped halfway, double-click START.cmd again: finished steps are skipped.
  - New versions: at every start (at most once an hour) START.cmd asks github.com for the latest version, before it
    shows the menu or before it opens the browser page directly (it says so first; one request, nothing about this computer is sent). If there is a
    newer one, it asks before the menu whether to update. When the page opens directly, nothing is asked in the black
    window: the page shows an Update button. Nothing changes until you answer y or press the button.
    "Check for updates" in the Model tab of the page asks at once, one request, only when you press it
    (also with --no-update-check).
    An update copies the current start files to the backup folder and installs the engine and the package again
    (the models are kept). Check now: 7 in the menu or START.cmd --action update. Go back: START.cmd --action rollback.
    Turn the check off: START.cmd --no-update-check (on again: --update-check).
  - Inkling-2 Mini (experimental since 0.3.1; one request at a time; a card with 24 GB or more) is choice 4
    of the model question since 0.3.2 (or START.cmd --model inkling2). The quickstart downloads our converted copy (about
    15.9 GB) from Hugging Face and checks it like the other models. Cards under 24 GB are not offered it.
    Since 0.3.5 it also reads images (the Image button of the browser page). For a folder downloaded with 0.3.4 or older,
    START.cmd asks once whether to add the file for images (about 0.64 GB): Enter alone downloads it, n keeps text only;
    later: START.cmd --image-file. A folder you downloaded or converted yourself still opens with
    START.cmd --model-dir <folder>. The steps are in the section
    "Inkling-2 Mini" of README.md in the repository.
  - The manual steps and the full documentation are at https://github.com/yasakani-dev/yasakani

License: Apache License 2.0 (see the repository).
