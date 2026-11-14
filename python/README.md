# yasakani (Python package)

Run **Qwen3.8-27B** on one **NVIDIA RTX 5090** under **Windows** with the YASAKANI engine library (`yskn.dll`).

This package is the part you talk to: a local **OpenAI-compatible server**, a **terminal chat**, a **benchmark**, and two checks that tell you
whether your files and your installation are the reference ones. The engine itself is the DLL; the package loads it, applies the
default settings the published numbers were measured with, handles the chat template and the tokenizer, and streams the answer.

Status: version `0.1.2`. Text, and optionally still images (new in 0.1.2, Windows only, see [Image input](#image-input-optional)). One request is generated at a time. Tool calling is new in 0.1.1.

## What you need

| | |
|---|---|
| GPU | NVIDIA **RTX 5090** (the engine is built for this card only) and a current NVIDIA driver |
| OS | **Windows 11**, 64-bit (the only version tested), with the *Microsoft Visual C++ 2015-2022 x64 runtime* ([download](https://aka.ms/vs/17/release/vc_redist.x64.exe)) |
| Python | 3.10 or newer |
| Disk | about 15 GB for the quality model folder, about 13 GB for the fastest model folder |
| Engine library | `yskn.dll` from the release page |

You do **not** need the CUDA Toolkit, PyTorch, or a C++ compiler.

## Model folder

YASAKANI publishes two model folders; you need one. Both hold the same files except for the weights:

```
qwen38_27b_quality.yskn   the weights of the quality model (12.4 GB)   - or -
qwen38_27b_speed.yskn     the weights of the fastest model (10.3 GB)
bdraft2_k8.yskn           small draft model used to speed up generation (1.2 GB)
bdraft2_k8.shead          its output table (1.4 GB)
kvq_u5.bin                settings of the compressed key-value cache (17 KB)
tokenizer.json            } from the Qwen3.8-27B repository, unchanged
tokenizer_config.json     }
chat_template.jinja       }
yasakani.json             file list with SHA-256 checksums; it also names the weights file
selfcheck.json            reference output for `yasakani selfcheck` (see below)
```

## Install

```bat
pip install .
yasakani info --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

`info` checks Python, the three libraries (numpy, tokenizers, jinja2), the DLL, the GPU and the model folder, and tells you what is missing.
Put `yskn.dll` in a `bin` folder next to the package, or point to it with `--dll` / the `YASAKANI_DLL` environment variable. The model folder
can be given with `--model-dir` or `YASAKANI_MODEL_DIR`.

## Commands

| command | what it does |
|---|---|
| `yasakani info` | checks this computer, the DLL and (with `--model-dir`) the model folder |
| `yasakani verify` | compares every model file with the checksums in `yasakani.json` (reads 13 to 15 GB, up to a minute or two) |
| `yasakani selfcheck` | runs three short fixed greedy prompts and compares the output with `selfcheck.json`: *does this installation reproduce the reference output?* |
| `yasakani chat` | chat in the terminal. `/think on\|off`, `/effort low\|medium\|xhigh`, `/reset`, `/exit`; Ctrl+C stops the answer being written |
| `yasakani serve` | starts the OpenAI-compatible server (below) |
| `yasakani bench` | speed measurement: greedy, fixed length, your own prompts with `--prompts-file`; `--json` for a machine-readable result |

Loading the model takes about a minute; `serve` answers `/health` with `503 loading` until it is ready.

## The server

```bat
yasakani serve --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

It listens on `127.0.0.1:8765` (this computer only). Point any OpenAI-compatible client at `http://127.0.0.1:8765/v1`; the model id is `qwen3.8-27b`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")
r = client.chat.completions.create(model="qwen3.8-27b", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.reasoning_content)   # the thinking (the model thinks first by default)
print(r.choices[0].message.content)             # the answer
```

```bat
curl http://127.0.0.1:8765/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}"
```

Supported request fields: `messages` (text, and images only with the optional image input below; the roles `system`, `developer`, `user`, `assistant` and `tool`), `tools` with `tool_choice`
`auto` or `none`, `max_tokens` / `max_completion_tokens`, `temperature` (0-2; 0 is greedy), `top_p`, `top_k` (at most 20),
`min_p`, `seed`, `stop` (up to 4 strings), `stream` with `stream_options.include_usage`, `reasoning_effort` (`low`, `medium`, `high`, `xhigh`),
`chat_template_kwargs` (for example `{"enable_thinking": false}`), `ignore_eos`, `loop_guard`. The defaults are the model's own:
temperature 1.0, top_p 0.95, top_k 20. Not supported (answered with 400): images when the server runs without the image input options, the old `functions` / `function_call` fields, a forced `tool_choice`, `n` other than 1, `logprobs`.

Tool calling (new in 0.1.1): send `tools` (functions) with `tool_choice` `"auto"` (the default) or `"none"`. The model's own chat template
describes the tools to the model. The calls it writes come back as `message.tool_calls`, and `finish_reason` is `tool_calls`; in a stream each
finished call arrives as one chunk with the whole arguments. Send the results back as messages with the role `tool`. An input that does
not fit into the context is answered with 400 and the error code `context_length_exceeded` before any text is sent. Also accepted:
`thinking_budget` (a number of tokens after which the thinking is closed and the answer starts; 0 = no limit) and
`"thinking": {"type": "enabled"}` / `{"type": "disabled"}`. Tool calling is new and has been tested far less than plain chat.

```python
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather of a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = client.chat.completions.create(model="qwen3.8-27b", messages=[{"role": "user", "content": "How is the weather in Osaka?"}], tools=tools)
print(r.choices[0].message.tool_calls)
```

* The thinking comes back as `reasoning_content` (and as `delta.reasoning_content` when streaming), the answer as `content`.
* By default the model thinks first, at its longest thinking setting, and `max_tokens` counts the thinking too. With a small limit a request can
  end while the model is still thinking (`finish_reason` is `length` and `content` is empty). Raise the limit, or shorten the thinking with
  `reasoning_effort` (`medium`, `low`), or switch it off with `{"enable_thinking": false}` in `chat_template_kwargs`.
* A stream carries one chunk per token. Where one character is split over several tokens, its text comes with the last of them, so the number
  of chunks is slightly smaller than the number of tokens; the exact count is `usage.completion_tokens` (ask for it with
  `stream_options: {"include_usage": true}`). The chunks arrive in small groups, about 64 times a second.
* **Loop guard.** If the thinking falls into an endless repetition, the guard cuts it, asks the model to answer from what it has written, and reports this in
  `usage.loop_guard`. Pass `"loop_guard": false` to turn it off.
* Input, thinking and answer together are limited to the context length (262,144 tokens by default; `--max-ctx` lowers it and the memory it needs).
* Asking again with the earlier conversation in front of a new question does not recompute the earlier turns (`usage.prompt_tokens_details.cached_tokens`).

### Image input (optional)

Image input works on **Windows only**. The engine library already has the entry for image features (`yskn_set_visual`); the server feeds it
when it is started with three more files. Without them the server is text only, as before: a request with an image is answered with 400 at
once, without waiting for a running generation.

| | |
|---|---|
| `--mmproj` | the projector of Qwen3.8-27B: `mmproj-Qwen3.8-27B-BF16.gguf` from [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) (931,145,888 bytes, sha256 `de2a49866988ea272c43dc2a43ee2662c25725a717ca5b9059ea420a97f53fe8`) |
| `--llama-dir` | a folder with the unpacked `llama-b10488-bin-win-cpu-x64.zip` of the [llama.cpp release b10488](https://github.com/ggml-org/llama.cpp/releases/tag/b10488) (18,490,962 bytes, sha256 `6c938f6d79aac96cb90fda673aade20cff9b1b6c1e97de04f4d5d60bca107082`); its multimodal library turns the image into features. Only this CPU build is accepted: at start the server compares its `llama.dll`, `mtmd.dll`, `ggml.dll` and `ggml-base.dll` by SHA-256 and refuses a folder with GPU backends (`ggml-cuda.dll` and the like) |
| `--vision-model` | a Qwen3.8-27B GGUF. Only its settings and vocabulary are read, never its weights. We tested `Qwen3.8-27B-Q4_K_M.gguf` of [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) (17,442,399,968 bytes); other GGUF files of this model should work the same way, but we have not tried them |
| Pillow | install the package with `pip install ".[vision]"` instead of `pip install .`, or add Pillow to the Python that runs `yasakani` with `pip install Pillow` (in a quick-start folder: `venv\Scripts\python -m pip install Pillow`) |

```bat
pip install ".[vision]"
yasakani serve --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll ^
  --mmproj D:\gguf\mmproj-Qwen3.8-27B-BF16.gguf --llama-dir D:\llama-b10488 --vision-model D:\gguf\Qwen3.8-27B-Q4_K_M.gguf
```

These options are checked at start, before the model loads; a missing file, the projector and the GGUF given the wrong way round, another
llama.cpp build or a missing Pillow stops the server with an error message. Relative paths are taken from the folder the server is started in.

Send the image as an OpenAI `image_url` part with a `data:image/...;base64,` URI, in a user message. Limits:

* at most 16 images per request. The whole request body is at most 32 MiB and base64 makes the data a third larger, so the images of one
  request can be about 24 MiB together;
* each image at most 40 million pixels, and its longer side at most 200 times its shorter side;
* each image is downsized to at most **1,024 image tokens** (about one million pixels), keeping its shape: a 1920 × 1080 screenshot
  becomes 1,008 image tokens and a 4032 × 3024 phone photo 972 (at full size they would take 2,040 and 4,015). Downsizing loses detail,
  so small text in a full-screen screenshot can become hard to read; crop the part you need, or raise the limit.
  `--vision-max-image-tokens N` sets it, from 64 to 4,096 (the projector's own limit: images up to about 4 million pixels keep their size).
  Below 256 a very long, thin image can stay a little above the limit (113 image tokens at 64);
* at most 8,192 image tokens per request, 8 times the per-image limit (`--vision-max-request-image-tokens N` changes it). The images
  of earlier turns count too, because a client sends the whole conversation again with every turn.

An image outside these limits, or one that cannot be decoded, is answered with 400 before it is encoded. When the images make the input longer than
the context, the answer is 400 with the error code `context_length_exceeded`, as for long text. `detail` is accepted and ignored.
`--vision-allow-local-files DIR` also accepts local file paths (and `file:` URLs) of files inside the folder DIR; remote URLs are never fetched.
`--vision-cache FOLDER` keeps the features of images seen before, one file per image: it holds information about every image sent to the
server, so delete the folder when you no longer want it.

* The image is encoded on the CPU; generation then runs on the GPU as usual. The encode time grows with the square of the image tokens.
  On our PC (AMD Ryzen 9 9950X3D2, 16 cores) with the default of one thread per physical core (16 threads): 1.2 s for a 512 × 512 image
  (256 image tokens), 12 s for 1280 × 720 (920), 12 to 15 s for any image downsized to the default limit (1920 × 1080, 2048 × 2048,
  4032 × 3024). At full size the same encoder took 51 s for a 1920 × 1080 screenshot with 16 threads and 5 1/2 minutes for a
  2048 × 2048 image with 8 threads. `--vision-threads N` sets the number of threads: with 32 (two per core) the same 1920 × 1080
  image took 10.9 s instead of 13.5 s, with the same features.
* **While an image is encoded, the server answers no other request**, and the client gets nothing back yet, not even the start of a
  stream: a request with eight large images waits about 2 minutes (8 × 14 s) for its first token. Give the client a long enough timeout.
  An image seen before is not encoded again: the server keeps the features of up to 512 MiB of images in memory (about 50 images at
  the default limit), and `--vision-cache FOLDER` also keeps them on disk.
* Image understanding was measured by the contributor of this feature (pull request #4) with their own test of 5 levels and 49 questions,
  which is not published: **87.8** with the quality model, against 89.8 for the original BF16 model in llama.cpp. With the projector on the GPU
  their score fell to 79.6 and 81.6 in two runs, which is why the projector runs on the CPU here. We have not measured image quality ourselves; we checked that it works on an
  RTX 5090 with the quality model and a few simple test images.
* Image tokens get Qwen's three-axis positions (time, row, column); the text after an image continues from the largest of them.
* Requests without images are handled as before: the same prompt tokens and the same answers. After a request with an image the engine
  starts the next prompt from the beginning (no reuse of the earlier turns).

### Safety of the server

* By default only this computer can connect. The `Host` header must name the address the server is bound to, and a browser request from another web page
  (`Origin` / `Sec-Fetch-Site`) is refused, so a web page open in your browser cannot use your GPU.
* To serve other computers use `--host 0.0.0.0 --api-key YOUR_KEY` (or `YASAKANI_API_KEY`). A non-local address without a key is refused at start.
  The server speaks plain HTTP: put it behind a VPN or a TLS proxy when it leaves your LAN.

## The self-check

`selfcheck.json` holds, for three short fixed prompts, a hash of the prompt ids and a hash of the greedy output ids made with the reference installation.
`yasakani selfcheck` produces the same records on your computer and compares them. A *prompt ids differ* result means the tokenizer or template files are
not the reference ones; an *output differs* result means the weights, the DLL, or the GPU/driver behave differently from the reference. Send the result
with your report in that case.

The three prompts are short on purpose. Output is not guaranteed to be bit-identical from run to run while other programs use the GPU at the
same time (see `docs/MEASUREMENTS.md` in the repository). In our checks the three self-check requests were identical in all 180 repeats and
the command passed 12 of 12 times, but if a case fails, run the command once more before you report it.

## Use it from Python

```python
from yasakani import Engine, ModelPack
from yasakani.chat import ChatParams, ChatRunner
from yasakani.textio import ChatFormatter

pack = ModelPack.load(r"D:\models\qwen38-quality")
with Engine(pack, max_ctx=32768, dll=r"D:\yasakani\yskn.dll") as eng:
    runner = ChatRunner(eng, ChatFormatter(pack.root))
    runner.warmup()
    params = ChatParams(messages=[{"role": "user", "content": "Hello!"}], max_tokens=512, template_kwargs={"enable_thinking": False})
    usage = runner.run(params, lambda kind, text: print(text, end="", flush=True))   # kind is "reasoning" or "content"
    print(usage.completion_tokens, "tokens")
```

`Engine.generate(prompt_ids, max_new, on_tokens=...)` works at the id level; `Engine.cancel()` stops a running generation from any thread.

## Settings and environment

The package sets the engine's environment variables itself (`yasakani.runtime.DEFAULT_ENV`) and removes any other `YSKN_*` variable left in your shell
before it opens the model, because a stray setting silently changes the engine. The published results were measured with the defaults; changing them
means running an engine configuration that the self-check and the published numbers do not cover.

## Troubleshooting

* *engine library not found / could not be loaded* - give `--dll`, and install the Visual C++ runtime linked above.
* *could not open the model ... not enough free GPU memory* - close other programs that use the GPU, or lower `--max-ctx`.
* *much slower than the published numbers* - look for the warning *the GPU memory is almost full* at start. When other programs use GPU memory
  and it runs out, Windows keeps part of the model in ordinary memory; there is no error, but generation becomes several times slower.
  Close the other programs, or lower `--max-ctx`.
* *only thinking, no answer* - the length limit was reached while the model was still thinking; see the note on `max_tokens` above
  (`/effort medium`, `/effort low` or `/think off` in `yasakani chat`).
* *cannot listen on 127.0.0.1:8765* - another program (or a second server) uses the port; pick another with `--port`.
* *output differs in selfcheck* - run `yasakani verify` first (a damaged download is the most common cause), then run the self-check once more.

## Tests

```
python -m unittest        # no GPU needed: the engine library is replaced by a small stand-in (tests/stub)
```

Set `YASAKANI_TEST_MODEL_DIR` to a folder containing the Qwen3.8 `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` to run the chat, server and
command-line tests as well.

## License

Apache License 2.0.
