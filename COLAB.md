The current notebook is [SDF Qwen controlled experiments](https://colab.research.google.com/drive/1jdnWe7EMSzLQPSVi7VU16yiQeO1iz4m-).
It uses an NVIDIA A100-SXM4 with 40 GB VRAM. No SSH or project API keys are needed.

The experiment branch is `codex/colab-experiments`. It includes the 575 existing
synthetic documents. `.env`, model caches, adapters, connection tokens, and
local rental state are ignored. No additional training data has been generated.

The portable notebook is `notebooks/colab_experiments.ipynb`. Its launch cell
checks for an active job and preserves previous logs and results before starting
a new pair of experiments. Select an A100 in Runtime > Change runtime type.
Download results using the final cell after the job completes.

Google's official [Colab MCP](https://github.com/googlecolab/colab-mcp) is installed
in the project virtual environment and registered in Codex. New Codex sessions
can use the registered server. The current session uses `colab_bridge.py`, an
MCP client bridge to the same official implementation, so it can work without
restarting Codex. Only one notebook can connect to each server instance.

To reproduce the local setup with Python 3.13 and uv:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-colab-mcp.txt
.venv/Scripts/python.exe colab_bridge.py
```

The bridge writes its ephemeral token and port to `.colab/connection.json`.
Append that fragment to the notebook URL, reload, and accept Colab's Connect
dialog. Chrome must allow local network access for `colab.research.google.com`.
`colab_mcp_local.py` fixes the official server's Windows dual-stack ephemeral
port mismatch by binding one IPv4 loopback socket. It preserves authentication
and Colab origin checks.

From another project terminal, Python can list tools, add code cells, and run
them through MCP:

```powershell
.venv/Scripts/python.exe colab_control.py list
.venv/Scripts/python.exe colab_control.py add_code_cell --code-file your_cell.py --index 0
.venv/Scripts/python.exe colab_control.py run_code_cell --cell-id ID_FROM_PREVIOUS_CALL
```

`--args-file` accepts JSON arguments for other tools, including `get_cells` and
`update_cell`. Responses are preserved in ignored `.colab/responses/`.

The notebook clones the branch, then starts `experiments/colab_job.py` in a
background Python process. That script installs training dependencies, runs
independent 120-step adapters at learning rates 5e-5 and 2e-4, and records status
in `colab_status.json` with logs in `colab_job.log`. It limits total job time to
45 minutes and archives results on completion or failure. `colab_results.zip`
contains metrics, final adapters, tokenizer files, and logs; intermediate
checkpoints remain in the runtime. Existing result directories are never
overwritten: use a new experiment name for each iteration.

To iterate, edit locally, commit and push the experiment branch, then run
`git pull --ff-only` in a notebook code cell through MCP before launching a new
experiment. A new subprocess imports the updated code. Read `EXPERIMENTS.md`
for the evaluation design and criteria for meaningful behavioral transfer.

For a bounded longer run using the same documents, a notebook Python cell can
call `experiments/start_colab.py --experiment long_lr 2e-4 360 --eval-every 120
--minutes 25`. Labels receive a timestamp automatically; the active-job guard
and preservation of previous results apply to custom experiments too.
