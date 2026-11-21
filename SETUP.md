# Colab GPU setup (tested 2026-11-20)

Local: macOS 14.5 arm64, Python 3.13.12 (miniforge), uv 0.12.23, pip 26.0.1, git 2.55.0.
No tokens or credentials are stored in this repo. The CLI keeps its own auth under `~/.config/colab-cli/`.

## One-time setup
```bash
uv tool install google-colab-cli      # installs `colab` to ~/.local/bin
colab sessions                        # first run: prints a Google URL; open it, sign in,
                                      # paste the authorization code back into the terminal
                                      # (needs a real interactive terminal)
```

## SSH key (Ed25519 required)
The server rejects RSA keys, and `~/.ssh/google_compute_engine` is RSA.
```bash
ssh-keygen -q -t ed25519 -N "" -C colab-hack -f ~/.ssh/colab_ed25519
```
`~/.ssh/config` entry:
```
Host colab-runtime
    User root
    IdentityFile ~/.ssh/colab_ed25519
    ProxyCommand colab ssh --proxy-mode -s hack -i ~/.ssh/colab_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```
Note: `ProxyCommand ... -s hack` will create a session named `hack` if none exists. Add `--gpu T4`
to the ProxyCommand if you want it to be a GPU one. Without it you get a CPU runtime.

## Start / stop a GPU session
```bash
colab new -s hack --gpu T4     # T4 only unless approved; takes ~seconds
colab sessions                 # list (a "[?]" row = runtime not in local state)
colab stop -s hack             # ALWAYS do this when done
```

## Run things
```bash
echo '!nvidia-smi' | colab exec -s hack          # shell line via "!"
colab exec -s hack -f script.py                  # local .py runs on the runtime, no upload needed
ssh colab-runtime nvidia-smi                     # via SSH (runs in the kernel's VM as root)
colab install -s hack <pkg>                      # pip-style install on the VM (see `colab help install`)
```
cupy is preinstalled on the T4 runtime. No `pip install cupy-cuda12x` was needed.

## Copy files
```bash
colab upload   -s hack local.txt /content/local.txt
colab download -s hack /content/out.txt ./out.txt
colab ls       -s hack /content
scp colab-runtime:/content/out.txt .             # also works
scp file colab-runtime:/content/
```

## Test results
- `nvidia-smi`: Tesla T4, 15360 MiB, driver 580.82.07, CUDA 13.0.
- cupy check: `45 b'Tesla T4'`
- Timing: 4000x4000 float32, 200 iterations of `a = 0.5*(a + roll(a, 1, axis=0))`, GPU synchronised before stopping the timer, 3 warm-up iterations first:

| | time |
|---|---|
| NumPy on the runtime's CPU (2 vCPU) | 14.168 s |
| CuPy on T4 | 0.364 s |
| speedup | ~38.9x (results identical, max diff 0) |

- `ssh colab-runtime nvidia-smi`: works. `scp` works.

## Errors hit and fixes
1. **Sign-in aborted in a non-interactive shell** ("Enter the authorization code ... Aborted"). Run `colab sessions` in your own terminal and complete the browser step.
2. **`Server rejected pubkey: unsupported key type`**. The only key present was RSA. Generate Ed25519 (above) and point `-i` / `IdentityFile` at it.

## Not done
- **Colab MCP server (step 5):** skipped. https://github.com/googlecolab/colab-mcp is a bridge to a notebook open in a
  browser tab (it opens colab.research.google.com and a localhost WebSocket). It has no create or delete runtime tools,
  so the requested test is not possible. Its documented config is an `mcpServers` block running
  `uvx git+https://github.com/googlecolab/colab-mcp`. Not installed.

## Housekeeping
- All runtimes created in this test (`hack`) were stopped. `colab sessions` afterwards showed only
  `gpu-t4-s-use1c0-ilszsz0mj0ce`, a T4 runtime that was already running before this test and was NOT created by it.
  Stop it yourself if you don't need it (it may be in the Colab web UI under Runtime > Manage sessions).
- The first `~/.ssh/config` had no backup needed (`~/.ssh/config.bak-colab` is created only if a config existed).
