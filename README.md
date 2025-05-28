# Profiling of Vidur

## Steps to Follow

1. **Install Sarathi-Serve**
   ```bash
   cd sarathi-serve-for-block
   pip install -e . --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6
   ```

2. **Add the New Model Config**
   - Update the configuration under `vidur: model_config.py`.

3. **Build Vidur**
   ```bash
   cd vidur_opt_scheduler
   pip install -e .
   ```

4. **Profiling**
   ```bash
   python vidur/profiling/mlp/main.py --models Qwen/Qwen-2-7B --num_gpus 1
   python vidur/profiling/attention/main.py --models Qwen/Qwen-2-7B --num_gpus 1
   ```
