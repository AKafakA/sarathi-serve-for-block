from sarathi.config import SystemConfig, ModelConfig, ReplicaConfig, ParallelConfig, BaseSchedulerConfig, \
    VllmSchedulerConfig, MetricsConfig, WorkerConfig, CacheConfig
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine


class LLMEngine:

    @classmethod
    def from_system_config(cls, config: SystemConfig) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        if config.parallel_config.pipeline_parallel_size > 1:
            engine = PipelineParallelLLMEngine(config)
        else:
            engine = BaseLLMEngine(config)

        return engine

    @classmethod
    def from_cli_args(cls,
                      model_name,
                      batch_size,
                      pipeline_parallel_size,
                      tensor_parallel_size,
                      output_dir,
                      gpu_utilization=0.9,
                      block_size=16,
                      max_len=4096):

        replica_config = ReplicaConfig(
            replica_id=0,
            output_dir=output_dir
        )
        # model config
        model = model_name
        model_config = ModelConfig(
            model=model_name,
            dtype="float16",
            trust_remote_code=True,
            max_model_len=max_len
        )
        parallel_config = ParallelConfig(
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size
        )

        scheduler_config = VllmSchedulerConfig(
            max_num_seqs=batch_size
        )
        metrics_config = MetricsConfig(
            enable_op_level_metrics=False,
            enable_cpu_op_level_metrics=True,
            keep_individual_batch_metrics=False,
        )
        worker_config = WorkerConfig(
            gpu_memory_utilization=gpu_utilization
        )
        cache_config = CacheConfig(
            block_size=block_size
        )

        system_config = SystemConfig(
            replica_config=replica_config,
            model_config=model_config,
            parallel_config=parallel_config,
            metrics_config=metrics_config,
            scheduler_config=scheduler_config,
            worker_config=worker_config,
            cache_config=cache_config
        )
        return cls.from_system_config(system_config)
