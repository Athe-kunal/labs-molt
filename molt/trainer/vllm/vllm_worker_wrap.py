class WorkerWrap:
    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl"):
        """Init torch process group for model weights update"""
        import torch
        from molt.utils.distributed_util import stateless_init_process_group

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weights_packed(self, metas):
        """Receive ONE packed broadcast carrying many weights.

        ``metas`` is a list of ``(name, dtype, shape)``. Producer (rank 0 in
        the trainer) cats all tensors into a single uint8 buffer in the same
        order; here we split + reinterpret-cast back. Replaces thousands of
        per-tensor RPC+broadcast pairs with a handful of ~1 GiB ones.
        """
        import math

        import torch

        sizes = []
        for _, dtype, shape in metas:
            assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
            sizes.append(math.prod(shape) * dtype.itemsize)

        buf = torch.empty(sum(sizes), dtype=torch.uint8, device="cuda")
        self._model_update_group.broadcast(buf, src=0, stream=torch.cuda.current_stream())

        weights = [
            (name, part.view(dtype).view(*shape)) for (name, dtype, shape), part in zip(metas, buf.split(sizes))
        ]
        self.model_runner.model.load_weights(weights=weights)
        del buf
