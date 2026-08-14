import os
import datetime
import torch
import torch.distributed as dist

if __name__=='__main__':
    rank=int(os.environ['RANK'])
    world=int(os.environ['WORLD_SIZE'])
    local=int(os.environ['LOCAL_RANK'])
    addr=os.environ['MASTER_ADDR']
    port=os.environ['MASTER_PORT']
    torch.cuda.set_device(local)
    dist.init_process_group('gloo', init_method=f'tcp://{addr}:{port}?use_libuv=0', rank=rank, world_size=world, timeout=datetime.timedelta(seconds=120))
    t=torch.tensor([rank],device=f'cuda:{local}',dtype=torch.float32)
    dist.all_reduce(t)
    print(f'rank={rank} local={local} sum={t.item()}', flush=True)
    dist.destroy_process_group()
