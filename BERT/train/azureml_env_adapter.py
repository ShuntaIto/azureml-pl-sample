import os

def set_environment_variables():
    os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
    os.environ["MASTER_PORT"] = "6105"

    # node rank is the world rank from mpi run
    os.environ["NODE_RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    print("MASTER_ADDR = {}".format(os.environ["MASTER_ADDR"]))
    print("MASTER_PORT = {}".format(os.environ["MASTER_PORT"]))
    print("NODE_RANK = {}".format(os.environ["NODE_RANK"]))
    
def set_environment_variables_for_nccl_backend(single_node=False, master_port=6105):
    if not single_node:
        master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
        os.environ["MASTER_ADDR"] = master_node_params[0]

        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)
        
        try:
            os.environ["NODE_RANK"] = os.environ[
                "OMPI_COMM_WORLD_RANK"
            ]  # node rank is the world_rank from mpi run
        except:
            pass

    else:
        os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        os.environ["MASTER_PORT"] = "54965"
        os.environ["NODE_RANK"] = "0"

    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"

    os.environ["HOROVOD_GPU_OPERATIONS"] = "NCCL"

    print("MASTER_ADDR = {}".format(os.environ["MASTER_ADDR"]))
    print("MASTER_PORT = {}".format(os.environ["MASTER_PORT"]))
    print("NODE_RANK = {}".format(os.environ["NODE_RANK"]))