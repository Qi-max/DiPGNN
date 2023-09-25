from dipgnn.utils.common_util import CommonArgs
from dipgnn.utils.register import registers

if __name__ == "__main__":
    args = CommonArgs().get_args()
    task = registers.task.get_class(args.task_name)(args)
    task.run()
