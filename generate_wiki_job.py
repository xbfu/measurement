import os
import argparse
import jinja2


def generate_yaml(job_name, model_name, num_workers, ss, batch_size, lr, port=29600):
    # service
    with open('j2/wiki_service.j2', 'r') as f:
        buffer = f.read()

    service = {'job_name': job_name, 'port': port}
    template = jinja2.Template(buffer)

    with open(f'./yaml/{job_name}_{ss}.yaml', mode='w') as file:
        file.write(template.render(service))

    # ps
    with open('j2/wiki_ps.j2', 'r') as f:
        buffer = f.read()

    ps = {'job_name': job_name, 'model_name': model_name, 'rank': 0, 'num_workers': num_workers,
          'ss': ss, 'batch_size': batch_size, 'node': '00', 'lr': lr, 'port': port, 'cpu': -1}
    template = jinja2.Template(buffer)

    with open(f'./yaml/{job_name}_{ss}.yaml', mode='a') as file:
        file.write(template.render(ps))

    # worker
    with open('j2/wiki_worker.j2', 'r') as f:
        buffer = f.read()

    for rank in range(1, num_workers + 1):
        node = f'0{rank}'
        worker = {'job_name': job_name, 'model_name': model_name, 'rank': rank, 'num_workers': num_workers,
                  'ss': ss, 'batch_size': batch_size, 'node': node, 'lr': lr, 'port': port, 'cpu': -1}
        template = jinja2.Template(buffer)
        with open(f'./yaml/{job_name}_{ss}.yaml', mode='a') as file:
            file.write(template.render(worker))

    # tester
    with open('j2/wiki_tester.j2', 'r') as f:
        buffer = f.read()

    tester = {'job_name': job_name, 'model_name': model_name, 'rank': (num_workers + 1), 'num_workers': num_workers,
              'ss': ss, 'batch_size': batch_size, 'node': '05', 'lr': lr, 'port': port, }
    template = jinja2.Template(buffer)

    with open(f'./yaml/{job_name}_{ss}.yaml', mode='a') as file:
        file.write(template.render(tester))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate yaml file")
    parser.add_argument("--job_name", type=str, default="transformer", help="The job's name.")
    parser.add_argument("--model_name", type=str, default="transformer", help="The model's name.")
    parser.add_argument("--num_workers", type=int, default=4, help="Total number of workers.")
    parser.add_argument("--ss", type=str, default="asgd", help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size of each worker during training.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")

    args = parser.parse_args()

    # job_name = ['transformer', 'lstm']
    job_name = args.job_name
    model_name = job_name
    num_workers = args.num_workers
    ss = args.ss
    batch_size = args.batch_size
    lr = args.lr
    generate_yaml(job_name=job_name, model_name=model_name, num_workers=num_workers,
                  ss=ss, batch_size=batch_size, lr=lr)
    os.system(f'sudo kubectl apply -f yaml/{job_name}_{ss}.yaml')
