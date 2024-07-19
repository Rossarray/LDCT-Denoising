import os
import argparse

from torch.backends import cudnn
from loader import get_loader
from solver import Solver


def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'results')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='/media/Data/lyl/code/D45_3mm/ori_d45_3mm')
    parser.add_argument('--saved_path', type=str, default=r'/media/Data/lyl/code/D45_3mm/npy_d45_3mm')
    parser.add_argument('--save_path', type=str, default='/media/Data/lyl/code/mywork_ddpm/save_trans')
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--result_fig', type=bool, default=True)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2)

    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--print_iters', type=int, default=1)


    parser.add_argument('--decay_iters', type=int, default=40000)
    parser.add_argument('--save_iters', type=int, default=10000)
    parser.add_argument('--test_iters', type=int, default=380)
    parser.add_argument('--beta_1', type=float, default=1e-4, help='start beta value')
    parser.add_argument('--beta_T', type=float, default=0.02, help='end beta value')
    parser.add_argument('--T', type=int,default=1000, help='total diffusion steps')
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    main(args)
