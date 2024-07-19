import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from img2img import Unet
from prep import printProgressBar
from networks import Generator
from measure import compute_measure
import math
from tqdm.auto import tqdm
from einops import rearrange, reduce
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', linear_beta_schedule(T))
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        p2_loss_weight_k = 1
        p2_loss_weight_gamma = 0.
        self.register_buffer('p2_loss_weight',
                    (p2_loss_weight_k + alphas_bar / (1 - alphas_bar)) ** -p2_loss_weight_gamma)

    def forward(self,y0,x):
        """
        Algorithm 1.
        """
        # t = torch.randint(
        #     low=0, high=self.T, size=(y0.shape[0] // 2 + 1,), device=y0.device
        # )
        # t = torch.cat([t, self.T - t - 1], dim=0)[:y0.shape[0]]
        t = torch.randint(0,self.T, size=(y0.shape[0], ), device=y0.device)
        noise = torch.randn_like(y0)
        y_t = (
            extract(self.sqrt_alphas_bar, t, y0.shape) * y0 +
            extract(self.sqrt_one_minus_alphas_bar, t, y0.shape) * noise)
        loss = F.l1_loss(self.model(y_t, x,t), y0, reduction='none')
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev', 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        # self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.sampling_timesteps=100
        self.ddim_sampling_eta= 1.
        self.register_buffer(
            'betas', linear_beta_schedule(T))
        print(self.betas)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('alphas_bar', alphas_bar)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape)
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t,ldct, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t,ldct ,t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t,ldct ,t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':# the model predicts epsilon
            eps = self.model(x_t,ldct, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var
    def ddim_sample(self, img, ldct):
        batch, device, total_timesteps, sampling_timesteps, eta = img.shape[0], self.betas.device, self.T, self.sampling_timesteps, self.ddim_sampling_eta


        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # img = torch.randn(shape, device = device)

        x_start = None
        for time, time_next in tqdm(time_pairs, desc = 'ddim_sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # self_cond = x_start if self.self_condition else None
            # pred_noise = self.model(img,ldct, time_cond)
            # x_start = self.predict_xstart_from_eps(img, time_cond, eps=pred_noise)
            # x_start = torch.clip(x_start, -1., 1.)
            x_start = self.model(img,ldct ,time_cond)
            x_start=torch.clip(x_start, -1., 1.)
            pred_noise = self.predict_noise_from_start(img, time_cond, x_start)
            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_bar[time]
            alpha_next = self.alphas_bar[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img
    def forward(self, x_T,ldct):
        """
        Algorithm 2.
        """

        x_0=self.ddim_sample(x_T,ldct)
        return torch.clip(x_0, -1., 1.)
        # x_t = x_T
        # for time_step in reversed(range(self.T)):
        #     t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
        #     mean, log_var = self.p_mean_variance(x_t=x_t,ldct=ldct, t=t)
        #     # no noise when t == 0
        #     if time_step > 0:
        #         noise = torch.randn_like(x_t)
        #     else:
        #         noise = 0
        #     x_t = mean + torch.exp(0.5 * log_var) * noise
        #
        # x_0 = x_t
        # return torch.clip(x_0, -1., 1.)
class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader

        self.beta_1, self.beta_T, self.T = args.beta_1,args.beta_T,args.T
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max
        self.T=args.T
        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        self.GeneratorUNet =  Generator(T=self.T)

        # if (self.multi_gpu) and (torch.cuda.device_count() > 1):
        #     print('Use {} GPUs'.format(torch.cuda.device_count()))
        #     self.Generator = nn.DataParallel(self.Generator)
        self.GeneratorUNet.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.GeneratorUNet.parameters(),betas=(0.9, 0.99))
        self.trainer = GaussianDiffusionTrainer(
        self.GeneratorUNet, args.beta_1, args.beta_T, args.T).to(self.device)

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'GeneratorUNet_{}iter.ckpt'.format(iter_))
        torch.save(self.GeneratorUNet.state_dict(), f)

    def normalize_to_neg_one_to_one(self,img):
        return img * 2 - 1

    def unnormalize_to_zero_to_one(self,img):
        return (img + 1) * 0.5
    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'GeneratorUNet_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.GeneratorUNet.load_state_dict(state_d)
        else:
            self.GeneratorUNet.load_state_dict(torch.load(f))


    def lr_decay(self):
        lr = self.lr * 0.5
        print(lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()


    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()
        for epoch in range(1, self.num_epochs):
            self.GeneratorUNet.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(1).float().to(self.device)
                y = y.unsqueeze(1).float().to(self.device)
                x=self.normalize_to_neg_one_to_one(x)
                y = self.normalize_to_neg_one_to_one(y)
                self.optimizer.zero_grad()
                loss = self.trainer(y,x).mean()
                # pred = self.Generator(x)
                # loss = self.criterion(pred, y)
                # # print(loss)
                # self.Generator.zero_grad()


                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(),
                                                                                                        time.time() - start_time))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))


    def test(self):
        # del self.GeneratorUNet
        # # load
        self.GeneratorUNet = Generator(T=self.T).to(self.device)
        self.load_model(self.test_iters)
        self.sampler = GaussianDiffusionSampler(
        self.GeneratorUNet, self.beta_1, self.beta_T, self.T, img_size=self.patch_size,
        mean_type='epsilon', var_type='fixedlarge').to(self.device)
        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        self.GeneratorUNet.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                x=self.normalize_to_neg_one_to_one(x)
                x_T = torch.randn((1, 1, 512, 512)).to(self.device)
                pred = self.sampler(x_T,x)
                pred =self.unnormalize_to_zero_to_one(pred)
                x = self.unnormalize_to_zero_to_one(x)
                print('finish one')
                # pred = self.GeneratorUNet(x)

                # denormalize, truncate
                # x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                # y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                # pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))
                x = x.view(shape_, shape_).cpu().detach()
                y = y.view(shape_, shape_).cpu().detach()
                pred = pred.view(shape_, shape_).cpu().detach()
                data_range = self.trunc_max - self.trunc_min
                original_result, pred_result = compute_measure(x, y, pred, 1.0)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))
                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                            ori_ssim_avg/len(self.data_loader), 
                                                                                            ori_rmse_avg/len(self.data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                  pred_ssim_avg/len(self.data_loader), 
                                                                                                  pred_rmse_avg/len(self.data_loader)))
