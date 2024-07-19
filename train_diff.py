
from img2img import Unet, GaussianDiffusionImg, Trainer
if __name__ == "__main__":
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels=1,
        self_condition=True,
        condition_channel=1,
    ).cuda()

    diffusion = GaussianDiffusionImg(
        model,
        image_size = 256,
        timesteps = 100,           # number of steps
        sampling_timesteps = 100,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1' ,           # L1 or L2
        objective="pred_x0"
    ).cuda()

    trainer = Trainer(
        diffusion,
        r'./data/img2/train/fd',
        r'./data/img2/val/fd',
        train_batch_size = 4,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        save_and_sample_every=1000,
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                      # turn off mixed precision
        num_workers=4,
        num_samples=4,
        results_folder="./results_qd_frfd",
        keys=["fd",  "qd_frft"]#"qd",
    )

    trainer.train()