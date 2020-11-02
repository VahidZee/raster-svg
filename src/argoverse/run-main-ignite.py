from model_and_utils import *

def run(
    epochs,
    lr,
    log_interval,
    log_dir,
    checkpoint_every,
    resume_from=None,
    crash_iteration=-1,
    deterministic=False,
    
):
    # Setup seed to have same model's initialization:
    manual_seed(75)

    train_loader, eval_dataloader  = get_data_loaders(args)
    model = LyftMultiModel()
    criterion = nn.MSELoss()
    model.to(device)  # Move model before creating optimizer

    
    
    writer = SummaryWriter(log_dir)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=1)
    
    def train_step(engine, data):
        model.train()
        optimizer.zero_grad()
        loss, _,_ = forward(data, model, device,criterion)
        loss.backward()
        optimizer.step()
        return loss.item()

    
    trainer = Engine(train_step)
    
#     Apply learning rate scheduling
    @trainer.on(Events.EPOCH_COMPLETED)
    def lr_step(engine):
        lr_scheduler.step()

    desc = "Epoch {} - loss: {:.4f} - lr: {:.4f}"
    pbar = tqdm(initial=0, leave=False, total=epochs*len(train_loader), desc=desc.format(0, 0, lr))

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        lr = optimizer.param_groups[0]["lr"]
        pbar.desc = desc.format(engine.state.epoch, engine.state.output, lr)
        pbar.update(log_interval)
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
        writer.add_scalar("lr", lr, engine.state.iteration)

    
    if resume_from is not None:

        @trainer.on(Events.STARTED)
        def _(engine):
            pbar.n = engine.state.iteration 


    # Compute and log validation metrics
    @trainer.on(Events.ITERATION_COMPLETED(every=checkpoint_every))
    def log_validation_results(engine):
        # ==== EVAL LOOP
        model.eval()
        loss_sum=0
        with torch.no_grad():
            for batch_idx,data in enumerate(eval_dataloader):

                loss,outputs = forward(data, model, device,criterion)
                loss_sum+=loss
        metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])

        avg_nll = loss_sum/len(eval_dataloader)
        tqdm.write(
            "Validation Results - Epoch: {}   loss: {:.4f}".format(
                engine.state.epoch, avg_nll
            )
        )
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.iteration)

    # Setup object to checkpoint
    objects_to_checkpoint = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
    training_checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(log_dir, require_empty=False),
        n_saved=None,
        global_step_transform=lambda *_: trainer.state.iteration,
    )
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=checkpoint_every), training_checkpoint)


    if resume_from is not None:
        tqdm.write("Resume from the checkpoint: {}".format(resume_from))
        checkpoint = torch.load(resume_from)

        
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)

    try:
        # Synchronize random states
        manual_seed(15)
        trainer.run(train_loader, max_epochs=epochs)
    except Exception as e:
        import traceback

        print(traceback.format_exc())

    pbar.close()
    writer.close()



    
    

run(
        100,
        1e-3,
        5,
#         "ignite-log/" + 'train_full_10m_mask_with_nearest_agents_multi_gpu',
        "ignite-log/" + 'train_raster_baseline/',
#         "ignite-log/" + 'test-valid-data',
    
    
#         "/home/mkhorasa/l5kit/examples/agent_motion_prediction/ignite-log/train_baseline_without_agents10/15/20-12:46:07/",
        50,
        
#        model_path,

    )
    
    

