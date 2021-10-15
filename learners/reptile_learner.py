import torch

# import torch_xla
# import torch_xla.core.xla_model as xm


def reptile_learner(model, queue, optimizer, iteration, args):
    model.train()

    old_vars = [param.data.clone() for param in model.parameters()]
    # running_vars = [
    #     torch.zeros(param.shape, device=device) for param in model.parameters()
    # ]

    queue_length = len(queue)
    losses = 0

    for k in range(args.update_step):
        for i in range(queue_length):
            optimizer.zero_grad()

            support_data = queue[i]["batch"][k]["support"]
            task = queue[i]["task"]

            output = model.forward(task, support_data)
            loss = output[0].mean()

            # loss_cls = criterion(logits, support_labels)
            # loss_cls = loss_cls.mean()
            # loss = loss_cls

            loss.backward()
            losses += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # if args.tpu:
            #     # Optimizer for TPU
            #     xm.optimizer_step(optimizer, barrier=True)
            # else:
            #     # Optimizer for GPU
            #     optimizer.step()
            optimizer.step()

        # for idx, param in enumerate(model.parameters()):
        #     running_vars[idx].data += param.data.clone()
        #     param.data = old_vars[idx].data.clone()

    # for param in running_vars:
    #     param.data /= queue_length

    # for idx, param in enumerate(model.parameters()):
    #     param.data += args.beta * running_vars[idx].data
    #     param.data -= args.beta * old_vars[idx].data

    beta = args.beta * (1 - iteration / args.meta_iteration)
    for idx, param in enumerate(model.parameters()):
        param.data = (1 - beta) * old_vars[idx].data + beta * param.data

    return losses / (queue_length * args.update_step)


def reptile_evaluate(model, dataloader, criterion, device):
    with torch.no_grad():
        total_loss = 0.0
        model.eval()
        for i, data in enumerate(dataloader):

            sample, labels = data
            sample, labels = sample.to(device), labels.to(device)

            logits, _ = model.forward(sample)

            loss = criterion(logits, labels)
            loss = loss.mean()
            total_loss += loss.item()

        total_loss /= len(dataloader)
        return total_loss
