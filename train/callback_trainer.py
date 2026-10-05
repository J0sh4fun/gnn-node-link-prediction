"""Phase 1 training helpers extracted unchanged from the shared notebook.

See docs/phase1_cleanup.md for task-specific semantic differences.
"""

import csv, math
from pathlib import Path
import torch

class EarlyStopping:
    def __init__(self, mode='min', patience=10, min_delta=0.):
        if mode not in ('min','max') or not isinstance(patience,int) or patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('mode/patience/min_delta không hợp lệ')
        self.mode, self.patience, self.min_delta = mode,patience,min_delta
        self.best, self.best_epoch, self.bad_epochs = None,None,0

    def update(self, score, epoch):
        if not math.isfinite(score):
            raise ValueError('Validation metric không hữu hạn')
        improved = self.best is None or (score < self.best-self.min_delta if self.mode == 'min' else score > self.best+self.min_delta)
        if improved:
            self.best, self.best_epoch, self.bad_epochs = score,epoch,0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience

def fit(model, optimizer, train_input, validation_input, train_loss, validate, config, run_path):
    if config['max_epochs'] < 1:
        raise ValueError('max_epochs phải dương')
    stopper = EarlyStopping(config['mode'],config['patience'],config['min_delta'])
    checkpoint = Path(run_path)/'checkpoints/best.pt'
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    history = []
    for epoch in range(config['max_epochs']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(model,train_input)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError('Loss phải là scalar hữu hạn')
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
            raise ValueError('Gradient thiếu hoặc không hữu hạn')
        optimizer.step()
        model.eval()
        with torch.no_grad():
            metrics = validate(model,validation_input)
        score = float(metrics[config['monitor']])
        improved, stop = stopper.update(score,epoch)
        history.append({'epoch':epoch,'train_loss':float(loss.detach()),config['monitor']:score,'improved':improved})
        if improved:
            torch.save({'model_state':model.state_dict(),'epoch':epoch,'score':score},checkpoint)
        if stop:
            break
    state = torch.load(checkpoint,map_location=next(model.parameters()).device,weights_only=True)
    model.load_state_dict(state['model_state'])
    model.eval()
    with (Path(run_path)/'history.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return {'history':history,'best_epoch':state['epoch'],'best_score':state['score'],'checkpoint':str(checkpoint)}
