def __call__(self, val_loss, model, path):
    # 根据状态调用save_checkpoint保存模型参数
    score = -val_loss
    if self.best_score is None:
        self.best_score = score
        self.save_checkpoint(val_loss, model, path)


def save_checkpoint(self, val_loss, model, path):
    # 通过torch.save(model.state_dict(), path+'/'+'checkpoint.pth')将当前训练轮次的参数保存下来
    if self.verbose:
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
    self.val_loss_min = val_loss


checkpoint_path = 'path/to/checkpoint.pth'
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# 继续训练就行
model.train()  # 确保模型处于训练模式
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
