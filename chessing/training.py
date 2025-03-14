import torch
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard
writer = SummaryWriter(log_dir="runs/chess_training")


def train_model(model, train_loader, num_epochs=10, log_interval=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0  # To keep track of correct predictions
        total_samples = 0  # To keep track of total squares processed

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # Forward pass

            # Flatten labels and outputs for loss calculation
            labels = labels.view(-1)  # Flatten labels to [batch_size*num_squares]
            outputs = outputs.view(
                -1, len(LABELS)
            )  # Flatten outputs to [batch_size*num_squares, num_classes]

            # Compute the loss
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights

            running_loss += loss.item()

            # Calculate accuracy: find the index with max probability (predicted label) and compare with the true label
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()  # Count correct predictions
            running_corrects += correct
            total_samples += labels.size(0)  # Count the total number of labels

            # Log progress every 'log_interval' batches
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_accuracy = running_corrects / total_samples  # Calculate accuracy
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}"
                )
                writer.add_scalar(
                    "Training Loss", avg_loss, epoch * len(train_loader) + batch_idx
                )
                writer.add_scalar(
                    "Training Accuracy",
                    avg_accuracy,
                    epoch * len(train_loader) + batch_idx,
                )
                running_loss = 0.0  # Reset loss tracking
                running_corrects = 0  # Reset correct count
                total_samples = 0  # Reset total samples

        # Log full epoch loss and accuracy
        epoch_accuracy = running_corrects / total_samples
        writer.add_scalar("Epoch Loss", running_loss / len(train_loader), epoch)
        writer.add_scalar("Epoch Accuracy", epoch_accuracy, epoch)
