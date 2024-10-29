import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGCNN(nn.Module):
    def __init__(self, num_classes):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv7 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)  # 5 classes for ECG classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)

        x = self.global_avg_pool(x)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
# Tạo dữ liệu giả lập để kiểm tra mô hình
def generate_dummy_data(batch_size, seq_length):
    # Tạo dữ liệu đầu vào giả lập với batch_size và seq_length
    x = torch.randn(batch_size, 1, seq_length)
    # Tạo nhãn giả lập (5 lớp phân loại)
    y = torch.randint(0, 5, (batch_size,))
    return x, y

# Khởi tạo mô hình
model = ECGCNN(5)

# Tạo dữ liệu giả lập
batch_size = 10
seq_length = 320
x, y = generate_dummy_data(batch_size, seq_length)

# Chạy mô hình với dữ liệu giả lập
output = model(x)
# Chuyển đổi đầu ra thành nhãn dự đoán
_, predicted_labels = torch.max(output, 1)

print("Predicted labels:", predicted_labels)
