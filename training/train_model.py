import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("synthetic_basketball_shots.csv")

X = df[[
    "release_angle",
    "release_speed",
    "release_height",
    "distance",
    "apex_height"
]].values

y_make = df["made"].values
y_angle_corr = df["release_angle"].values - 49.5  # learn correction

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_m_train, y_m_test, y_a_train, y_a_test = train_test_split(
    X, y_make, y_angle_corr, test_size=0.2
)

joblib.dump(scaler, "scaler.pkl")

class ShotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.make_head = nn.Linear(16, 1)
        self.angle_head = nn.Linear(16, 1)

    def forward(self, x):
        f = self.fc(x)
        return self.make_head(f), self.angle_head(f)

model = ShotNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()

X_train = torch.tensor(X_train, dtype=torch.float32)
y_m_train = torch.tensor(y_m_train, dtype=torch.float32).unsqueeze(1)
y_a_train = torch.tensor(y_a_train, dtype=torch.float32).unsqueeze(1)

for epoch in range(30):
    opt.zero_grad()
    pred_make, pred_angle = model(X_train)
    loss = bce(pred_make, y_m_train) + mse(pred_angle, y_a_train)
    loss.backward()
    opt.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "shot_model.pt")
print("✅ Model trained and saved")