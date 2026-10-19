import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class KrModel(nn.Module):
    def __init__(self, B):
        super(KrModel, self).__init__()
        
        # Image processing branch
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.fc_image = nn.Sequential(
            nn.Linear(32 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
        )
        
        # Velocities, cmd_vel, and angles processing
        self.fc_other_inputs = nn.Linear(6 + 3 + 2, 100)
        
        # Ensemble model
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(150, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 12)
        ) for _ in range(B)])

        self.fc_final = nn.Sequential(
            nn.Linear(6 + 6 + 3, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 25),
            nn.ReLU(),
            nn.Linear(25, 12),
            nn.ReLU(),
            nn.Linear(12, 2),
        ) 

        # Final processing
        # self.fc_final1 = nn.Linear(6 + 6 + 3, 50)
        # self.fc_final2 = nn.Linear(100, 50)
        # self.fc_out = nn.Linear(50, 2)  # Output for roll and pitch
        
    def forward(self, velocities, angles, image, cmd_vel):
        # Process image
        x_image = F.relu(self.conv1(image))
        x_image = self.pool(x_image)
        x_image = F.relu(self.conv2(x_image))
        x_image = x_image.view(x_image.size(0), -1)  # Flatten #Size: 32 * 4 * 4

        x_image = self.fc_image(x_image)

        # x_image = F.relu(self.fc_image1(x_image))
        # x_image = F.relu(self.fc_image2(x_image))
        
        # Process other inputs
        other_inputs = torch.cat((velocities, angles, cmd_vel), dim=1)
        x_other_inputs = F.relu(self.fc_other_inputs(other_inputs))
        
        # Concatenate image and other inputs for ensemble
        x_concat = torch.cat((x_image, x_other_inputs), dim=1)
        
        # Ensemble model
        ensemble_outputs = torch.stack([ensemble(x_concat) for ensemble in self.ensemble], dim=1)
        
        # Mean and std dev extraction from ensemble outputs
        means = ensemble_outputs[..., :6].mean(dim=1)  # Mean of means
        # std_devs = ensemble_outputs[..., 6:].mean(dim=1)  # Mean of std devs
        # Final processing with mean, velocities, and angles
        final_input = torch.cat((means, velocities, angles), dim=1)
        # pdb.set_trace() 
        x_final = self.fc_final(final_input)

        # x_final = F.relu(self.fc_final1(final_input))
        # x_final = F.relu(self.fc_final2(x_final))
        # output = self.fc_out(x_final)
        
        return x_final 
