import torch
import numpy as np
import segmentation_models_pytorch as smp
import cv2

class LiverTumorModel:
    def __init__(self, model_path, device='cpu'):
        self.device = device

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=3,
            activation='softmax2d'
        )

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_numpy):
        if image_numpy.shape[0] != 512 or image_numpy.shape[1] != 512:
            image_numpy = cv2.resize(image_numpy, (512, 512))

        image = np.clip(image_numpy, -100, 250)
        # min_hu, max_hu = -100.0, 250.0
        # image = (image - min_hu) / (max_hu - min_hu)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        return image_tensor

    def predict(self, image_numpy):
        input_tensor = self.preprocess(image_numpy).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            probs = torch.softmax(output, dim=1)
            tumor_probability_map = probs[0, 2, :, :].cpu().numpy()
            max_prob = tumor_probability_map.max()

        return predicted_mask, max_prob