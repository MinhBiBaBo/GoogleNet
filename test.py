"""
Thanh Le  16 April 2024
How to test a trained model with a single input image
"""
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

CLASS = {0: 'Cat', 1: 'Dog'}

if __name__ == "__main__":

    # 1. Create a new deep model ViT
    import torchvision.models as models
    from torchvision.models import ViT_B_16_Weights
    model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT, num_classes=1000)
    #fine tune vit
    num_classes = 2  # Example: for a dataset with 10 classes
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    
    
    #num_features = model.fc.in_features
    #model.fc = torch.nn.Linear(num_features, 2)
    model.to('cuda')

    # 2. Load the weights trained on the Cat-Dog dataset
    #SỬA ĐỂ CHẠY TRÊN CPU
    model.load_state_dict(torch.load('content/drive/My Drive/GoogleNet/checkpoints/epoch_5_acc_0.9633.pt', 'cuda'))
    #model.load_state_dict(torch.load('checkpoints/epoch_8_acc_0.9750.pt', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    # 3. Load an input image
    img = Image.open('dataset/cat_dog/test/dogs/dog_521.jpg')
    plt.imshow(img)
    plt.show()

    # 4. Resize and convert the image to a tensor
    img = T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR)(img)
    #SỬA ĐỂ CHẠY TRÊN CPU
    img = T.ToTensor()(img).to('cuda' , dtype=torch.float).unsqueeze(dim=0)  # expand along the first dimension
    #img = T.ToTensor()(img).to('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float).unsqueeze(dim=0)  # expand along the first dimension

    # 5. Perform a forward pass
    logits = model(img)

    # 6. Get the prediction
    prediction = logits.argmax(axis=1).item()
    print(f'This is a {CLASS[prediction]}')
