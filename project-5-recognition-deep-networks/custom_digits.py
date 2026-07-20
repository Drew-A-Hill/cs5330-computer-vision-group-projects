from PIL import Image                   
from torchvision.transforms import v2       
import torch                                                                                                   
from model import NeuralNetwork

transform = v2.Compose([v2.ToImage(), v2.Resize((28, 28)), v2.ToDtype(torch.float32, scale=True)])
model = NeuralNetwork() 
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()                                 
                                                                                                                
digits = [1, 2, 3, 4, 5, 6, 7, 8, 9]                                                                           
for digit in digits:                                                                                           
    img = Image.open(f"./digits/digit_{digit}_28.jpeg").convert("L")                                         
    img_tensor = transform(img)     
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())   
    img_tensor[img_tensor < 0.6] = 0
    pred = model(img_tensor.unsqueeze(0))
    pred_digit = pred.argmax()       
    values = [round(x, 2) for x in pred.data.tolist()[0]]
    print(values, pred_digit.item(), digit)                                                                 
    
