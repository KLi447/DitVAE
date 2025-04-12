import torch
from models.models import Decoder
import torchvision.transforms as transforms
import clip
import time


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    captions = ["A dog chasing a ball in the grass",
                "A brown dog jumps to catch a frisbee in the park.",
                "A man riding a bicycle along a mountain trail.",
                "Two children are playing with a soccer ball on the grass.",
                "A plate of pancakes topped with fresh berries and syrup.",
                "A cat lying on a windowsill in the sunlight.",
                "A woman holding an umbrella while crossing the street in the rain.",
                "A train traveling through a snowy landscape during sunset."]

    samples = []

    for text in captions:
        start = time.time()

        tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_text(tokens)

        end = time.time()
        emb_time = (end - start) * 1000
        print(f"CLIP time: {emb_time:.3f} ms")

        samples.append(embedding)
    
    samples = torch.cat(samples, dim=0)
    samples = samples.to(torch.float32)
    
    d_model = Decoder(512, 1024)
    state_dict = torch.load('./decoderv4_1500.pt', weights_only=True)
    d_model.load_state_dict(state_dict)
    d_model = torch.compile(d_model)
    d_model.to(device)
    d_model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        d_model.sample(samples, device=device, filename="sampled_from_checkpoint.jpg")

    end_time = time.time()

    inference_time = (end_time - start_time) * 1000
    print(f"Inference time: {inference_time:.3f} ms")
    
