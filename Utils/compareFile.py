import torch
import safetensors.torch


def main():
    file1path = "/nfs5/wrz/src/python/SegRefiner/MyRefiner-VAE/output/25-06-22_09-23/checkpoints/best_epoch.pth"
    # file2path = "/nfs5/wrz/.cache/huggingface/hub/models--mit-han-lab--dc-ae-f32c32-sana-1.0/model.safetensors"
    file2path = "/nfs5/wrz/src/python/SegRefiner/MyRefiner-VAE/Modules/tokenizer/Checkpoints/vavae-imagenet256-f16d32-dinov2.pt"
    # file2path = "/home/wrz/src/python/SegRefiner/MyRefiner-VAE/Modules/tokenizer/Checkpoints/vavae-imagenet256-f16d32-dinov2.pt"
    # file2path = "/nfs5/wrz/src/python/Marigold/BASE_CKPT_DIR/models--stabilityai--stable-diffusion-2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79/vae/diffusion_pytorch_model.safetensors"
    # load ckpt
    file1 = torch.load(file1path, map_location='cpu')['vae1_state_dict']
    print(f"len file1: {len(file1)}")
    # safetensors
    # file2 = safetensors.torch.load_file(file2path, device='cpu')
    file2 = torch.load(file2path, map_location='cpu')['state_dict']
    print(f"len file2: {len(file2)}")

    for k in file1.keys():
        if k not in file2:
            print(f"Key {k} is in file1 but not in file2")
        else:
            if not torch.equal(file1[k], file2[k]):
                print(f"Key {k} has different values in file1 and file2")
            else:
                print(f"Key {k} is equal in both files")
    print("\n---------------------\n")
    for k in file2.keys():
        if k not in file1:
            print(f"Key {k} is in file2 but not in file1")
        else:
            if not torch.equal(file2[k], file1[k]):
                print(f"Key {k} has different values in file2 and file1")
            else:
                print(f"Key {k} is equal in both files")


if __name__ == '__main__':
    main()
