

if __name__ == "__main__":
    print("hello yolov5...")



    device = select_device(cfg)
    model = Model(cfg).to(device)