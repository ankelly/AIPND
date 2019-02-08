from my_imports import *

default_gpu = False
default_arch = ['densenet121']
default_learning_rate = ['0.001']
default_save_dir = "save/"
default_epochs = ['15']
default_hidden_units = ['512']
default_data_dir = 'flowers/'
cat_to_name_file = 'cat_to_name.json'
default_k = ['5']

def train_parse_setup():
    train_parser = argparse.ArgumentParser(description='Train a network.')
    train_parser.add_argument('data_dir')
    train_parser.add_argument('--save_dir', nargs=1, default=default_save_dir)
    train_parser.add_argument('--arch', nargs=1, default=default_arch)
    train_parser.add_argument('--learning_rate', nargs=1, default=default_learning_rate)
    train_parser.add_argument('--gpu', action="store_true", default=default_gpu)
    train_parser.add_argument('--hidden_units', nargs=1, default=default_hidden_units)
    train_parser.add_argument('--epochs', nargs=1, default=default_epochs)
    return train_parser

def predict_parse_setup():
    predict_parser = argparse.ArgumentParser(description="Prediction made on pre-trained network.")
    predict_parser.add_argument('input')
    predict_parser.add_argument('checkpoint')
    predict_parser.add_argument('--top_k', nargs=1, default=default_k)
    predict_parser.add_argument('--gpu', action="store_true", default=default_gpu)
    predict_parser.add_argument('--category_names', nargs=1, default=cat_to_name_file)
    return predict_parser


def is_valid_dir(train_dir, valid_dir, test_dir):
    if os.path.isdir(train_dir) & os.path.isdir(valid_dir) & os.path.isdir(test_dir):
        return True
    else:
        return False
    

def validate_args(args, action):
    GPU = args.gpu
    if action == 'train':
        data_dir = args.data_dir
        train_dir, valid_dir, test_dir = data_dir + '/train', data_dir + '/valid', data_dir + '/test'
        if not is_valid_dir(train_dir, valid_dir, test_dir):
            print("data_dir is invalid")
            sys.exit()
        GPU = args.gpu
        save_dir = args.save_dir
        arch = args.arch[0]
        learning_rate = float(args.learning_rate[0])
        hidden_units = int(args.hidden_units[0])
        epochs = int(args.epochs[0])
        return GPU, arch, learning_rate, save_dir, epochs, hidden_units, data_dir
    else:
        input_file = args.input
        if not os.path.isfile(input_file):
            print("Image file is invalid.")
            sys.exit()
        checkpoint = args.checkpoint
        top_k = int(args.top_k[0])
        category_names = args.category_names
        return GPU, input_file, checkpoint, top_k, category_names

def print_parameters(GPU, arch, learning_rate, save_dir, epochs, hidden_units, data_dir):
    print("Starting Model Training with the following settings:")
    print("Data directory: ", data_dir)
    print("Save directory: ", save_dir)
    if GPU:
        print("GPU: on (if available)")
    else:
        print("GPU: off")
    print("Architecture: ", arch)
    print("Epochs: ", epochs)
    print("Hidden Units: ", hidden_units)
    print("Learning Rate: ", learning_rate)
    print("\n\n")
    
def validation(model, dataloader, criterion, device):
    loss = 0
    accuracy = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return loss, accuracy

def get_datasets_dataloaders(train_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),                                       
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    image_datasets = [train_data, validation_data, test_data]

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    image_loaders = [trainloader, validloader, testloader]
    return image_datasets, image_loaders

def model_setup(arch, GPU, hidden_units, epochs, learning_rate, image_datasets):
    with open(cat_to_name_file, 'r') as f:
        class_to_name = json.load(f)
    
    model = getattr(models, arch)(pretrained=True)
    in_feature = int(str(model.classifier).split("in_features=")[1].split(',')[0])
    model.arch = arch
    model.idx_to_class = {v:k for k, v in image_datasets[0].class_to_idx.items()}
    model.class_to_name = class_to_name
    model.learning_rate = float(learning_rate)
    model.epochs = int(epochs)
    model.gpu = GPU
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_feature, hidden_units)),
        ('relu1', nn.ReLU()),
        ('do1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model

def train_model(model, trainloader, validloader):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=model.learning_rate)
    print_every = 40
    steps = 0
    running_loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() & model.gpu else "cpu")
    model.to(device)
     
    for e in range(model.epochs):
        print ("Starting epoch {}".format(e+1))

        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, v_accuracy = validation(model, validloader, criterion, device)
            
                print("Epoch: {}/{}...".format(e+1, model.epochs),
                    "Training Loss: {:0.4f}".format(running_loss/print_every),
                    "Valid Loss: {:.4f}..".format(valid_loss/len(validloader)),
                    "Valid Accuracy: {:.4f}".format(v_accuracy/len(validloader)))
            
                running_loss = 0
                model.train()

    print("Training complete")
    return model, optimizer, criterion

def save_model(model, save_dir, optimizer, criterion):
    checkpoint = {'arch': model.arch,
                  'epochs': model.epochs,
                  'idx_to_class': model.idx_to_class,
                  'class_to_name': model.class_to_name,
                  'learning_rate': model.learning_rate,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optim_state': optimizer.state_dict(),
                  'criterion': criterion}
    
    name = save_dir + "checkpoint_" + str(model.arch) + "_" + str(model.epochs) + ".pth"
    print("Saving to ", name)
    torch.save(checkpoint, name)

def load_model(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.idx_to_class = checkpoint['idx_to_class']
    model.class_to_name = checkpoint['class_to_name']
    model.learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.classifier = checkpoint['classifier']
    model.criterion = checkpoint['criterion']
    return model

def crop_img(im, size):
    w, h = im.size
   
    left = (w - size)//2
    right = (w + size)//2

    top = (h - size)//2
    bottom = (h + size)//2
    
    im = im.crop((left, top, right, bottom))
    return im

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    w, h = im.size
    #Need image to at least be 256x256
    if w > h:
        size = (256*w, 256)
    else:
        size = (256, 256*h)
        
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    sd = [0.229, 0.224, 0.225]

    im.thumbnail(size,Image.ANTIALIAS)
    im = crop_img(im, 224)
    im = np.array(im)/255
    im = (im - mean) / std
    im = im.T
    
    return im

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    prob_list = []
    flower_list = []
    model.eval()
    model.to('cpu')
    img = process_image(image_path)
    
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    ps =  torch.exp(model.forward(img))

    probs, classes = ps.topk(5)

    idx_to_class = model.idx_to_class
    classes = [x for x in classes.data.numpy()[0]]
    probs = [y for y in probs.data.numpy()[0]]
    for n in range(len(classes)):
        flower_list.append(model.class_to_name[idx_to_class[classes[n]]])
        prob_list.append(probs[n])
    return flower_list, prob_list

def print_probabilities(image, flower_list, prob_list):
    print("\nProbabilities for following image: ", image)
    for x in range(len(flower_list)):
        print("{:.4%}\t{}".format(prob_list[x], flower_list[x]))

def plot_probs(flower_list, prob_list, im):
    plt.title(flower_list[0])
    plt.axis('off')
    plt.imshow(im)
   
    fig, ax = plt.subplots()
    y_pos = np.arange(len(flower_list))
    ax.barh(y_pos, prob_list, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(flower_list)
    ax.invert_yaxis()
    plt.show()
