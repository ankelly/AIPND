from func import *

train_parser = train_parse_setup()
args = train_parser.parse_args()
GPU, arch, learning_rate, save_dir, epochs, hidden_units, data_dir = validate_args(args, 'train')


train_dir, valid_dir, test_dir = data_dir + '/train', data_dir + '/valid', data_dir + '/test'
image_datasets, image_loaders = get_datasets_dataloaders(train_dir, valid_dir, test_dir)

print_parameters(GPU, arch, learning_rate, save_dir, epochs, hidden_units, data_dir)

model = model_setup(arch, GPU, hidden_units, epochs, learning_rate, image_datasets)
model, optimizer, criterion = train_model(model, image_loaders[0], image_loaders[1])

save_model(model, save_dir, optimizer, criterion)