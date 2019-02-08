from func import *
#Basic usage: python predict.py /path/to/image checkpoint
predict_parser = predict_parse_setup()
args = predict_parser.parse_args()
GPU, input_file, checkpoint, top_k, category_names = validate_args(args, 'predict')

model = load_model(checkpoint)
model.eval()

flower_list, prob_list = predict(input_file, model, top_k)
print_probabilities(input_file, flower_list, prob_list)

#im = Image.open(input_file)
#plot_probs(flower_list, prob_list, im)