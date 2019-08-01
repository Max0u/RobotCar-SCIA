import md
from keras2graph import keras_to_graph

if __name__ == '__main__':
    try:
        model_path = "../models/model-dashed.h5"
        model = md.build_model()
        model.load_weights(model_path)
        model_in = model.input.name.split(':')[0]
        model_out = model.output.name.split(':')[0]
        graph_path = "./graph"
        print(model_in)
        print(model_out)

    except:
        print('Run with arguments !\nArguments:\nmodel_path model_in model_out graph_path')
        exit()

    keras_to_graph(model, model_in, model_out, graph_path, False)
