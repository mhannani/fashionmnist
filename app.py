import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from loaders import load_model
from models import CifarModel
import numpy as np
import os
from PIL import Image
import datetime
import matplotlib.pyplot as plt
image_directory = 'assets/images/'
app = dash.Dash(__name__, assets_url_path='assets/')
app.layout = html.Div(
    id='app-container',
    children=[
        html.Div(
            id='images_upload_container',
            children=[
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    multiple=True
                ),
            ],
        ),
        html.Div(id='flex-container', children=[
            html.Div(id='output-image-upload', className='flex-item'),
            html.Div(id='output-prediction', className='flex-item', children=[html.H2('ddddd')]),
        ])
    ])


def parse_contents(contents, filename):
    return html.Div([
        html.H5(filename),
        html.Img(src=contents),
    ])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children


def load_and_preprocess(image):
    img = plt.imread(os.path.join(image_directory, image))
    return img


def normalize_img(img):
    """
    Normalize the given images.
    :param img: array_like
    :return: array_like
        Normalized image.
    """
    img = np.array(img)
    return img.astype('float32') / 255.0


model = load_model(CifarModel, 'cifar_model')


@app.callback(Output('output-prediction', 'children'),
              Input('upload-image', 'filename'))
def prediction(img):
    if img is None:
        raise dash.exceptions.PreventUpdate
    else:
        print('prediction image')
        print(img[0])
        img = load_and_preprocess(img[0])
        print(img.shape)
        scaled_img = normalize_img(img)
        y = model.predict(scaled_img[:, :, :3])
    return y


if __name__ == '__main__':
    app.run_server(debug=True)
