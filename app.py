import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from loaders import load_model
from models import CifarModel
from utils import normalize_img
import numpy as np
from PIL import Image

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
            html.Div(id='output-image-upload', className='flex-item', children=[html.H2('fffff')]),
            html.Div(id='output-prediction', className='flex-item', children=[html.H2('ddddd')]),
        ])


])


def parse_contents(contents):
    return html.Div([
        html.Img(className='img', src=contents),
    ])


model = load_model(CifarModel, 'cifar_model')


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c) for c in list_of_contents]
        return children


def load_and_preprocess(image):
    image1 = Image.open(image)
    rgb = Image.new('RGB', image1.size)
    rgb.paste(image1)
    image = rgb
    test_image = image.resize((64, 64))
    return test_image


@app.callback(Output('output-prediction', 'children'),
              Input('upload-image', 'filename'))
def prediction(img):
    if img is None:
        raise dash.exceptions.PreventUpdate
    img = load_and_preprocess(img)
    scaled_img = normalize_img(img)
    y = model.predict(scaled_img)
    return y


if __name__ == '__main__':
    app.run_server(debug=True)
