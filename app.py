import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from loaders import load_model
from models import CifarModel

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


@app.callback(Output('output-prediction', 'children'),
              Input('upload-image', 'filename'))
def prediction(image):
    final_img = load_and_preprocess(image)
    final_img = np_array_normalise(final_img)
    Y = model.predict(final_img)
    return Y



if __name__ == '__main__':
    app.run_server(debug=True)
