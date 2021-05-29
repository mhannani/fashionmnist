import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import datetime

app = dash.Dash(__name__, assets_url_path='assets/')
app.layout = html.Div([
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
            html.Div(id='output-image-upload')],
    ),

    html.Div(id='output-prediction'),
])


# For the image display
def parse_contents(contents):
    return html.Div([
        html.Img(src=contents),
    ])


# For the image display
@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c) for c in list_of_contents]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)
