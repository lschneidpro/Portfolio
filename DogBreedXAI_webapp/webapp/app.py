# Setup
import os

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_uploader as du
import plotly.graph_objects as go

from PIL import Image


from python import xai


BS = "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
app = dash.Dash(
    __name__, external_stylesheets=[BS], assets_folder=os.getcwd() + "/assets/static"
)

du.configure_upload(app, "temp", use_upload_id=False)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink(
                "Dog Dataset",
                href="https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip",
            )
        ),
        dbc.NavItem(
            dbc.NavLink("DenseNet-121", href="https://arxiv.org/abs/1608.06993")
        ),
        dbc.NavItem(dbc.NavLink("Grad-CAM", href="https://arxiv.org/abs/1610.02391")),
    ],
    brand="Explainable Artificial Intelligence Webapp",
    brand_href="#",
    className="navbar navbar-expand-lg navbar-dark bg-dark sticky-top",
    fluid=True,
)


app.layout = html.Div(
    [
        # top navbar
        navbar,
        dbc.Row(
            [
                dbc.Col(
                    # left column
                    html.Div(
                        [
                            html.A(
                                [
                                    html.Img(
                                        src=app.get_asset_url("linkedinlogo.png"),
                                        className="img-fluid mb-4 ml-2",
                                    )
                                ],
                                href="https://www.linkedin.com/in/lschneidpro",
                            ),
                            html.A(
                                [
                                    html.Img(
                                        src=app.get_asset_url("githublogo.png"),
                                        className="img-fluid ml-2",
                                    )
                                ],
                                href="https://github.com/lschneidpro",
                            ),
                        ],
                        className="mt-3",
                    ),
                    className="col-1 border-right",
                ),
                # middle
                dbc.Col(
                    [
                        # text area
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.H2(
                                                    "Explainable Dog Breed Classifier"
                                                ),
                                                html.H6(
                                                    "Breed probability from DenseNet-121 and Explanations from Grad-CAM",
                                                    className="text-muted",
                                                ),
                                            ],
                                            className="mt-3",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # nlp area
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                du.Upload(
                                                    id="upload-image",
                                                    text="Drag and Drop or Select File",
                                                    text_completed="Image Analysis of ",
                                                    max_file_size=100,
                                                )
                                            ],
                                            className="container mt-5 text-center",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(id="output-image-raw"),
                                    ],
                                    className="container center-block text-center col-xs-12 col-sm-6",
                                ),
                                dbc.Col(
                                    [
                                        html.Div(id="output-image-XAI"),
                                    ],
                                    className="container center-block text-center col-xs-12 col-sm-6",
                                ),
                            ],
                            className="mt-5",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(className="container col-2"),
                                dbc.Col(
                                    [
                                        html.Div(id="output-graph-class", children=[]),
                                    ],
                                    className="container mt-5 text-center",
                                ),
                                dbc.Col(className="container col-2"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


def parse_image(contents):
    return html.Div(
        [
            html.Img(src=contents, className="img-fluid"),
        ]
    )


def parse_graph(d_probs):

    colors = [
        "#1f77b4",
    ] * 5

    if d_probs["probability"][0] > 0.75:
        colors[0] = "#2ca02c"

    fig = go.Figure(
        data=[
            go.Bar(
                x=d_probs["breed"],
                y=d_probs["probability"],
                text=d_probs["probability"],
                textposition="auto",
                marker_color=colors,  # marker color can be a single color value or an iterable
            )
        ]
    )
    fig.update_traces(texttemplate="%{text:.1%f}")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
    fig.update_layout(yaxis=dict(range=[0, 1]))
    fig.layout.yaxis.tickformat = ",.0%"

    return html.Div([dcc.Graph(figure=fig)])


@app.callback(
    [
        Output("output-image-raw", "children"),
        Output("output-image-XAI", "children"),
        Output("output-graph-class", "children"),
    ],
    [Input("upload-image", "isCompleted")],
    [State("upload-image", "fileNames"), State("upload-image", "upload_id")],
)
def callback_on_completion(iscompleted, filename, upload_id):
    if not iscompleted:
        return None, None, None

    if filename is not None:

        pil_img = Image.open("temp/{}".format(filename[0]))

        # Deep Learning
        pil_raw_img, pil_grad_img, d_probs = xai.get_xai(pil_img)

        # Convert Pil to string
        str_raw_img = xai.pil_to_b64(pil_raw_img)
        str_grad_img = xai.pil_to_b64(pil_grad_img)

        os.remove("temp/{}".format(filename[0]))

        return parse_image(str_raw_img), parse_image(str_grad_img), parse_graph(d_probs)

    return None, None, None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run_server(host="0.0.0.0", port=port)
